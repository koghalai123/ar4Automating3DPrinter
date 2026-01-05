#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply, quaternion_inverse
import tf2_ros
import time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped
import math

# Reuse your existing servo implementation
from servo import AR4ServoPID

def clamp(v, lo, hi):
    return np.maximum(np.minimum(v, hi), lo)

class VisualServoNode(Node):
    def __init__(self):
        super().__init__("visual_servo_node")

        # Frames
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "ee_camera_optical_frame")
        self.declare_parameter("target_frame", "target")  # Replace with SLAM/fiducial frame later
        # Optional: publish static TFs (camera mount + demo target)
        self.declare_parameter("publish_static_tfs", True)
        # Prefer sim time so TF timestamps align with Gazebo
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)

        # Servo parameters
        self.declare_parameter("pos_gain", 0.5)      # scale of position error -> commanded delta
        self.declare_parameter("rot_gain", 0.5)      # scale of orientation error -> commanded delta
        self.declare_parameter("pos_step_max", 0.01) # m per step
        self.declare_parameter("rot_step_max_deg", 5.0) # deg per step
        self.declare_parameter("rate_hz", 2.0)       # control rate

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value

        self.pos_gain = float(self.get_parameter("pos_gain").value)
        self.rot_gain = float(self.get_parameter("rot_gain").value)
        self.pos_step_max = float(self.get_parameter("pos_step_max").value)
        self.rot_step_max = math.radians(float(self.get_parameter("rot_step_max_deg").value))
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        # Apply use_sim_time immediately so TF listens to /clock (force ON)
        use_sim_time = True
        try:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)])
        except Exception:
            pass

        # TF
        self.tf_buffer = tf2_ros.Buffer()  # simpler ctor
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._last_tf_warn = 0.0

        # Optionally publish static TFs (helpful when not running separate publishers)
        self.publish_static_tfs = bool(self.get_parameter("publish_static_tfs").value)
        if self.publish_static_tfs:
            self._publish_static_tfs()

        # Your servo driver
        self.servo = AR4ServoPID()

        # Control timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_step)

        self.get_logger().info(f"Visual servo running. base={self.base_frame}, cam={self.camera_frame}, target={self.target_frame}, use_sim_time={use_sim_time}, publish_static_tfs={self.publish_static_tfs}")

    def _control_step(self):
        # Request latest available transform (Time() == 0)
        now = rclpy.time.Time()

        # Wait until transforms are available
        has_cam = self.tf_buffer.can_transform(self.base_frame, self.camera_frame, now, timeout=Duration(seconds=0.2))
        has_tgt = self.tf_buffer.can_transform(self.base_frame, self.target_frame, now, timeout=Duration(seconds=0.2))
        if not (has_cam and has_tgt):
            t = time.time()
            if t - self._last_tf_warn > 2.0:
                self.get_logger().warn(f"Waiting for TF frames: base={self.base_frame}, cam={self.camera_frame}, target={self.target_frame}")
                self._last_tf_warn = t
            return

        try:
            t_base_cam = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, now, timeout=Duration(seconds=0.5))
            t_base_tgt = self.tf_buffer.lookup_transform(self.base_frame, self.target_frame, now, timeout=Duration(seconds=0.5))
        except Exception as ex:
            t = time.time()
            if t - self._last_tf_warn > 2.0:
                self.get_logger().warn(f"TF lookup failed: {ex}")
                self._last_tf_warn = t
            return

        # Positions
        p_cam = np.array([t_base_cam.transform.translation.x,
                          t_base_cam.transform.translation.y,
                          t_base_cam.transform.translation.z], dtype=float)
        p_tgt = np.array([t_base_tgt.transform.translation.x,
                          t_base_tgt.transform.translation.y,
                          t_base_tgt.transform.translation.z], dtype=float)
        e_pos = p_tgt - p_cam  # in base frame

        # Orientations (base-frame quaternions)
        q_cam = np.array([t_base_cam.transform.rotation.x,
                          t_base_cam.transform.rotation.y,
                          t_base_cam.transform.rotation.z,
                          t_base_cam.transform.rotation.w], dtype=float)
        q_tgt = np.array([t_base_tgt.transform.rotation.x,
                          t_base_tgt.transform.rotation.y,
                          t_base_tgt.transform.rotation.z,
                          t_base_tgt.transform.rotation.w], dtype=float)

        # Relative rotation from camera to target (expressed in base): q_err = q_cam*^-1 * q_tgt
        q_err = quaternion_multiply(quaternion_inverse(q_cam), q_tgt)
        rpy_err = np.array(euler_from_quaternion(q_err), dtype=float)

        # Simple P control with step limits
        dpos_cmd = clamp(self.pos_gain * e_pos, -self.pos_step_max*np.ones(3), self.pos_step_max*np.ones(3))
        drot_cmd = clamp(self.rot_gain * rpy_err, -self.rot_step_max*np.ones(3), self.rot_step_max*np.ones(3))

        # Send a small servo step
        ok = self.servo.servo_to_delta(
            delta_pos=dpos_cmd,
            delta_rpy=drot_cmd,
            rate_hz=30.0,
            max_duration=0.8,   # short blocking step
            pos_tol=5e-4,
            rot_tol=2e-3,
        )
        if not ok:
            self.get_logger().warn("Servo step failed")

    def _publish_static_tfs(self):
        stf = tf2_ros.StaticTransformBroadcaster(self)

        def make_stamped(parent, child, xyz, rpy):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = xyz[0]
            t.transform.translation.y = xyz[1]
            t.transform.translation.z = xyz[2]
            q = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            return t

        # link_6 -> ee_camera_link (adjust as needed)
        t1 = make_stamped("link_6", "ee_camera_link", [0.0, 0.0, 0.08], [0.0, 0.0, 0.0])
        # ee_camera_link -> ee_camera_optical_frame (standard optical frame rotation)
        t2 = make_stamped("ee_camera_link", "ee_camera_optical_frame", [0.0, 0.0, 0.0], [-math.pi/2, 0.0, -math.pi/2])
        # base_link -> target (demo target)
        t3 = make_stamped("base_link", "target", [0.2, 0.0, 0.3], [0.0, 0.0, 0.0])

        stf.sendTransform([t1, t2, t3])
        self.get_logger().info("Published static camera and target TFs from visualServo node")

def main():
    rclpy.init()
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()