#!/usr/bin/env python3

import rclpy
import numpy as np
import subprocess
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.time import Time
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from webVideoServer import WebVideoStream
from poseReader import PoseReader
from simulated3DPrinter import Simulated3DPrinter


class ArucoDetectionViewer(PoseReader):

    def __init__(self):
        super().__init__('aruco_detection_viewer', enable_pose_print=False)

        self.fps = 30.0
        self.markerNamePrefix = "aruco_marker_"
        self.filterStates = np.zeros((100, 6))

        # Single object handles web server + aruco detection + frame compositing.
        # We pass source="none" so it doesn't create its own ROS node or webcam —
        # this class owns the ROS subscriptions and feeds frames manually.
        self.stream = WebVideoStream(
            source="ros",
            port=5000,
            fps=self.fps,
            display_scale=2.0,
            depth_colormap="turbo",
            marker_sizes=[0.03, 0.05],
            dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
            enrich_fn=self._enrich_marker_pose,
            log_fn=lambda msg: self.get_logger().info(msg),
            color_topic="/rgbd_camera/image",
            depth_topic="/rgbd_camera/depth_image",
            camera_info_topic="/rgbd_camera/camera_info",
        )

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, self)

    # ---- Marker enrichment ----

    def _enrich_marker_pose(self, entry: dict) -> dict:
        pos, euler = self.cameraToBase(entry['positionFromCamera'], entry['eulerFromCamera'],
                                       markerID=entry['id'])
        if pos is None:
            return None
        entry['tf2Name'] = f"{self.markerNamePrefix}{entry['id']}"
        entry['positionInWorld'] = pos
        entry['orientInWorld'] = {'roll': np.degrees(euler[0]), 'pitch': np.degrees(euler[1]), 'yaw': np.degrees(euler[2])}
        return entry

    # ---- Frame transforms ----

    def applyFrameChange(self, posInFrame, eulerInFrame,
                         source_frame="base_link", target_frame="ee_camera_link"):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(posInFrame[0]), float(posInFrame[1]), float(posInFrame[2])
        q = R.from_euler("XYZ", eulerInFrame, degrees=False).as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        transform = self.tf2_buffer.lookup_transform(source_frame, target_frame, Time())
        transformed = do_transform_pose(pose, transform)

        tf2_quat = [transformed.orientation.x, transformed.orientation.y,
                     transformed.orientation.z, transformed.orientation.w]
        return (np.array([transformed.position.x, transformed.position.y, transformed.position.z]),
                R.from_quat(tf2_quat).as_euler("XYZ", degrees=False))

    def cameraToBase(self, posInFrame, eulerInFrame, markerID=0):
        try:
            badPos, badEuler = self.applyFrameChange(posInFrame, eulerInFrame,
                                                     source_frame="base_link", target_frame="ee_camera_link")
        except Exception:
            return None, None

        # Low-pass filter
        fCutoff = 1.0
        RC = 1 / (2 * np.pi * fCutoff)
        alpha = self.dt / (RC + self.dt)
        prev = self.filterStates[markerID, :]
        filteredPos = alpha * badPos + (1 - alpha) * prev[0:3]
        filteredEuler = alpha * badEuler + (1 - alpha) * prev[3:6]
        self.filterStates[markerID, :] = np.hstack((filteredPos, filteredEuler))

        goodPos, goodEuler = self.to_good_frame(filteredPos, filteredEuler)

        self.broadcast_marker_transform(badPos, badEuler,
                                        child_frame=f"{self.markerNamePrefix}{markerID}")
        return goodPos, goodEuler

    def broadcast_marker_transform(self, marker_pos, marker_orient,
                                   parent_frame="base_link", child_frame="aruco_marker"):
        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])
        q = R.from_euler("XYZ", marker_orient, degrees=False).as_quat()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.tf2_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    

    printer = Simulated3DPrinter([0, 0, 0], [0, 0, 0])
    node = ArucoDetectionViewer()
    

    node.move_to_pose(np.array([0.4,0.0,0.4]), np.array([0.0,0.0,0.0]))
    # Spin both the ROS node (PoseReader/tf2) and the stream's internal ROS node
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.stream._ros_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.stream._ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()