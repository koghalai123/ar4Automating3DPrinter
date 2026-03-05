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

    def __init__(self,
                 source="ros",
                 camera_index=None,
                 camera_keyword="GENERAL WEBCAM",
                 color_topic="/rgbd_camera/image",
                 depth_topic="/rgbd_camera/depth_image",
                 camera_info_topic="/rgbd_camera/camera_info"):
        super().__init__('aruco_detection_viewer', enable_pose_print=False)

        self.fps = 30.0
        self.markerNamePrefix = "aruco_marker_"
        self.filterStates = np.zeros((100, 6))

        # Default dt in case joint_states hasn't arrived yet
        if not hasattr(self, 'dt') or self.dt is None:
            self.dt = 1.0 / self.fps

        # Single object handles web server + aruco detection + frame compositing.
        # Pass through source / camera options so caller can choose "ros" or "webcam"
        self.stream = WebVideoStream(
            source=source,
            port=5000,
            fps=self.fps,
            display_scale=2.0,
            depth_colormap="turbo",
            marker_sizes=[0.03, 0.05],
            dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
            enrich_fn=self._enrich_marker_pose,
            log_fn=lambda msg: self.get_logger().info(msg),
            camera_index=camera_index,
            camera_keyword=camera_keyword,
            color_topic=color_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
        )

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, self)

        # Togglable 1-second timer for printing marker_poses
        self._marker_print_timer = self.create_timer(1.0, self._print_marker_poses)
        self._marker_print_enabled = False
        self._marker_print_timer.cancel()

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
        # Guard: if dt is not yet set by PoseReader, use default
        if not hasattr(self, 'dt') or self.dt is None or self.dt == 0:
            self.dt = 1.0 / self.fps

        try:
            badPos, badEuler = self.applyFrameChange(posInFrame, eulerInFrame,
                                                     source_frame="base_link", target_frame="ee_camera_link")
        except Exception:
            return None, None

        # Low-pass filter
        fCutoff = 3.0
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

    # ---- Marker pose printing ----

    def enable_marker_print(self):
        """Start printing marker_poses to console every second."""
        if not self._marker_print_enabled:
            self._marker_print_enabled = True
            self._marker_print_timer.reset()
            self.get_logger().info('Marker pose printing enabled')

    def disable_marker_print(self):
        """Stop printing marker_poses to console."""
        if self._marker_print_enabled:
            self._marker_print_enabled = False
            self._marker_print_timer.cancel()
            self.get_logger().info('Marker pose printing disabled')

    def toggle_marker_print(self):
        """Toggle marker_poses console printing on/off."""
        if self._marker_print_enabled:
            self.disable_marker_print()
        else:
            self.enable_marker_print()

    def _print_marker_poses(self):
        poses = self.marker_poses
        if not poses:
            self.get_logger().info('[marker_poses] No markers seen yet')
            return
        for m in poses:
            gp = m.get('global_pose')
            if gp:
                pos = gp['position']
                ori = gp['orientation']
                self.get_logger().info(
                    f"[marker_poses] ID:{m['id']}  dict:{m['dict_name']}  "
                    f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                    f"rpy=({ori['roll']:+.1f}, {ori['pitch']:+.1f}, {ori['yaw']:+.1f})")
            else:
                self.get_logger().info(
                    f"[marker_poses] ID:{m['id']}  dict:{m['dict_name']}  global_pose: N/A")

    @property
    def marker_poses(self):
        """Persistent list of all markers ever detected, with id, dict size, and global pose."""
        result = []
        for entry in self.stream.marker_poses:
            item = dict(entry)
            item['dict_name'] = entry.get('dict_name', 'unknown')
            if 'positionInWorld' in entry and 'orientInWorld' in entry:
                item['global_pose'] = {
                    'position': entry['positionInWorld'],
                    'orientation': entry['orientInWorld'],
                }
            result.append(item)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionViewer()
    node.enable_marker_print()
    printer = Simulated3DPrinter(
        node=node,
        pos=[0.0, -0.67, 0.38],
        orient=[0.0, 0.0, np.pi],
    )
    printer.spawn_fast()
    
    

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