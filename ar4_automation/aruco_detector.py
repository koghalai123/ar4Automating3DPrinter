#!/usr/bin/env python3

import rclpy
import numpy as np
import subprocess
import threading
import time as _time
import warnings
import json
import pathlib
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
from rclpy.time import Time
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_geometry_msgs import do_transform_pose

from .web_video_server import WebVideoStream
from .pose_reader import PoseReader
from .simulated3DPrinter import Simulated3DPrinter


class ArucoDetectionViewer(PoseReader):

    def __init__(self,
                 source="ros",
                 camera_index=None,
                 camera_keyword="GENERAL WEBCAM",
                 color_topic=None,
                 depth_topic=None,
                 camera_info_topic=None,
                 feed_rotation_deg=0.0,
                 marker_sizes=None,
                 calibration_file=None,
                 hand_eye_file=None,
                 robot='ar4'):
        super().__init__('aruco_detection_viewer', enable_pose_print=False, robot=robot)

        # camera frame + topic defaults come from the robot config
        self.camera_frame_name = self.robot_config['camera_frame']
        self.hand_eye_transform = None
        self.hand_eye_file = None
        if robot in {'lite6', 'xarm6'}:
            default_hand_eye = (pathlib.Path(__file__).resolve().parents[1] /
                                'calibration' / f'{robot}_hand_eye.json')
            path = pathlib.Path(hand_eye_file) if hand_eye_file else default_hand_eye
            if path.is_file():
                try:
                    payload = json.loads(path.read_text())
                    if payload.get('quality_passed') is not True:
                        raise ValueError('calibration did not pass quality checks')
                    metrics = payload.get('metrics') or {}
                    if (float(metrics.get('translation_rmse_m', float('inf'))) > 0.01 or
                            float(metrics.get('rotation_rmse_deg', float('inf'))) > 3.0):
                        raise ValueError('calibration quality metrics exceed limits')
                    if payload.get('parent_frame') != self.end_effector_name:
                        raise ValueError(
                            f"parent_frame={payload.get('parent_frame')!r}, expected "
                            f"{self.end_effector_name!r}")
                    tf = np.asarray(payload['matrix'], dtype=float)
                    if tf.shape != (4, 4) or not np.isfinite(tf).all():
                        raise ValueError('matrix must be a finite 4x4 transform')
                    self.hand_eye_transform = tf
                    self.hand_eye_file = str(path)
                    self.get_logger().info(
                        f"Using measured hand-eye calibration: {path}")
                except Exception as exc:
                    self.get_logger().error(
                        f"Ignoring invalid hand-eye calibration {path}: {exc}")
        color_topic = color_topic or self.robot_config['color_topic']
        depth_topic = depth_topic or self.robot_config['depth_topic']
        camera_info_topic = camera_info_topic or self.robot_config['camera_info_topic']

        self.fps = 30.0
        self.markerNamePrefix = "aruco_marker_"
        self.filterStates = np.zeros((100, 7))

        # Default dt in case joint_states hasn't arrived yet
        if not hasattr(self, 'dt') or self.dt is None:
            self.dt = 1.0 / self.fps

        # Single object handles web server + aruco detection + frame compositing.
        # Pass through source / camera options so caller can choose "ros" or "webcam"
        if marker_sizes is None:
            marker_sizes = [0.03, 0.05]
        self.stream = WebVideoStream(
            source=source,
            port=5000,
            fps=self.fps,
            display_scale=2.0,
            depth_colormap="turbo",
            marker_sizes=marker_sizes,
            dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
            enrich_fn=self._enrich_marker_pose,
            log_fn=lambda msg: None,  # suppress per-frame detection noise
            camera_index=camera_index,
            camera_keyword=camera_keyword,
            color_topic=color_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            feed_rotation_deg=feed_rotation_deg,
            calibration_file=calibration_file,
        )

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, self)
        self.tf2_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # RViz visualization publisher
        self._marker_array_pub = self.create_publisher(MarkerArray, '/aruco_markers_viz', 10)
        self._rviz_publish_thread = threading.Thread(target=self._rviz_publish_loop, daemon=True)
        self._rviz_publish_thread.start()

        # Togglable 1-second timer for printing marker_poses
        self._marker_print_timer = self.create_timer(1.0, self._print_marker_poses)
        self._marker_print_enabled = False
        self._marker_print_timer.cancel()

    # ---- Marker enrichment ----

    def _enrich_marker_pose(self, entry: dict) -> dict:
        # Skip estimated markers only if no real camera data has arrived yet
        if entry.get('estimated') and np.allclose(entry.get('positionFromCamera', [0.0, 0.0, 0.0]), 0.0):
            return entry
        # Clear the estimated flag now that we have a real detection
        entry.pop('estimated', None)
        badPos, badEuler = self.cameraToBase(
            entry['positionFromCamera'], entry['eulerFromCamera'],
            markerID=entry['id'], camera_frame=entry.get('camera_frame'))
        if badPos is None:
            self.get_logger().warn(
                f"[enrich] ID={entry['id']} cameraToBase FAILED — "
                f"TF lookup returned None. found_markers will NOT be updated with real pose."
            )
            return None
        entry['tf2Name'] = f"{self.markerNamePrefix}{entry['id']}"

        # Store base_link frame values (from TF2) for RViz and TF operations
        entry['positionInBase'] = badPos
        entry['eulerInBase'] = badEuler

        # Compute user-facing "world" values using frame rotation only (no EEF offset angles)
        R_BF_GF = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
        goodPos = R_BF_GF.apply(badPos)
        goodEuler = (R_BF_GF * R.from_euler("XYZ", badEuler, degrees=False)).as_euler("XYZ", degrees=False)
        entry['positionInWorld'] = goodPos
        entry['orientInWorld'] = {'roll': np.degrees(goodEuler[0]), 'pitch': np.degrees(goodEuler[1]), 'yaw': np.degrees(goodEuler[2])}
        return entry

    # ---- Frame transforms ----

    def applyFrameChange(self, posInFrame, eulerInFrame,
                         source_frame=None, target_frame=None):
        source_frame = source_frame or self.base_link_name
        target_frame = target_frame or self.camera_frame_name
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(posInFrame[0]), float(posInFrame[1]), float(posInFrame[2])
        q = R.from_euler("XYZ", eulerInFrame, degrees=False).as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        transform = self.tf2_buffer.lookup_transform(source_frame, target_frame, Time())
        transformed = do_transform_pose(pose, transform)

        tf2_quat = [transformed.orientation.x, transformed.orientation.y,
                     transformed.orientation.z, transformed.orientation.w]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            euler = R.from_quat(tf2_quat).as_euler("XYZ", degrees=False)
        return (np.array([transformed.position.x, transformed.position.y, transformed.position.z]),
                euler)

    def cameraToBase(self, posInFrame, eulerInFrame, markerID=0,
                     camera_frame=None):
        # Guard: if dt is not yet set by PoseReader, use default
        if not hasattr(self, 'dt') or self.dt is None or self.dt == 0:
            self.dt = 1.0 / self.fps

        try:
            if self.hand_eye_transform is not None:
                base_to_eef = np.eye(4)
                # xArm's robot_state_publisher may retain an old /joint_states
                # transform during teach mode.  Map detections with FK from the
                # driver's direct joint feedback instead.  Other robots retain
                # the standard TF path.
                measured_pos = measured_quat = None
                if self.robot == 'xarm6':
                    measured_pos, measured_quat, _source = (
                        self._eef_pose_from_measured_joints())
                if measured_pos is not None and measured_quat is not None:
                    base_to_eef[:3, :3] = R.from_quat(measured_quat).as_matrix()
                    base_to_eef[:3, 3] = measured_pos
                else:
                    base_to_eef_msg = self.tf2_buffer.lookup_transform(
                        self.base_link_name, self.end_effector_name, Time())
                    t = base_to_eef_msg.transform.translation
                    q = base_to_eef_msg.transform.rotation
                    base_to_eef[:3, :3] = R.from_quat(
                        [q.x, q.y, q.z, q.w]).as_matrix()
                    base_to_eef[:3, 3] = [t.x, t.y, t.z]
                camera_to_marker = np.eye(4)
                camera_to_marker[:3, :3] = R.from_euler(
                    'XYZ', eulerInFrame, degrees=False).as_matrix()
                camera_to_marker[:3, 3] = np.asarray(posInFrame, dtype=float)
                base_to_marker = (base_to_eef @ self.hand_eye_transform @
                                  camera_to_marker)
                badPos = base_to_marker[:3, 3]
                badEuler = R.from_matrix(base_to_marker[:3, :3]).as_euler(
                    'XYZ', degrees=False)
            else:
                badPos, badEuler = self.applyFrameChange(
                    posInFrame, eulerInFrame,
                    source_frame=self.base_link_name,
                    target_frame=camera_frame or self.camera_frame_name)
        except Exception as e:
            # expected while the TF buffer fills at startup; throttled so a
            # persistent TF problem still shows
            self.get_logger().warn(f"cameraToBase: TF transform failed: {e}",
                                   throttle_duration_sec=5.0)
            return None, None

        # Low-pass filter (position: linear lerp, orientation: quaternion SLERP)
        fCutoff = 0.3
        RC = 1 / (2 * np.pi * fCutoff)
        alpha = self.dt / (RC + self.dt)
        prev = self.filterStates[markerID, :]
        # filterStates layout: [x, y, z, qx, qy, qz, qw]
        q_new = R.from_euler("XYZ", badEuler, degrees=False).as_quat()  # [x,y,z,w]
        if np.allclose(prev, 0.0):
            filteredPos = badPos
            filteredQuat = q_new
        else:
            filteredPos = alpha * badPos + (1 - alpha) * prev[0:3]
            q_prev = prev[3:7]
            # Ensure shortest-path interpolation (flip if dot product is negative)
            if np.dot(q_prev, q_new) < 0:
                q_new = -q_new
            q_interp = R.from_quat(q_prev) * R.from_rotvec(
                alpha * (R.from_quat(q_prev).inv() * R.from_quat(q_new)).as_rotvec()
            )
            filteredQuat = q_interp.as_quat()
        self.filterStates[markerID, :7] = np.hstack((filteredPos, filteredQuat))
        filteredEuler = R.from_quat(filteredQuat).as_euler("XYZ", degrees=False)

        # Notify subclasses of the raw (pre-filter) measurement every frame
        try:
            q_raw = R.from_euler("XYZ", badEuler, degrees=False).as_quat()
            q_cam = R.from_euler("XYZ", eulerInFrame, degrees=False).as_quat()
            self._on_raw_marker_measurement(markerID, badPos, q_raw, posInFrame, q_cam)
        except Exception as e:
            self.get_logger().warn(f"raw-measurement hook failed: {e}",
                                   throttle_duration_sec=5.0)

        try:
            self.broadcast_marker_transform(filteredPos, filteredEuler,
                                            child_frame=f"{self.markerNamePrefix}{markerID}")
        except Exception as e:
            # TF broadcast failure is non-fatal
            self.get_logger().warn(f"marker TF broadcast failed: {e}",
                                   throttle_duration_sec=5.0)
        return filteredPos, filteredEuler

    def _on_raw_marker_measurement(self, marker_id, pos_in_base, quat_in_base,
                                    pos_in_camera, quat_in_camera):
        """Hook called with each raw (pre-filter) camera detection. Override in subclasses."""
        pass

    def broadcast_marker_transform(self, marker_pos, marker_orient,
                                   parent_frame=None, child_frame="aruco_marker"):
        parent_frame = parent_frame or self.base_link_name
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

    # ---- RViz marker visualization ----

    def _rviz_publish_loop(self):
        """Background thread that publishes RViz markers at ~10 Hz."""
        while rclpy.ok():
            try:
                self._publish_rviz_markers_impl()
            except Exception as e:
                self.get_logger().warn(f'RViz marker publish failed: {e}', throttle_duration_sec=5.0)
            _time.sleep(0.1)

    def _publish_rviz_markers_impl(self):
        """Split out so the wrapper can catch exceptions."""
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()
        next_id = 0  # global unique id across all marker sub-parts
        frame_id = self.base_link_name  # RViz resolves to display frame via TF

        for entry in self.stream.marker_poses:
            if 'positionInBase' not in entry or 'eulerInBase' not in entry:
                continue

            # base_link values directly; RViz transforms via TF for display
            pos = entry['positionInBase']
            euler_rad = entry['eulerInBase']
            q = R.from_euler('XYZ', euler_rad).as_quat()
            rot = R.from_euler('XYZ', euler_rad)
            size = entry.get('marker_size', 0.05)
            marker_id = int(entry['id'])
            dict_name = entry.get('dict_name', 'unknown')
            # Extract short type like "4x4" from "DICT_4X4_50"
            dict_short = dict_name.replace('DICT_', '').rsplit('_', 1)[0].replace('X', 'x')

            # --- Flat rectangular prism ---
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = 'aruco_body'
            m.id = next_id; next_id += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.x = float(q[0])
            m.pose.orientation.y = float(q[1])
            m.pose.orientation.z = float(q[2])
            m.pose.orientation.w = float(q[3])
            m.scale.x = float(size)
            m.scale.y = float(size)
            m.scale.z = 0.002
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            msg.markers.append(m)

            thickness = 0.002
            # Local Z axis of the marker = normal to front face
            normal = rot.apply([0.0, 0.0, 1.0])
            front_offset = normal * (thickness / 2.0 + 0.001)
            back_offset = -front_offset

            # --- "front" text on front face ---
            ft = Marker()
            ft.header.stamp = now
            ft.header.frame_id = frame_id
            ft.ns = 'aruco_front'
            ft.id = next_id; next_id += 1
            ft.type = Marker.TEXT_VIEW_FACING
            ft.action = Marker.ADD
            ft.pose.position.x = float(pos[0] + front_offset[0])
            ft.pose.position.y = float(pos[1] + front_offset[1])
            ft.pose.position.z = float(pos[2] + front_offset[2])
            ft.scale.z = float(size * 0.3)
            ft.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            ft.text = 'front'
            msg.markers.append(ft)

            # --- "back" text on back face ---
            bt = Marker()
            bt.header.stamp = now
            bt.header.frame_id = frame_id
            bt.ns = 'aruco_back'
            bt.id = next_id; next_id += 1
            bt.type = Marker.TEXT_VIEW_FACING
            bt.action = Marker.ADD
            bt.pose.position.x = float(pos[0] + back_offset[0])
            bt.pose.position.y = float(pos[1] + back_offset[1])
            bt.pose.position.z = float(pos[2] + back_offset[2])
            bt.scale.z = float(size * 0.3)
            bt.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            bt.text = 'back'
            msg.markers.append(bt)

            # --- ID + type label above marker ---
            lt = Marker()
            lt.header.stamp = now
            lt.header.frame_id = frame_id
            lt.ns = 'aruco_labels'
            lt.id = next_id; next_id += 1
            lt.type = Marker.TEXT_VIEW_FACING
            lt.action = Marker.ADD
            lt.pose.position.x = float(pos[0])
            lt.pose.position.y = float(pos[1])
            lt.pose.position.z = float(pos[2]) + float(size) + 0.02
            lt.scale.z = 0.03
            lt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lt.text = f'ID:{marker_id} ({dict_short})'
            msg.markers.append(lt)

            # --- Debug: show raw coordinates used for this marker ---
            euler_deg = np.degrees(euler_rad)
            debug_lines = [
                f'p=[{pos[0]:.3g},{pos[1]:.3g},{pos[2]:.3g}]',
                f'[{euler_deg[0]:.3g},{euler_deg[1]:.3g},{euler_deg[2]:.3g}]',
                f'q=[{q[0]:.3g},{q[1]:.3g},{q[2]:.3g},{q[3]:.3g}]',
            ]
            line_spacing = 0.018
            base_z = float(pos[2]) - float(size) - 0.01
            for i, text_line in enumerate(debug_lines):
                dbg = Marker()
                dbg.header.stamp = now
                dbg.header.frame_id = frame_id
                dbg.ns = 'aruco_debug'
                dbg.id = next_id; next_id += 1
                dbg.type = Marker.TEXT_VIEW_FACING
                dbg.action = Marker.ADD
                dbg.pose.position.x = float(pos[0])
                dbg.pose.position.y = float(pos[1])
                dbg.pose.position.z = base_z - i * line_spacing
                dbg.scale.z = 0.01
                dbg.color = ColorRGBA(r=0.8, g=0.8, b=1.0, a=1.0)
                dbg.text = text_line
                msg.markers.append(dbg)

            # --- Coordinate axes (X=red, Y=green, Z=blue) ---
            axis_len = float(size * 0.8)
            axis_radius = 0.003
            axes = [
                ([1, 0, 0], ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)),  # X
                ([0, 1, 0], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)),  # Y
                ([0, 0, 1], ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)),  # Z
            ]
            for local_dir, color in axes:
                world_dir = rot.apply(local_dir)
                # Arrow start = marker center, end = center + axis_len * direction
                ax = Marker()
                ax.header.stamp = now
                ax.header.frame_id = frame_id
                ax.ns = 'aruco_axes'
                ax.id = next_id; next_id += 1
                ax.type = Marker.ARROW
                ax.action = Marker.ADD
                # Use start/end points
                start = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                end = Point(
                    x=float(pos[0] + world_dir[0] * axis_len),
                    y=float(pos[1] + world_dir[1] * axis_len),
                    z=float(pos[2] + world_dir[2] * axis_len),
                )
                ax.points = [start, end]
                ax.scale.x = axis_radius * 2   # shaft diameter
                ax.scale.y = axis_radius * 4   # head diameter
                ax.scale.z = 0.0               # auto head length
                ax.color = color
                msg.markers.append(ax)

        # --- Camera pose cube ---
        try:
            cam_tf = self.tf2_buffer.lookup_transform(self.base_link_name, self.camera_frame_name, Time())
            cam = Marker()
            cam.header.stamp = now
            cam.header.frame_id = frame_id
            cam.ns = 'camera_pose'
            cam.id = next_id; next_id += 1
            cam.type = Marker.CUBE
            cam.action = Marker.ADD
            cam.pose.position.x = cam_tf.transform.translation.x
            cam.pose.position.y = cam_tf.transform.translation.y
            cam.pose.position.z = cam_tf.transform.translation.z
            cam.pose.orientation = cam_tf.transform.rotation
            cam.scale.x = 0.03
            cam.scale.y = 0.04
            cam.scale.z = 0.02
            cam.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
            msg.markers.append(cam)

            # Label above camera
            cam_label = Marker()
            cam_label.header.stamp = now
            cam_label.header.frame_id = frame_id
            cam_label.ns = 'camera_pose'
            cam_label.id = next_id; next_id += 1
            cam_label.type = Marker.TEXT_VIEW_FACING
            cam_label.action = Marker.ADD
            cam_label.pose.position.x = cam_tf.transform.translation.x
            cam_label.pose.position.y = cam_tf.transform.translation.y
            cam_label.pose.position.z = cam_tf.transform.translation.z + 0.04
            cam_label.scale.z = 0.02
            cam_label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            cam_label.text = 'Camera'
            msg.markers.append(cam_label)
        except Exception:
            pass  # TF not yet available

        if msg.markers:
            self._marker_array_pub.publish(msg)
        else:
            self.get_logger().warn('RViz: no markers with positionInBase to publish', throttle_duration_sec=5.0)

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
