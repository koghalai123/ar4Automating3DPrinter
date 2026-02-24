#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from webVideoServer import WebVideoStream


class CameraViewer(Node):
    def __init__(self, node_name: str = None, web_port: int = 5000, enable_aruco: bool = True,
                 use_webcam: bool = False, webcam_index: int = 0):
        super().__init__(node_name or 'camera_viewer')

        self.declare_parameter('color_topic', '/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/rgbd_camera/depth_image')
        self.declare_parameter('colormap', 'turbo')
        self.declare_parameter('use_webcam', use_webcam)
        self.declare_parameter('webcam_index', webcam_index)

        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.colormap_name = self.get_parameter('colormap').get_parameter_value().string_value
        self.use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        self.webcam_index = self.get_parameter('webcam_index').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None

        self.dt = 1.0 / 30.0
        self.timer = self.create_timer(self.dt, self.render)

        # Web video stream instead of cv2.imshow
        self.web_stream = WebVideoStream(port=web_port)
        self.web_stream.start()

        # Display scaling for larger window
        self.display_scale = 2.0

        # ArUco detection setup
        self.enable_aruco = enable_aruco
        self.marker_poses = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        # --- Video source setup ---
        self.webcam_cap = None
        if self.use_webcam:
            self._init_webcam()
        else:
            self._init_topics(color_topic, depth_topic)

        if self.enable_aruco:
            self._init_aruco()

        source_desc = f'webcam (index {self.webcam_index})' if self.use_webcam else f'topics color: {color_topic}, depth: {depth_topic}'
        self.get_logger().info(f'Camera viewer using {source_desc}')
        self.get_logger().info(f'View stream at http://localhost:{web_port}')
        if self.enable_aruco:
            self.get_logger().info('ArUco detection enabled')

    # ------------------------------------------------------------------
    # Video source initialisation
    # ------------------------------------------------------------------

    def _init_topics(self, color_topic: str, depth_topic: str):
        """Subscribe to ROS image topics for color and depth."""
        self.color_sub = self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

    def _init_webcam(self):
        """Open a local webcam via OpenCV VideoCapture."""
        self.webcam_cap = cv2.VideoCapture(self.webcam_index)
        if not self.webcam_cap.isOpened():
            self.get_logger().error(f'Failed to open webcam at index {self.webcam_index}')
            self.webcam_cap = None
        else:
            self.get_logger().info(f'Webcam opened at index {self.webcam_index}')

    def _init_aruco(self):
        """Initialize ArUco detection components."""
        self.aruco_dicts = [
            self._get_aruco_dict('DICT_4X4_50'),
            self._get_aruco_dict('DICT_6X6_50'),
        ]
        self.marker_sizes = [0.03, 0.05]  # Sizes in meters, indexed by dictionary
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.last_marker_count = 0
        self._last_log_time = 0.0
        self.log_interval_s = 1.0

        if self.use_webcam:
            # For webcam, use a default/approximate camera matrix if no CameraInfo topic
            self.get_logger().info('Webcam mode: waiting for manual or default camera calibration')
        else:
            self.camera_info_sub = self.create_subscription(
                CameraInfo, '/rgbd_camera/camera_info', self.camera_info_callback, 10
            )

        self.get_logger().info(
            f'ArUco detector initialized with marker sizes: {self.marker_sizes}m'
        )

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera calibration parameters from CameraInfo message."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_frame = "ee_camera_link"
            self.get_logger().info(f'Camera calibration parameters received (frame: {self.camera_frame})')

    def set_webcam_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Manually set camera calibration for webcam mode.
        Call this if you have calibration data for your webcam.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_frame = "webcam"
        self.get_logger().info('Webcam calibration set manually')

    def set_default_webcam_calibration(self, width: int, height: int):
        """
        Set an approximate camera matrix based on frame dimensions.
        Suitable for rough ArUco detection when no real calibration is available.
        """
        focal_length = max(width, height)
        cx, cy = width / 2.0, height / 2.0
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.camera_frame = "webcam"
        self.get_logger().info(
            f'Default webcam calibration set (focal={focal_length}, cx={cx:.0f}, cy={cy:.0f})'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def color_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Color image error: {str(e)}')

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth image error: {str(e)}')

    def _read_webcam(self):
        """Grab a frame from the webcam and update latest_color."""
        if self.webcam_cap is None or not self.webcam_cap.isOpened():
            return
        ret, frame = self.webcam_cap.read()
        if ret:
            self.latest_color = frame
            # Auto-set default calibration on first frame if none provided
            if self.enable_aruco and self.camera_matrix is None:
                h, w = frame.shape[:2]
                self.set_default_webcam_calibration(w, h)
        else:
            self.get_logger().warn('Failed to read frame from webcam', throttle_duration_sec=2.0)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        # If using webcam, grab a new frame each tick
        if self.use_webcam:
            self._read_webcam()

        has_color = self.latest_color is not None
        has_depth = self.latest_depth is not None

        # In webcam mode, depth is optional; in topic mode, both are required
        if not has_color:
            return
        if not self.use_webcam and not has_depth:
            return

        color_img = self.latest_color.copy()

        # Detect ArUco markers if enabled
        if self.enable_aruco:
            color_img, marker_poses = self._detect_aruco_markers(color_img)
        else:
            marker_poses = []

        # Build the combined frame
        if has_depth:
            depth_raw = self.latest_depth
            # Resize depth to match color dimensions
            if depth_raw.shape[:2] != color_img.shape[:2]:
                depth_raw = cv2.resize(
                    depth_raw, (color_img.shape[1], color_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth_vis = self._visualize_depth(depth_raw)
            if len(depth_vis.shape) == 2:
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        else:
            depth_vis = None

        if len(color_img.shape) == 2:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)

        if depth_vis is not None:
            combined = np.hstack([color_img, depth_vis])
        else:
            combined = color_img

        # Create a separate text panel and draw info below the video feed
        panel_height = max(160, 40 + len(marker_poses) * 160)
        text_panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
        self._draw_pose_overlay(text_panel, marker_poses)

        final_frame = np.vstack([combined, text_panel])

        # Scale up display for readability
        if self.display_scale and self.display_scale != 1.0:
            frame_to_show = cv2.resize(
                final_frame, None, fx=self.display_scale, fy=self.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_to_show = final_frame

        self.web_stream.update_frame(frame_to_show)

    # ------------------------------------------------------------------
    # ArUco detection
    # ------------------------------------------------------------------

    def _detect_aruco_markers(self, image: np.ndarray) -> tuple:
        """
        Detect ArUco markers in the image, draw them, and estimate poses.
        Returns (annotated_image, list_of_marker_pose_dicts).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        all_corners = []
        all_ids = []
        dict_indices = []
        for idx, aruco_dict in enumerate(self.aruco_dicts):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids)
                dict_indices.extend([idx] * len(ids))

        output_image = image.copy()
        marker_poses = []

        if all_ids:
            all_corners_np = all_corners
            all_ids_np = np.array(all_ids).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(output_image, all_corners_np, all_ids_np)

            if self.camera_matrix is None or self.dist_coeffs is None:
                self.get_logger().warn(
                    "Camera intrinsics not yet received. Skipping pose estimation.",
                    throttle_duration_sec=0.5,
                )
                self.marker_poses = marker_poses
                return output_image, marker_poses

            seen = set()
            for i, corner in enumerate(all_corners):
                marker_id = all_ids[i][0]
                if marker_id in seen:
                    continue
                seen.add(marker_id)
                dict_idx = dict_indices[i]
                marker_size = self.marker_sizes[dict_idx]

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, marker_size, self.camera_matrix, self.dist_coeffs,
                )

                position_cam = tvec[0][0]
                distance = np.linalg.norm(position_cam)
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                rot = R.from_matrix(rotation_matrix)
                roll, pitch, yaw = rot.as_euler("XYZ", degrees=False)
                euler_cam = np.array([roll, pitch, yaw])

                # Build basic entry (camera-frame only)
                entry = {
                    'id': marker_id,
                    'positionFromCamera': position_cam,
                    'eulerFromCamera': euler_cam,
                    'orientFromCamera': {
                        'roll': np.degrees(roll),
                        'pitch': np.degrees(pitch),
                        'yaw': np.degrees(yaw),
                    },
                    'distanceFromCamera': distance,
                }

                # Hook for subclasses to enrich with world-frame data
                entry = self._enrich_marker_pose(entry)
                if entry is not None:
                    marker_poses.append(entry)

                # Draw axis
                cv2.drawFrameAxes(
                    output_image, self.camera_matrix, self.dist_coeffs,
                    rvec[0], tvec[0], marker_size * 0.5,
                )

                # Display ID and distance near marker
                corner_center = corner[0].mean(axis=0).astype(int)
                cv2.putText(
                    output_image,
                    f"ID:{marker_id} D:{distance:.2f}m",
                    tuple(corner_center + np.array([0, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

            # Log when marker count changes
            current_marker_count = len(all_ids)
            if current_marker_count != self.last_marker_count:
                self.last_marker_count = current_marker_count
                marker_id_list = [id[0] for id in all_ids]
                self.get_logger().info(f'Detected {current_marker_count} ArUco marker(s): {marker_id_list}')
                for entry in marker_poses:
                    self._log_marker_pose(entry)
        else:
            if self.last_marker_count > 0:
                self.get_logger().info('No ArUco markers detected')
                self.last_marker_count = 0

        self.marker_poses = marker_poses
        return output_image, marker_poses

    def _enrich_marker_pose(self, entry: dict) -> dict:
        """
        Hook for subclasses to add world-frame pose data to a detected marker entry.
        The base implementation returns the entry unchanged (camera-frame only).
        Return None to skip this marker.
        """
        return entry

    def _log_marker_pose(self, entry: dict):
        """Log a single marker's pose. Override in subclasses for richer logging."""
        position_cam = entry['positionFromCamera']
        orient_cam = entry['orientFromCamera']
        distance_cam = entry['distanceFromCamera']
        marker_id = entry['id']

        log_line = (
            f'  Marker {marker_id}: '
            f'Camera [{position_cam[0]:.3f}m, {position_cam[1]:.3f}m, {position_cam[2]:.3f}m, '
            f"R={orient_cam['roll']:.1f}°, P={orient_cam['pitch']:.1f}°, Y={orient_cam['yaw']:.1f}°] "
            f'Dist={distance_cam:.3f}m'
        )

        if 'positionInWorld' in entry:
            pb = entry['positionInWorld']
            ob = entry['orientInWorld']
            log_line += (
                f" | World [{pb[0]:.3f}m, {pb[1]:.3f}m, {pb[2]:.3f}m, "
                f"R={ob['roll']:.1f}°, P={ob['pitch']:.1f}°, Y={ob['yaw']:.1f}°]"
            )

        self.get_logger().info(log_line)

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_pose_overlay(self, image: np.ndarray, marker_poses: list):
        """Draw camera→marker and (optionally) base→marker pose in the text panel."""
        y_offset = 18
        line_height = 14

        if not marker_poses:
            cv2.putText(image, "No markers detected", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        for pose_data in marker_poses:
            marker_id = pose_data['id']
            position = pose_data['positionFromCamera']
            orientation = pose_data['orientFromCamera']
            distance = pose_data['distanceFromCamera']

            cv2.putText(image, f"Marker {marker_id} (Camera->Marker):", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += line_height
            cv2.putText(image,
                        f"  Pos: X={position[0]:+.3f} Y={position[1]:+.3f} Z={position[2]:+.3f}  Dist={distance:.3f}m",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(image,
                        f"  RPY: R={orientation['roll']:+.1f} P={orientation['pitch']:+.1f} Y={orientation['yaw']:+.1f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height

            if 'positionInWorld' in pose_data:
                pb = pose_data['positionInWorld']
                ob = pose_data['orientInWorld']
                cv2.putText(image, f"  (Base->Marker):", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
                y_offset += line_height
                cv2.putText(image,
                            f"    Pos: X={pb[0]:+.3f} Y={pb[1]:+.3f} Z={pb[2]:+.3f}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(image,
                            f"    RPY: R={ob['roll']:+.1f} P={ob['pitch']:+.1f} Y={ob['yaw']:+.1f}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height

            y_offset += 2

    # ------------------------------------------------------------------
    # Depth visualisation helpers
    # ------------------------------------------------------------------

    def _get_aruco_dict(self, dict_name: str):
        """Get the ArUco dictionary based on name."""
        dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
            'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])

    def _visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            d = depth.copy()
            mask = np.isfinite(d) & (d > 0)
            if not np.any(mask):
                return np.zeros_like(d, dtype=np.uint8)
            low = float(np.percentile(d[mask], 1.0))
            high = float(np.percentile(d[mask], 99.0))
            if high <= low:
                high = low + 1e-3
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)
        elif depth.dtype == np.uint16:
            d = depth.astype(np.float32)
            low = float(np.percentile(d[d > 0], 1.0)) if np.any(d > 0) else 0.0
            high = float(np.percentile(d[d > 0], 99.0)) if np.any(d > 0) else 1.0
            if high <= low:
                high = low + 1.0
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)
        else:
            d = depth.astype(np.float32)
            low = float(np.percentile(d, 1.0))
            high = float(np.percentile(d, 99.0))
            if high <= low:
                high = low + 1.0
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)

        cmap = self._cv_colormap(self.colormap_name)
        return cv2.applyColorMap(d_norm, cmap)

    def _cv_colormap(self, name: str) -> int:
        if name.lower() == 'turbo':
            return cv2.COLORMAP_TURBO
        if name.lower() == 'jet':
            return cv2.COLORMAP_JET
        if name.lower() == 'inferno':
            return cv2.COLORMAP_INFERNO
        return cv2.COLORMAP_TURBO

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        if self.webcam_cap is not None:
            self.webcam_cap.release()
            self.get_logger().info('Webcam released')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()