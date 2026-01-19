# Aruco marker can be spawned ingazebo with these commands: 
# export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models
# ros2 run ros_gz_sim create -file /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf -name aruco_marker -x 0 -y -1 -z 1 -R 0 -P 0 -Y -1.57





#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from showVideoFeed import CameraViewer
from poseReader import PoseReader
from rclpy.time import Time
import tf2_ros
from scipy.spatial.transform import Rotation as R


class ArucoDetectionViewer(PoseReader, CameraViewer):
    """
    Extends CameraViewer to detect and display ArUco markers.
    """
    def __init__(self):
        # Cooperative multiple inheritance with explicit node name
        super().__init__('aruco_detection_viewer')
        
        # Declare ArUco-specific parameters
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('marker_size', 0.15)  # Size in meters (matching your model.sdf)
        self.declare_parameter('show_rejected', False)
        
        aruco_dict_name = self.get_parameter('aruco_dict').get_parameter_value().string_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.show_rejected = self.get_parameter('show_rejected').get_parameter_value().bool_value
        
        # Initialize ArUco detector
        self.aruco_dict = self._get_aruco_dict(aruco_dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera intrinsics - will be populated from camera_info topic
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        # TF buffer/listener to get base->camera transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribe to camera info for calibration parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/rgbd_camera/camera_info', self.camera_info_callback, 10
        )
        
        # Track the number of detected markers to log only on changes
        self.last_marker_count = 0
        
        self.get_logger().info(f'ArUco detector initialized with dictionary: {aruco_dict_name}')
        self.get_logger().info(f'Marker size: {self.marker_size}m')

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera calibration parameters from CameraInfo message."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            # Remember camera frame for TF lookup
            self.camera_frame = msg.header.frame_id or 'rgbd_camera'
            self.get_logger().info(f'Camera calibration parameters received (frame: {self.camera_frame})')

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
        
        if dict_name not in dict_map:
            self.get_logger().warn(f'Unknown dictionary {dict_name}, defaulting to DICT_4X4_50')
            dict_name = 'DICT_4X4_50'
        
        return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])

    def _detect_aruco_markers(self, image: np.ndarray) -> tuple:
        """
        Detect ArUco markers in the image and draw them.
        Returns the image with markers drawn and a list of pose data.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers (using older OpenCV API)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        # Draw detected markers
        output_image = image.copy()
        marker_poses = []
        
        if ids is not None and len(ids) > 0:
            # Draw all detected markers
            cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
            
            # If we have camera calibration, estimate pose
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    
                    # Estimate pose of each marker (camera -> marker)
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    
                    # Extract position (translation vector)
                    position_cam = tvec[0][0]  # [x, y, z] in meters (camera frame)
                    distance = np.linalg.norm(position_cam)
                    
                    # Convert rotation vector to euler angles
                    rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                        yaw = 0
                    
                    roll_deg = np.degrees(roll)
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)

                    # Compute base -> marker using TF (base -> camera) * (camera -> marker)
                    position_base = None
                    orientation_base = None
                    if self.camera_frame and hasattr(self, 'base_link_name'):
                        try:
                            tf_bc = self.tf_buffer.lookup_transform(
                                self.base_link_name,  # target (base)
                                self.camera_frame,    # source (camera)
                                Time()
                            )
                            t_bc = np.array([
                                tf_bc.transform.translation.x,
                                tf_bc.transform.translation.y,
                                tf_bc.transform.translation.z,
                            ])
                            q_bc = [
                                tf_bc.transform.rotation.x,
                                tf_bc.transform.rotation.y,
                                tf_bc.transform.rotation.z,
                                tf_bc.transform.rotation.w,
                            ]
                            R_bc = R.from_quat(q_bc).as_matrix()
                            t_cm = position_cam.reshape(3)
                            R_cm = rotation_matrix

                            t_bm = R_bc @ t_cm + t_bc
                            R_bm = R_bc @ R_cm
                            rpy_bm = R.from_matrix(R_bm).as_euler('xyz', degrees=True)

                            position_base = t_bm
                            orientation_base = {
                                'roll': float(rpy_bm[0]),
                                'pitch': float(rpy_bm[1]),
                                'yaw': float(rpy_bm[2]),
                            }
                        except Exception as e:
                            # TF may be unavailable early; continue with camera frame only
                            if self.last_marker_count == 0:
                                self.get_logger().warn(f'Base->Camera TF lookup failed: {e}')
                    
                    # Store pose data for overlay
                    entry = {
                        'id': marker_id,
                        'position': position_cam,
                        'orientation': {
                            'roll': roll_deg,
                            'pitch': pitch_deg,
                            'yaw': yaw_deg
                        },
                        'distance': distance
                    }
                    if position_base is not None and orientation_base is not None:
                        entry['position_base'] = position_base
                        entry['orientation_base'] = orientation_base
                    marker_poses.append(entry)
                    
                    # Draw axis for each marker
                    cv2.drawFrameAxes(
                        output_image, self.camera_matrix, self.dist_coeffs,
                        rvec[0], tvec[0], self.marker_size * 0.5
                    )
                    
                    # Display ID and distance near marker
                    corner_center = corner[0].mean(axis=0).astype(int)
                    
                    cv2.putText(
                        output_image,
                        f"ID:{marker_id} D:{distance:.2f}m",
                        tuple(corner_center + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            else:
                # Just display marker IDs without pose estimation
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    corner_center = corner[0].mean(axis=0).astype(int)
                    
                    cv2.putText(
                        output_image,
                        f"ID:{marker_id}",
                        tuple(corner_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            
            # Only log when the number of markers changes
            current_marker_count = len(ids)
            if current_marker_count != self.last_marker_count:
                marker_ids = ids.flatten().tolist()
                self.get_logger().info(f'Detected {current_marker_count} ArUco marker(s): {marker_ids}')
                
                # Log pose information if camera is calibrated
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    for i, corner in enumerate(corners):
                        marker_id = ids[i][0]
                        # Estimate pose of each marker
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # Extract position (translation vector)
                        position_cam = tvec[0][0]  # [x, y, z] in meters (camera)
                        
                        # Convert rotation vector to euler angles for easier interpretation
                        rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                        # Extract euler angles (roll, pitch, yaw) in degrees
                        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                        singular = sy < 1e-6
                        if not singular:
                            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                        else:
                            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                            yaw = 0
                        
                        roll_deg = np.degrees(roll)
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        
                        distance = np.linalg.norm(position_cam)
                        
                        # Also try base->marker if TF available
                        if self.camera_frame and hasattr(self, 'base_link_name'):
                            try:
                                tf_bc = self.tf_buffer.lookup_transform(
                                    self.base_link_name,
                                    self.camera_frame,
                                    Time()
                                )
                                t_bc = np.array([
                                    tf_bc.transform.translation.x,
                                    tf_bc.transform.translation.y,
                                    tf_bc.transform.translation.z,
                                ])
                                q_bc = [
                                    tf_bc.transform.rotation.x,
                                    tf_bc.transform.rotation.y,
                                    tf_bc.transform.rotation.z,
                                    tf_bc.transform.rotation.w,
                                ]
                                R_bc = R.from_quat(q_bc).as_matrix()
                                t_cm = position_cam.reshape(3)
                                R_cm = rotation_matrix
                                t_bm = R_bc @ t_cm + t_bc
                                R_bm = R_bc @ R_cm
                                rpy_bm = R.from_matrix(R_bm).as_euler('xyz', degrees=True)
                                self.get_logger().info(
                                    f'  Marker {marker_id}: Camera [x={position_cam[0]:.3f}m, y={position_cam[1]:.3f}m, z={position_cam[2]:.3f}m, dist={distance:.3f}m] '
                                    f'R/P/Y [{roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°] | '
                                    f'Base [x={t_bm[0]:.3f}m, y={t_bm[1]:.3f}m, z={t_bm[2]:.3f}m] '
                                    f'R/P/Y [{rpy_bm[0]:.1f}°, {rpy_bm[1]:.1f}°, {rpy_bm[2]:.1f}°]'
                                )
                            except Exception:
                                self.get_logger().info(
                                    f'  Marker {marker_id}: '
                                    f'Camera [x={position_cam[0]:.3f}m, y={position_cam[1]:.3f}m, z={position_cam[2]:.3f}m, dist={distance:.3f}m] '
                                    f'R/P/Y [{roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°]'
                                )
                        else:
                            self.get_logger().info(
                                f'  Marker {marker_id}: '
                                f'Camera [x={position_cam[0]:.3f}m, y={position_cam[1]:.3f}m, z={position_cam[2]:.3f}m, dist={distance:.3f}m] '
                                f'R/P/Y [{roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°]'
                            )
                else:
                    self.get_logger().warn('  Camera not calibrated - pose estimation unavailable')
                
                self.last_marker_count = current_marker_count
        else:
            # Log when markers disappear
            if self.last_marker_count > 0:
                self.get_logger().info('No ArUco markers detected')
                self.last_marker_count = 0
        
        # Optionally draw rejected markers
        if self.show_rejected and rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(output_image, rejected, borderColor=(100, 0, 255))
        
        return output_image, marker_poses

    def render(self):
        """Override render to include ArUco detection."""
        if self.latest_color is None or self.latest_depth is None:
            return

        color_img = self.latest_color.copy()
        depth_raw = self.latest_depth

        # Detect ArUco markers on the color image and get pose data
        color_with_markers, marker_poses = self._detect_aruco_markers(color_img)

        # Resize depth to match color height
        if depth_raw.shape[:2] != color_with_markers.shape[:2]:
            depth_raw = cv2.resize(depth_raw, (color_with_markers.shape[1], color_with_markers.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)

        depth_vis = self._visualize_depth(depth_raw)

        # Ensure both are 3-channel BGR for display
        if len(color_with_markers.shape) == 2:
            color_with_markers = cv2.cvtColor(color_with_markers, cv2.COLOR_GRAY2BGR)
        if len(depth_vis.shape) == 2:
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # Add pose information overlay on the color image
        self._draw_pose_overlay(color_with_markers, marker_poses)

        combined = np.hstack([color_with_markers, depth_vis])
        
        # Add instruction text
        cv2.putText(combined, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('RGB with ArUco (left) + Depth (right)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit key pressed, shutting down...')
            rclpy.shutdown()
    
    def _draw_pose_overlay(self, image: np.ndarray, marker_poses: list):
        """Draw 6 DOF pose information as overlay on the image."""
        if not marker_poses:
            # Display message when no markers detected
            cv2.putText(image, "No markers detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return
        
        # Starting position for text overlay
        y_offset = 60
        line_height = 25
        
        for pose_data in marker_poses:
            marker_id = pose_data['id']
            position = pose_data['position']
            orientation = pose_data['orientation']
            distance = pose_data['distance']
            
            # Create semi-transparent background for text
            overlay = image.copy()
            cv2.rectangle(overlay, (5, y_offset - 20), (400, y_offset + line_height * 6 + 5), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            # Display marker ID and distance
            cv2.putText(image, f"Marker ID: {marker_id}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
            
            cv2.putText(image, f"Distance: {distance:.3f} m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Position (Translation) — camera frame
            cv2.putText(image, f"Position (m):", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_offset += line_height
            
            cv2.putText(image, f"  Cam X: {position[0]:+.3f}  Y: {position[1]:+.3f}  Z: {position[2]:+.3f}", 
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Orientation (Rotation) — camera frame
            cv2.putText(image, f"Orientation (deg):", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_offset += line_height
            
            cv2.putText(image, 
                       f"  R: {orientation['roll']:+.1f}  P: {orientation['pitch']:+.1f}  Y: {orientation['yaw']:+.1f}", 
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height + 10

            # If base-frame data available, show it too
            if 'position_base' in pose_data and 'orientation_base' in pose_data:
                pb = pose_data['position_base']
                ob = pose_data['orientation_base']
                cv2.putText(image, f"Base Position (m):", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                y_offset += line_height
                cv2.putText(image, f"  X: {pb[0]:+.3f}  Y: {pb[1]:+.3f}  Z: {pb[2]:+.3f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(image, f"Base Orientation (deg):", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                y_offset += line_height
                cv2.putText(image,
                           f"  R: {ob['roll']:+.1f}  P: {ob['pitch']:+.1f}  Y: {ob['yaw']:+.1f}",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height + 10


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
