# Aruco marker can be spawned ingazebo with these commands: 
# export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models
# ros2 run ros_gz_sim create -file /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf -name aruco_marker -x -0.6 -y 0 -z 0.4 -R 0 -P 0 -Y 0

#To view frames: ros2 run tf2_tools view_frames



#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from showVideoFeed import CameraViewer
from poseReader import PoseReader
import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess

class ArucoDetectionViewer(PoseReader, CameraViewer):
    """
    Extends CameraViewer to detect and display ArUco markers.
    """
    def __init__(self):
        # Cooperative multiple inheritance with explicit node name
        super().__init__('aruco_detection_viewer', enable_pose_print=False)
        
        # Declare ArUco-specific parameters
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('marker_size', 0.05)  # Size in meters (matching your model.sdf)
        self.declare_parameter('show_rejected', False)
        
        # Declare calibration mode parameter
        self.declare_parameter('calibration_mode', True)
        
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


        # Subscribe to camera info for calibration parameters
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/rgbd_camera/camera_info', self.camera_info_callback, 10
        )
        
        # Track the number of detected markers to log only on changes
        self.last_marker_count = 0
        # Avoid spamming TF warnings
        self._warned_tf_failure = False
        # Throttle per-frame console logging
        self._last_log_time = 0.0
        self.log_interval_s = 1.0
        # Display scaling for larger window
        self.display_scale = 2.0
        
        self.get_logger().info(f'ArUco detector initialized with dictionary: {aruco_dict_name}')
        self.get_logger().info(f'Marker size: {self.marker_size}m')

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera calibration parameters from CameraInfo message."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            # Hardcode to the correct camera frame as per user confirmation
            self.camera_frame = "ee_camera_link"
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

    def _lookup_base_camera_transform(self):
        """
        Get the camera pose in the world frame using PoseReader's get_frame method.
        Returns a tuple: (position, euler_angles)
        """
        # Use the camera frame if available, otherwise default to 'ee_camera_link'
        frame = self.camera_frame if self.camera_frame else "ee_camera_link"
        pose = self.get_frame(frame)
        position = pose[:3]
        euler_angles = pose[3:]
        return position, euler_angles

    def pose_to_homogeneous_matrix(self, position, euler_angles):
        """
        Convert position and Euler angles to a 4x4 homogeneous transformation matrix.
        Args:
            position: [x, y, z] list or array
            euler_angles: [roll, pitch, yaw] in radians, 'XYZ' order
        Returns:
            4x4 numpy array (homogeneous matrix)
        """
        # Create rotation matrix from Euler angles
        rot = R.from_euler('XYZ', euler_angles, degrees=False)
        rotation_matrix = rot.as_matrix()
        
        # Create homogeneous matrix
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        homogeneous_matrix[:3, 3] = position
        
        return homogeneous_matrix

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
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                
                # Estimate pose of each marker (camera -> marker)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                # Extract position (translation vector)
                position_cam = tvec[0][0]  # [x, y, z] in meters (camera frame)
                distance = np.linalg.norm(position_cam)
                
                # Convert rotation vector to euler angles (radians)
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                rot = R.from_matrix(rotation_matrix)
                roll, pitch, yaw = rot.as_euler('XYZ', degrees=False)
                euler_cam = np.array([roll, pitch, yaw])
                
                # Get camera pose in base frame using get_frame
                camera_pos, camera_euler = self._lookup_base_camera_transform()
                
                # Check if calibration mode is enabled and not yet calibrated
                calibration_mode = self.get_parameter('calibration_mode').get_parameter_value().bool_value
                if calibration_mode:
                    HTM_camera = self.pose_to_homogeneous_matrix(camera_pos, camera_euler)
                    HTM_marker_cam = self.pose_to_homogeneous_matrix(position_cam, euler_cam)

                    self.get_logger().info('Calibration mode: Logging marker poses for all offset combinations (pi/2 increments)')
                    # Possible offsets: 0, 90°, 180°, 270° (in radians)
                    offsets = [0, np.pi/2, np.pi, 3*np.pi/2]
                    for roll_offset in offsets:
                        for pitch_offset in offsets:
                            for yaw_offset in offsets:
                                offset = np.array([roll_offset, pitch_offset, yaw_offset])
                                camera_euler_offset = camera_euler + offset
                                # Create HTMs
                                HTM_camera = self.pose_to_homogeneous_matrix(camera_pos, camera_euler_offset)
                                HTM_marker_base = HTM_camera @ HTM_marker_cam
                                position_base = HTM_marker_base[:3, 3]
                                rot_base = R.from_matrix(HTM_marker_base[:3, :3])
                                rpy_base = rot_base.as_euler('XYZ', degrees=True)
                                self.get_logger().info(
                                    f'Offset [{np.degrees(roll_offset):.0f}°, {np.degrees(pitch_offset):.0f}°, {np.degrees(yaw_offset):.0f}°]: '
                                    f'Pos [{position_base[0]:.3f}, {position_base[1]:.3f}, {position_base[2]:.3f}], '
                                    f'RPY [{rpy_base[0]:.1f}°, {rpy_base[1]:.1f}°, {rpy_base[2]:.1f}°]'
                                )
                    self.get_logger().info('Calibration complete. Disable calibration_mode and set the chosen offset manually.')
                
                # Use fixed offset (update this with the chosen calibration result)
                camera_euler = camera_euler + np.array([np.pi, np.pi, np.pi])  # Replace with calibrated values, e.g., np.array([0, np.pi, 0])
                
                # Create HTMs
                HTM_camera = self.pose_to_homogeneous_matrix(camera_pos, camera_euler)
                HTM_marker_cam = self.pose_to_homogeneous_matrix(position_cam, euler_cam)

                # Transform marker to base frame
                HTM_marker_base = HTM_camera @ HTM_marker_cam
                position_base = HTM_marker_base[:3, 3]
                rot_base = R.from_matrix(HTM_marker_base[:3, :3])
                rpy_base = rot_base.as_euler('XYZ', degrees=True)


                orientation_base = {
                    'roll': float(rpy_base[0]),
                    'pitch': float(rpy_base[1]),
                    'yaw': float(rpy_base[2]),
                }
                
                # Store pose data for overlay
                entry = {
                    'id': marker_id,
                    'position': position_cam,
                    'orientation': {
                        'roll': np.degrees(roll),
                        'pitch': np.degrees(pitch),
                        'yaw': np.degrees(yaw)
                    },
                    'distance': distance,
                    'position_base': position_base,
                    'orientation_base': orientation_base
                }
                
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
            
            
            # Only log when the number of markers changes
            current_marker_count = len(ids)
            if current_marker_count != self.last_marker_count:
                marker_ids = ids.flatten().tolist()
                self.get_logger().info(f'Detected {current_marker_count} ArUco marker(s): {marker_ids}')
                # Log pose information using already computed marker_poses
                for entry in marker_poses:
                    marker_id = entry['id']
                    position_cam = entry['position']
                    orientation = entry['orientation']
                    distance = entry['distance']
                    if 'position_base' in entry and 'orientation_base' in entry:
                        pb = entry['position_base']
                        ob = entry['orientation_base']
                        self.get_logger().info(
                            f'  Marker {marker_id}: Camera [x={position_cam[0]:.3f}m, y={position_cam[1]:.3f}m, z={position_cam[2]:.3f}m, dist={distance:.3f}m] '
                            f'R/P/Y [{orientation['roll']:.1f}°, {orientation['pitch']:.1f}°, {orientation['yaw']:.1f}°] | '
                            f'Base [x={pb[0]:.3f}m, y={pb[1]:.3f}m, z={pb[2]:.3f}m] '
                            f'R/P/Y [{ob['roll']:.1f}°, {ob['pitch']:.1f}°, {ob['yaw']:.1f}°]'
                        )
                    else:
                        self.get_logger().info(
                            f'  Marker {marker_id}: '
                            f'Camera [x={position_cam[0]:.3f}m, y={position_cam[1]:.3f}m, z={position_cam[2]:.3f}m, dist={distance:.3f}m] '
                            f'R/P/Y [{orientation['roll']:.1f}°, {orientation['pitch']:.1f}°, {orientation['yaw']:.1f}°]'
                        )
               
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

        # Compose side-by-side video feed
        combined = np.hstack([color_with_markers, depth_vis])

        # Create a separate text panel and draw info below the video feed
        panel_height = max(160, 40 + len(marker_poses) * 160)
        text_panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
        self._draw_pose_overlay(text_panel, marker_poses)

        # Stack video feed above text panel
        final_frame = np.vstack([combined, text_panel])

        # Add instruction text on the final frame
        cv2.putText(final_frame, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Scale up display for readability
        if self.display_scale and self.display_scale != 1.0:
            frame_to_show = cv2.resize(
                final_frame, None, fx=self.display_scale, fy=self.display_scale,
                interpolation=cv2.INTER_LINEAR
            )
        else:
            frame_to_show = final_frame

        cv2.imshow('RGB with ArUco (top) + Text (bottom)', frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit key pressed, shutting down...')
            rclpy.shutdown()
    

    def _draw_pose_overlay(self, image: np.ndarray, marker_poses: list):
        """Draw camera→marker, base→marker, and end effector pose in the text panel."""
        y_offset = 18  # Start a bit lower for top margin
        line_height = 14  # Reduced line height for tighter spacing

        if not marker_poses:
            cv2.putText(image, "No markers detected", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += line_height

        for pose_data in marker_poses:
            marker_id = pose_data['id']
            position = pose_data['position']
            orientation = pose_data['orientation']
            distance = pose_data['distance']

            # Camera→marker
            cv2.putText(image, f"Marker {marker_id} (Camera→Marker):", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += line_height
            cv2.putText(image, f"  Pos: X={position[0]:+.3f} Y={position[1]:+.3f} Z={position[2]:+.3f}  Dist={distance:.3f}m", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(image, f"  RPY: R={orientation['roll']:+.1f}° P={orientation['pitch']:+.1f}° Y={orientation['yaw']:+.1f}°", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height

            # Base→marker
            if 'position_base' in pose_data and 'orientation_base' in pose_data:
                pb = pose_data['position_base']
                ob = pose_data['orientation_base']
                cv2.putText(image, f"  (Base→Marker):", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
                y_offset += line_height
                cv2.putText(image, f"    Pos: X={pb[0]:+.3f} Y={pb[1]:+.3f} Z={pb[2]:+.3f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
                cv2.putText(image, f"    RPY: R={ob['roll']:+.1f}° P={ob['pitch']:+.1f}° Y={ob['yaw']:+.1f}°", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height

            y_offset += 2  # Small gap between markers

        # Show last known end effector pose from PoseReader
        y_offset += 10
        cv2.putText(image, "End Effector Pose (from FK):", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        y_offset += line_height
        try:
            p = self.pose  # [x, y, z, roll, pitch, yaw]
            q = self.quat  # [qx, qy, qz, qw]
            cv2.putText(image, f"  Pos: X={p[0]:+.3f} Y={p[1]:+.3f} Z={p[2]:+.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(image, f"  RPY: R={p[3]:+.1f}° P={p[4]:+.1f}° Y={p[5]:+.1f}°", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(image, f"  Quat: x={q[0]:+.3f} y={q[1]:+.3f} z={q[2]:+.3f} w={q[3]:+.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception:
            cv2.putText(image, "  (No pose available)", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionViewer()
    def spawn_aruco_marker(n: Node):
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf',
            '-name', 'aruco_marker',
            '-x', '-0.6', '-y', '0', '-z', '0.4',
            '-R', '0', '-P', '0', '-Y', '0'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or 'Spawn command executed successfully'
            n.get_logger().info(f'Aruco spawn: {msg}')
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            n.get_logger().error(f'Aruco spawn failed: {err}')
        except Exception as e:
            n.get_logger().error(f'Aruco spawn exception: {e}')

    spawn_aruco_marker(node)
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
