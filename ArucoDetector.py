# To create marker: ros2 run ros_gz_sim create -file /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf -name aruco_marker -x -0.6 -y 0 -z 0.4 -R 0 -P 0 -Y 0
# To delete the marker: gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 1000 --req 'name: "aruco_marker" type: MODEL'

#To view frames: ros2 run tf2_tools view_frames
# To view transformation: ros2 run tf2_ros tf2_echo world ee_link


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from showVideoFeed import CameraViewer
from poseReader import PoseReader
import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
import tf2_ros
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose

class ArucoDetectionViewer(PoseReader, CameraViewer): 
    """
    Extends CameraViewer to detect and display ArUco markers.
    """
    def __init__(self):
        super().__init__('aruco_detection_viewer', enable_pose_print=False)
        
        # Declare ArUco-spe cific parameters
        self.aruco_dicts = [self._get_aruco_dict('DICT_4X4_50'), self._get_aruco_dict('DICT_6X6_50')]
        self.marker_sizes = [0.03, 0.05]  # Sizes in meters, indexed by dictionary (4x4: 0.01m, 6x6: 0.05m) 
        self.calibration_mode = False
        
        # Initialize ArUco detector
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera intrinsics - will be populated from camera_info topic
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/rgbd_camera/camera_info', self.camera_info_callback, 10
        )

        self.last_marker_count = 0
        self._last_log_time = 0.0
        self.log_interval_s = 1.0
        # Display scaling for larger window
        self.display_scale = 2.0
        
        self.get_logger().info(f'ArUco detector initialized with dictionaries: {[str(d) for d in self.aruco_dicts]}')
        self.get_logger().info(f'Marker sizes: {self.marker_sizes}m')
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, self)

        self.markerNamePrefix = "aruco_marker_"
        self.filterStates = np.zeros((100,6))


    def applyFrameChange(self, posInFrame, eulerInFrame, source_frame = "base_link", target_frame = "ee_camera_link"):
        pose = Pose()
        pose.position.x = float(posInFrame[0])
        pose.position.y = float(posInFrame[1])
        pose.position.z = float(posInFrame[2])
        q_cam = R.from_euler("XYZ", eulerInFrame, degrees=False).as_quat()
        pose.orientation.x = float(q_cam[0])
        pose.orientation.y = float(q_cam[1])
        pose.orientation.z = float(q_cam[2])
        pose.orientation.w = float(q_cam[3])
        # Get the transform from source_frame to target_frame
        try:
            transform = self.tf2_buffer.lookup_transform(source_frame, target_frame, Time())
        except:
            print("Transform lookup failed")
            return None, None

        # Transform the pose
        transformed_pose = do_transform_pose(pose, transform)
        # Convert quaternion to Euler angles for logging
        tf2_quat = [
            transformed_pose.orientation.x,
            transformed_pose.orientation.y,
            transformed_pose.orientation.z,
            transformed_pose.orientation.w
        ]
        badEuler = R.from_quat(tf2_quat).as_euler("XYZ", degrees=False)
        badPos = np.array([transformed_pose.position.x, transformed_pose.position.y, transformed_pose.position.z])

        return badPos, badEuler
    
    def cameraToBase(self, posInFrame, eulerInFrame, source_frame: str = "base_link", target_frame: str = "ee_camera_link", markerID: int = 0):
        """
        Perform frame lookup and transformation using tf2_ros, similar to follow_aruco_marker.py.
        Returns the transformed pose (geometry_msgs/Pose) in the target frame.
        """
        badPos, badEuler = self.applyFrameChange(posInFrame, eulerInFrame, source_frame="base_link", target_frame="ee_camera_link")
                #self.get_logger().info(
        #    f"{source_frame} -> {target_frame}: position = {pos}, euler(rad)=[R={euler[0]:.3f}, P={euler[1]:.3f}, Y={euler[2]:.3f}]"
        #)
        if badPos is None:
            return None, None
        
        fCutoff = 1.0
        RC = 1/(2*np.pi*fCutoff)
        alpha = self.dt/(RC+self.dt)
        prevState = self.filterStates[markerID, :]
        badFilteredPos = alpha * badPos + (1 - alpha) * prevState[0:3]
        badFilteredEuler = alpha * badEuler + (1 - alpha) * prevState[3:6]


        self.filterStates[markerID, :] = np.hstack((badFilteredPos, badFilteredEuler))
        goodPos, goodEuler = self.to_good_frame(badFilteredPos,badFilteredEuler)

    
        # Broadcast marker transform (pose in world)
        self.broadcast_marker_transform(badPos, badEuler, parent_frame="base_link", child_frame=f"{self.markerNamePrefix}{markerID}")

        return goodPos, goodEuler
    
    def broadcast_marker_transform(self, marker_pos, marker_orient, parent_frame="base_link", child_frame="aruco_marker"):
            """
            Broadcasts the marker pose as a transform in the tf2 tree.
            marker_pos: [x, y, z] position in parent_frame
            marker_orient: [roll, pitch, yaw] in radians in parent_frame
            parent_frame: world/base frame
            child_frame: marker frame name
            """
            if not hasattr(self, 'tf2_broadcaster'):
                self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_frame
            t.child_frame_id = child_frame
            t.transform.translation.x = float(marker_pos[0])
            t.transform.translation.y = float(marker_pos[1])
            t.transform.translation.z = float(marker_pos[2])
            quat = R.from_euler("XYZ", marker_orient, degrees=False).as_quat()
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            self.tf2_broadcaster.sendTransform(t)
    

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
        
        return cv2.aruco.getPredefinedDictionary(dict_map[dict_name])

    def _detect_aruco_markers(self, image: np.ndarray) -> tuple:
        """
        Detect ArUco markers in the image and draw them.
        Returns the image with markers drawn and a list of pose data.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers with all dictionaries
        all_corners = []
        all_ids = []
        dict_indices = []  # to know which dict detected it
        for idx, aruco_dict in enumerate(self.aruco_dicts):
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids)
                dict_indices.extend([idx] * len(ids))
        
        # Draw detected markers
        output_image = image.copy()
        marker_poses = []
        self.cameraPose = None
        self.markerFromCamera = None

        if all_ids:
            # Convert to numpy arrays for OpenCV
            all_corners_np = all_corners
            all_ids_np = np.array(all_ids).reshape(-1, 1)
            # Draw all detected markers
            cv2.aruco.drawDetectedMarkers(output_image, all_corners_np, all_ids_np)
            
            if self.camera_matrix is None or self.dist_coeffs is None:
                self.get_logger().warn("Camera intrinsics not yet received. Skipping pose estimation.", throttle_duration_sec=0.5)
                return output_image, marker_poses
            
            # If we have camera calibration, estimate pose
            seen = set()
            for i, corner in enumerate(all_corners):
                marker_id = all_ids[i][0]
                if marker_id in seen:
                    continue
                seen.add(marker_id)
                dict_idx = dict_indices[i]
                marker_size = self.marker_sizes[dict_idx]
                
                # Estimate pose of each marker (camera -> marker)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                # Extract position (translation vector)
                position_cam = tvec[0][0]  # [x, y, z] in meters (camera frame)
                distance = np.linalg.norm(position_cam)
                
                # Convert rotation vector to euler angles (radians)
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                rot = R.from_matrix(rotation_matrix)
                roll, pitch, yaw = rot.as_euler("XYZ", degrees=False)
                euler_cam = np.array([roll, pitch, yaw])

                # --- Comparison using do_transform_pose ---
                # 1. Create Pose for marker in camera frame
                

                markerPos, markerOrient = self.cameraToBase(position_cam, euler_cam, source_frame="ee_camera_link", target_frame="base_link", markerID=marker_id)

                # Store pose data for overlay
                if markerPos is None:
                    continue

                entry = {
                    'id': marker_id,
                    'tf2Name': f"{self.markerNamePrefix}{marker_id}",
                    'positionFromCamera': position_cam,
                    'orientFromCamera': {
                        'roll': np.degrees(roll),
                        'pitch': np.degrees(pitch),
                        'yaw': np.degrees(yaw)
                    },
                    'distanceFromCamera': distance,
                    'positionInWorld': markerPos,
                    'orientInWorld': {
                        'roll': (markerOrient[0]),
                        'pitch': (markerOrient[1]),
                        'yaw': (markerOrient[2])
                    }   
                }

                marker_poses.append(entry)
                
                # Draw axis for each marker
                cv2.drawFrameAxes(
                    output_image, self.camera_matrix, self.dist_coeffs,
                    rvec[0], tvec[0], marker_size * 0.5
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
            current_marker_count = len(all_ids)
            if current_marker_count != self.last_marker_count:
                self.last_marker_count = current_marker_count
                marker_ids = [id[0] for id in all_ids]
                self.get_logger().info(f'Detected {current_marker_count} ArUco marker(s): {marker_ids}')
                # Log pose information using already computed marker_poses
                for entry in marker_poses:
                    marker_id = entry['id']
                    position_cam = entry['positionFromCamera']
                    orient_cam = entry['orientFromCamera']
                    distance_cam = entry['distanceFromCamera']
                    position_base = entry['positionInWorld']
                    orient_base = entry['orientInWorld']
                    self.get_logger().info(
                        f'  Marker {marker_id}: '
                        f'Camera [{position_cam[0]:.3f}m, {position_cam[1]:.3f}m, {position_cam[2]:.3f}m, {orient_cam['roll']:.1f}°, {orient_cam['pitch']:.1f}°, {orient_cam['yaw']:.1f}] '
                        f'World [{position_base[0]:.3f}m, {position_base[1]:.3f}m, {position_base[2]:.3f}m, {orient_base['roll']:.1f}°, {orient_base['pitch']:.1f}°, {orient_base['yaw']:.1f}] '
                    )
               
        else:
            # Log when markers disappear
            if self.last_marker_count > 0:
                self.get_logger().info('No ArUco markers detected')
                self.last_marker_count = 0
        self.marker_poses = marker_poses
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
            position = pose_data['positionFromCamera']
            orientation = pose_data['orientFromCamera']
            distance = pose_data['distanceFromCamera']

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
            if 'positionInWorld' in pose_data:
                pb = pose_data['positionInWorld']
                ob = pose_data['orientInWorld']
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



def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionViewer()
    def spawn_aruco_marker(n: Node):
        delete_cmd = [
            'gz', 'service', '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', 'name: "aruco_marker" type: MODEL'
        ]
        try:
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or 'Existing marker deleted successfully'
            n.get_logger().info(f'Aruco delete: {msg}')
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            n.get_logger().warn(f'Aruco delete failed (possibly no existing marker): {err}')
        except Exception as e:
            n.get_logger().warn(f'Aruco delete exception: {e}')


        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf',
            '-name', 'aruco_marker',
            '-x', '0', '-y', '-0.6', '-z', '0.4',
            '-R', '0', '-P', '0', '-Y', '1.57'
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
