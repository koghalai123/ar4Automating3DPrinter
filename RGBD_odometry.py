#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
import tf2_ros
from tf_transformations import quaternion_matrix, quaternion_from_matrix, euler_from_quaternion


class RGBDOdometry(Node):
    def __init__(self):
        super().__init__('rgbd_odometry')

        # Declare parameters
        self.declare_parameter('color_topic', '/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/rgbd_camera/depth_image')
        self.declare_parameter('camera_info_topic', '/rgbd_camera/camera_info')
        self.declare_parameter('publish_path', True)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('visualize', True)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('odometry_frame_id', 'odom')

        # Get parameters
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.publish_path = self.get_parameter('publish_path').get_parameter_value().bool_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter('odometry_frame_id').get_parameter_value().string_value

        # Initialize variables
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_intrinsics = None
        self.camera_info_received = False

        # Odometry state
        self.prev_rgbd = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix in camera frame
        self.initial_pose_offset = np.eye(4)  # Offset to align with robot frame
        self.trajectory = [self.current_pose.copy()]
        
        # Camera to end-effector transform (from URDF: <pose>0 0 0 1.5708 -1.5708 0</pose>)
        # This accounts for the camera's orientation relative to the end-effector
        # Roll: 90°, Pitch: -90°, Yaw: 0°
        from tf_transformations import quaternion_from_euler
        roll_cam = 1.5708  # 90 degrees
        pitch_cam = -1.5708  # -90 degrees
        yaw_cam = 0.0
        q_cam = quaternion_from_euler(roll_cam, pitch_cam, yaw_cam)
        self.camera_to_ee_transform = quaternion_matrix(q_cam)
        self.camera_to_ee_transform[:3, 3] = [0, 0, 0]  # No translation offset in this case

        # Open3D odometry option
        self.option = o3d.pipelines.odometry.OdometryOption()
        # Set iteration counts for the multi-scale approach
        self.option.iteration_number_per_pyramid_level = o3d.utility.IntVector([20, 10, 5])
        
        # Depth processing parameters
        self.min_depth = 0.3
        self.max_depth = 4.0

        # ROS subscriptions
        self.color_sub = self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )

        # ROS publishers
        if self.publish_path:
            self.path_pub = self.create_publisher(Path, '/rgbd_odometry/path', 10)
            self.pose_pub = self.create_publisher(PoseStamped, '/rgbd_odometry/pose', 10)
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.odom_frame_id

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Service for setting initial pose
        self.set_pose_sub = self.create_subscription(
            PoseStamped,
            '/rgbd_odometry/set_initial_pose',
            self.set_initial_pose_callback,
            10
        )
        self.reset_srv = self.create_service(
            Trigger,
            '/rgbd_odometry/reset',
            self.reset_odometry_callback
        )

        # Processing timer
        self.timer = self.create_timer(1.0 / 30.0, self.process_odometry)

        # Visualization timer (for rendering)
        if self.visualize:
            self.vis_timer = self.create_timer(1.0 / 30.0, self.render_visualization)

        self.get_logger().info(f'RGB-D Odometry node initialized')
        self.get_logger().info(f'Color topic: {color_topic}')
        self.get_logger().info(f'Depth topic: {depth_topic}')
        self.get_logger().info(f'Camera info topic: {camera_info_topic}')

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            # Extract camera intrinsics from CameraInfo message
            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]
            width = msg.width
            height = msg.height

            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width, height, fx, fy, cx, cy
            )
            self.camera_info_received = True
            self.get_logger().info(f'Camera intrinsics received: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}')

    def set_initial_pose_callback(self, msg: PoseStamped):
        """Set the initial pose offset to align odometry with robot frame
        
        This resets the odometry and sets the initial pose to the provided value.
        """
        try:
            # Convert pose message to transformation matrix
            pos = msg.pose.position
            ori = msg.pose.orientation
            
            # Create transformation matrix from position and quaternion
            quat = [ori.x, ori.y, ori.z, ori.w]
            transform = quaternion_matrix(quat)
            transform[0, 3] = pos.x
            transform[1, 3] = pos.y
            transform[2, 3] = pos.z
            
            # Reset odometry and set the provided pose as the starting point
            self.current_pose = np.eye(4)  # Reset to identity
            self.initial_pose_offset = transform  # Set the desired starting pose
            self.trajectory = [self.current_pose.copy()]
            self.prev_rgbd = None  # Force reinitialization of odometry tracking
            
            if self.publish_path:
                self.path_msg.poses.clear()
            
            self.get_logger().info(
                f'Initial pose set: pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), '
                f'ori=({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})'
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to set initial pose: {str(e)}')

    def reset_odometry_callback(self, request, response):
        """Reset odometry to identity (zero pose)"""
        try:
            self.current_pose = np.eye(4)
            self.initial_pose_offset = np.eye(4)
            self.trajectory = [self.current_pose.copy()]
            self.prev_rgbd = None
            if self.publish_path:
                self.path_msg.poses.clear()
            
            response.success = True
            response.message = 'Odometry reset successfully'
            self.get_logger().info('Odometry reset to zero')
            
        except Exception as e:
            response.success = False
            response.message = f'Failed to reset odometry: {str(e)}'
            self.get_logger().error(response.message)
        
        return response

    def color_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Color image error: {str(e)}')

    def depth_callback(self, msg: Image):
        try:
            # Convert depth image to float32 (meters)
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Handle different depth formats
            if depth.dtype == np.uint16:
                # Assume millimeters if uint16, convert to meters
                self.latest_depth = depth.astype(np.float32) / 1000.0
            elif depth.dtype == np.float32 or depth.dtype == np.float64:
                self.latest_depth = depth.astype(np.float32)
            else:
                self.get_logger().warn(f'Unexpected depth format: {depth.dtype}')
                self.latest_depth = depth.astype(np.float32)
                
        except Exception as e:
            self.get_logger().error(f'Depth image error: {str(e)}')

    def process_odometry(self):
        if not self.camera_info_received:
            return

        if self.latest_color is None or self.latest_depth is None:
            return

        try:
            # Get current images
            color_img = self.latest_color.copy()
            depth_img = self.latest_depth.copy()

            # Ensure images have the same size
            if color_img.shape[:2] != depth_img.shape[:2]:
                depth_img = cv2.resize(
                    depth_img, 
                    (color_img.shape[1], color_img.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )

            # Create Open3D RGB-D image
            color_o3d = o3d.geometry.Image(color_img)
            depth_o3d = o3d.geometry.Image(depth_img)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, 
                depth_o3d, 
                depth_scale=1.0,  # Already in meters
                depth_trunc=self.max_depth,
                convert_rgb_to_intensity=False
            )

            # Compute odometry
            if self.prev_rgbd is not None:
                # Compute transformation from previous to current frame
                result = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_image,
                    self.prev_rgbd,
                    self.camera_intrinsics,
                    np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    self.option
                )
                
                # Extract transformation matrix (result is [success, trans, info])
                success = result[0]
                trans = result[1]

                if success:
                    # Update cumulative pose (inverse because we go from current to previous)
                    self.current_pose = self.current_pose @ np.linalg.inv(trans)
                    self.trajectory.append(self.current_pose.copy())
                    
                    # Log corrected pose information (what's actually published)
                    corrected_pose = self.initial_pose_offset @ self.camera_to_ee_transform @ self.current_pose
                    pos = corrected_pose[:3, 3]
                    self.get_logger().info(
                        f'Pose: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}',
                        throttle_duration_sec=1.0
                    )
                    
                    # Publish pose and path
                    if self.publish_path:
                        self.publish_pose_and_path()
                    
                    if self.publish_tf:
                        self.publish_transform()
                    
                else:
                    self.get_logger().warn('Odometry computation failed', throttle_duration_sec=1.0)

            # Store current frame for next iteration
            self.prev_rgbd = rgbd_image

        except Exception as e:
            self.get_logger().error(f'Odometry processing error: {str(e)}')

    def publish_pose_and_path(self):
        timestamp = self.get_clock().now().to_msg()
        
        # Apply camera-to-end-effector transform, then initial pose offset to get pose in robot frame
        # current_pose is in camera frame -> transform to end-effector frame -> apply initial offset
        corrected_pose = self.initial_pose_offset @ self.camera_to_ee_transform @ self.current_pose
        
        # Publish current pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = self.odom_frame_id
        
        # Extract position and orientation from transformation matrix
        pose_msg.pose.position.x = float(corrected_pose[0, 3])
        pose_msg.pose.position.y = float(corrected_pose[1, 3])
        pose_msg.pose.position.z = float(corrected_pose[2, 3])
        
        # Convert rotation matrix to quaternion
        rotation_matrix = corrected_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)
        
        # Add to path and publish
        self.path_msg.header.stamp = timestamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

    def publish_transform(self):
        # Apply camera-to-end-effector transform, then initial pose offset to get pose in robot frame
        corrected_pose = self.initial_pose_offset @ self.camera_to_ee_transform @ self.current_pose
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.frame_id
        
        t.transform.translation.x = float(corrected_pose[0, 3])
        t.transform.translation.y = float(corrected_pose[1, 3])
        t.transform.translation.z = float(corrected_pose[2, 3])
        
        rotation_matrix = corrected_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def render_visualization(self):
        """Render RGB-D feed with odometry information overlaid"""
        if self.latest_color is None or self.latest_depth is None:
            return

        try:
            # Get current images
            color_img = self.latest_color.copy()
            depth_raw = self.latest_depth.copy()

            # Convert color from RGB to BGR for OpenCV display
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # Resize depth to match color height if needed
            if depth_raw.shape[:2] != color_bgr.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)

            # Visualize depth
            depth_vis = self._visualize_depth(depth_raw)

            # Ensure both images are 3-channel BGR
            if len(color_bgr.shape) == 2:
                color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_GRAY2BGR)
            if len(depth_vis.shape) == 2:
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

            # Draw odometry information on color image
            color_bgr = self._draw_odometry_info(color_bgr)

            # Combine color and depth side by side
            combined = np.hstack([color_bgr, depth_vis])

            # Display combined image
            cv2.imshow('RGB-D Odometry', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Quit key pressed, shutting down...')
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Visualization error: {str(e)}')

    def _visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to colorized visualization"""
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            d = depth.copy()
            # Mask invalid values
            mask = np.isfinite(d) & (d > 0)
            if not np.any(mask):
                return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            # Robust scaling using percentiles
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

        return cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

    def _draw_odometry_info(self, img: np.ndarray) -> np.ndarray:
        """Draw odometry pose information on the image"""
        # Apply initial pose offset to get corrected pose
        corrected_pose = self.initial_pose_offset @ self.current_pose
        
        # Extract position from corrected pose
        pos = corrected_pose[:3, 3]
        
        # Extract rotation and convert to Euler angles (roll, pitch, yaw) in degrees
        rotation_matrix = corrected_pose[:3, :3]
        euler = self._rotation_matrix_to_euler_angles(rotation_matrix)
        roll, pitch, yaw = np.degrees(euler)
        
        # Draw semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (310, 200), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Draw text with pose information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green text
        thickness = 1
        
        cv2.putText(img, 'RGB-D Odometry', (10, 25), font, 0.6, (0, 255, 255), 2)
        
        # Position
        cv2.putText(img, f'Position:', (10, 50), font, font_scale, color, thickness)
        cv2.putText(img, f'  X: {pos[0]:7.3f} m', (10, 70), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img, f'  Y: {pos[1]:7.3f} m', (10, 90), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img, f'  Z: {pos[2]:7.3f} m', (10, 110), font, font_scale, (255, 255, 255), thickness)
        
        # Orientation
        cv2.putText(img, f'Orientation:', (10, 135), font, font_scale, color, thickness)
        cv2.putText(img, f'  Roll:  {roll:7.2f} deg', (10, 155), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img, f'  Pitch: {pitch:7.2f} deg', (10, 175), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img, f'  Yaw:   {yaw:7.2f} deg', (10, 195), font, font_scale, (255, 255, 255), thickness)
        
        # Draw trajectory length in bottom corner
        traj_length = len(self.trajectory)
        img_height = img.shape[0]
        cv2.rectangle(overlay, (5, img_height - 30), (150, img_height - 5), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, f'Frames: {traj_length}', (10, img_height - 10), font, font_scale, (255, 200, 0), thickness)
        
        return img
    
    @staticmethod
    def _rotation_matrix_to_euler_angles(R):
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw) in radians
        Uses tf_transformations for consistency with ROS conventions
        """
        # Create 4x4 matrix from 3x3 rotation
        T = np.eye(4)
        T[:3, :3] = R
        # Convert to quaternion using tf_transformations
        q = quaternion_from_matrix(T)
        # Convert quaternion to euler (returns in 'sxyz' convention by default)
        # quaternion_from_matrix returns [w, x, y, z]
        euler = euler_from_quaternion([q[1], q[2], q[3], q[0]])  # Convert to [x, y, z, w] format
        return np.array(euler)

    def update_visualization(self, rgbd_image):
        # This method is no longer used but kept for compatibility
        pass

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]
        Uses tf_transformations for consistency"""
        # Create 4x4 matrix from 3x3 rotation
        T = np.eye(4)
        T[:3, :3] = R
        # Use tf_transformations which is consistent with ROS conventions
        q = quaternion_from_matrix(T)
        # quaternion_from_matrix returns [w, x, y, z], we need [x, y, z, w]
        return np.array([q[1], q[2], q[3], q[0]])

    def cleanup(self):
        if self.visualize:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = RGBDOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
