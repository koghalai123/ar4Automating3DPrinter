#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import time
import threading


class VisualSLAM(Node):
    def __init__(self):
        super().__init__('visual_slam')
        
        # Create CV Bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/rgb_camera_moving/image',
            self.image_callback,
            10
        )
        
        # Visual odometry variables
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Camera pose tracking
        self.camera_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.camera_rotation = np.eye(3)  # Rotation matrix
        self.trajectory = []  # List of positions over time
        
        # Feature detector (ORB is fast and works well for real-time)
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Feature matcher
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera intrinsics (approximate for the simulated camera)
        # These should ideally come from camera_info topic
        self.focal_length = 320  # pixels (approximate)
        self.principal_point = (160, 120)  # center of 320x240 image
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype=float)
        
        # Frame counter
        self.frame_count = 0
        self.process_every_n_frames = 1  # Process every frame
        
        # Statistics
        self.total_displacement = 0.0
        self.last_displacement = np.array([0.0, 0.0, 0.0])
        
        self.get_logger().info('Visual SLAM node started')
        self.get_logger().info('Subscribing to /rgb_camera_moving/image')
        self.get_logger().info('Press "q" in the window to quit, "r" to reset')
        
    def spawn_objects(self, scene_type='random', count=20, seed=None):
        """
        Spawn colorful objects using the object_generator.py script
        
        Args:
            scene_type: 'demo' or 'random'
            count: Number of objects (for random scene)
            seed: Random seed for reproducibility
        """
        self.get_logger().info(f'Spawning {count} {scene_type} objects into Gazebo...')
        try:
            script_path = '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/object_generator.py'
            
            # Build command with arguments
            cmd = ['python3', script_path, '--scene', scene_type, '--count', str(count)]
            if seed is not None:
                cmd.extend(['--seed', str(seed)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.get_logger().info('Objects spawned successfully!')
                self.get_logger().info(f'Scene: {scene_type}, Count: {count}')
                if result.stdout:
                    self.get_logger().info(f'Output: {result.stdout.strip()}')
            else:
                self.get_logger().error(f'Failed to spawn objects: {result.stderr}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('Spawning objects timed out')
        except Exception as e:
            self.get_logger().error(f'Error spawning objects: {str(e)}')
    
    def estimate_pose_from_essential(self, E, matched_pts1, matched_pts2):
        """Estimate camera pose from essential matrix"""
        # Decompose essential matrix
        _, R, t, mask = cv2.recoverPose(E, matched_pts1, matched_pts2, self.camera_matrix)
        return R, t, mask
    
    def triangulate_points(self, R, t, matched_pts1, matched_pts2):
        """Triangulate 3D points from matched 2D points"""
        # Create projection matrices
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        
        # Convert to homogeneous coordinates
        points_4d = cv2.triangulatePoints(
            self.camera_matrix @ P1,
            self.camera_matrix @ P2,
            matched_pts1.T,
            matched_pts2.T
        )
        
        # Convert to 3D
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T
    
    def process_frame(self, frame):
        """Process frame for visual odometry"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None or descriptors is None:
            # First frame or no features detected
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return frame, False
        
        if self.prev_descriptors is None or len(keypoints) < 10:
            # Not enough features
            return frame, False
        
        # Match features
        matches = self.bf_matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            self.get_logger().warn(f'Not enough good matches: {len(good_matches)}')
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return frame, False
        
        # Extract matched points
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            self.get_logger().warn('Could not compute essential matrix')
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return frame, False
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
        
        # Update camera pose
        # Scale factor (this is a limitation of monocular SLAM - scale is arbitrary)
        scale = 0.1  # Adjust this based on your scene
        
        # Update rotation and position
        self.camera_rotation = R @ self.camera_rotation
        displacement = scale * (self.camera_rotation @ t).flatten()
        self.camera_position += displacement
        
        # Track trajectory
        self.trajectory.append(self.camera_position.copy())
        
        # Keep only last 100 positions
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)
        
        # Calculate displacement magnitude
        self.last_displacement = displacement
        displacement_magnitude = np.linalg.norm(displacement)
        self.total_displacement += displacement_magnitude
        
        # Update for next iteration
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return frame, True
    
    def draw_overlay(self, frame):
        """Draw SLAM information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Draw trajectory on a separate view (top-right corner)
        traj_size = 200
        traj_display = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)
        
        if len(self.trajectory) > 1:
            # Scale trajectory to fit in display
            traj_array = np.array(self.trajectory)
            
            # Use X-Y plane for top-down view
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]
            
            # Normalize to display size
            if np.ptp(x_coords) > 0 and np.ptp(y_coords) > 0:
                x_norm = ((x_coords - np.min(x_coords)) / np.ptp(x_coords) * (traj_size - 40) + 20).astype(int)
                y_norm = ((y_coords - np.min(y_coords)) / np.ptp(y_coords) * (traj_size - 40) + 20).astype(int)
                
                # Draw trajectory
                for i in range(1, len(x_norm)):
                    cv2.line(traj_display, 
                            (x_norm[i-1], y_norm[i-1]), 
                            (x_norm[i], y_norm[i]), 
                            (0, 255, 0), 2)
                
                # Draw current position
                cv2.circle(traj_display, (x_norm[-1], y_norm[-1]), 5, (0, 0, 255), -1)
        
        # Add trajectory display to frame
        if width > traj_size and height > traj_size:
            frame[10:10+traj_size, width-traj_size-10:width-10] = traj_display
        
        # Draw text information
        y_offset = 20
        line_height = 25
        
        # Background rectangle for text
        cv2.rectangle(frame, (5, 5), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 180), (255, 255, 255), 2)
        
        # Camera position
        cv2.putText(frame, f"Position (m):", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  X: {self.camera_position[0]:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Y: {self.camera_position[1]:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Z: {self.camera_position[2]:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Last displacement
        cv2.putText(frame, f"Displacement (m/frame):", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  dX: {self.last_displacement[0]:.4f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  dY: {self.last_displacement[1]:.4f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Total displacement
        y_offset = height - 30
        cv2.putText(frame, f"Total Distance: {self.total_displacement:.3f} m", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.frame_count += 1
            
            # Process frame for visual odometry
            if self.frame_count % self.process_every_n_frames == 0:
                processed_frame, success = self.process_frame(cv_image.copy())
                
                if success:
                    # Draw overlay with SLAM info
                    display_frame = self.draw_overlay(processed_frame)
                else:
                    display_frame = cv_image
            else:
                display_frame = cv_image
            
            # Display the frame
            cv2.imshow('Visual SLAM - Camera Feed', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit key pressed, shutting down...')
                rclpy.shutdown()
            elif key == ord('r'):
                # Reset SLAM
                self.get_logger().info('Resetting SLAM...')
                self.camera_position = np.array([0.0, 0.0, 0.0])
                self.camera_rotation = np.eye(3)
                self.trajectory = []
                self.total_displacement = 0.0
                self.prev_frame = None
                self.prev_keypoints = None
                self.prev_descriptors = None
                
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    slam = VisualSLAM()
    
    # Spawn objects in a separate thread to not block the node
    spawn_thread = threading.Thread(target=slam.spawn_objects)
    spawn_thread.start()
    
    try:
        rclpy.spin(slam)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        slam.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
