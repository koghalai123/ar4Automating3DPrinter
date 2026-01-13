#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Stage definitions
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(21, 21),
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


def featureTracking(image_ref, image_cur, px_ref):
    """
    Track features between two frames using Lucas-Kanade optical flow
    """
    # Check if we have valid features to track
    if px_ref is None or len(px_ref) == 0:
        return np.array([]), np.array([])
    
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    # Flatten the status array properly
    st = st.flatten()
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2


class PinholeCamera:
    """Camera model with intrinsic parameters"""
    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    """Visual Odometry class for monocular camera"""
    def __init__(self, cam, scale=0.1):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        
        # Current pose
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
        # Scale factor (since monocular SLAM has scale ambiguity)
        self.scale = scale
        
        # Feature detector (FAST)
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        
        # For visualization
        self.trajectory = []
        
    def processFirstFrame(self):
        """Process the very first frame to detect initial features"""
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME
        
    def processSecondFrame(self):
        """Process the second frame to establish initial pose"""
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        # Need at least 5 points for essential matrix
        if len(self.px_cur) < 5:
            # Re-detect features and stay in second frame stage
            self.px_ref = self.detector.detect(self.new_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
            return
        
        E, mask = cv2.findEssentialMat(
            self.px_cur, self.px_ref, 
            focal=self.focal, pp=self.pp, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        # Check if essential matrix is valid (must be 3x3)
        if E is None or E.shape != (3, 3):
            # Re-detect features and stay in second frame stage
            self.px_ref = self.detector.detect(self.new_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
            return
            
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(
            E, self.px_cur, self.px_ref,
            focal=self.focal, pp=self.pp
        )
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur
        
    def processFrame(self):
        """Process subsequent frames for continuous odometry"""
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        # Need at least 5 points for essential matrix
        if len(self.px_cur) < 5:
            # Detect new features
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.px_ref = self.px_cur
            return
        
        # Calculate mean displacement of tracked features
        mean_displacement = np.mean(np.linalg.norm(self.px_cur - self.px_ref, axis=1))
        
        # Motion threshold: only update if significant motion detected (> 0.5 pixels)
        if mean_displacement < 0.5:
            # No significant motion, don't update pose
            self.px_ref = self.px_cur
            return
        
        E, mask = cv2.findEssentialMat(
            self.px_cur, self.px_ref,
            focal=self.focal, pp=self.pp,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        # Check if essential matrix is valid (must be 3x3)
        if E is None or E.shape != (3, 3):
            # Detect new features
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.px_ref = self.px_cur
            return
        
        _, R, t, mask = cv2.recoverPose(
            E, self.px_cur, self.px_ref,
            focal=self.focal, pp=self.pp
        )
        
        # Check if rotation and translation are significant
        # Only update if we have actual motion (avoid drift from noise)
        rotation_change = np.linalg.norm(R - np.eye(3))
        translation_norm = np.linalg.norm(t)
        
        # Only update pose if motion is significant
        if rotation_change > 0.01 or translation_norm > 0.5:
            # Update pose with scale
            # Note: In real monocular VO, you'd need external info for absolute scale
            # Here we use a constant scale factor
            self.cur_t = self.cur_t + self.scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        
        # Detect new features if we have too few
        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            
        self.px_ref = self.px_cur
        
    def update(self, img):
        """
        Update visual odometry with a new frame
        Args:
            img: grayscale image
        Returns:
            current position (x, y, z)
        """
        assert(img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), \
            "Frame: provided image has not the same size as the camera model or image is not grayscale"
        
        self.new_frame = img
        
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame()
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()
            
        self.last_frame = self.new_frame
        
        # Store trajectory
        if self.cur_t is not None:
            self.trajectory.append(self.cur_t.copy())
            
        return self.cur_t


class VisualOdometryROS(Node):
    """ROS2 node for visual odometry"""
    def __init__(self):
        super().__init__('visual_odometry_ros')
        
        # Create CV Bridge
        self.bridge = CvBridge()
        
        # Camera parameters (adjust for your camera)
        # These are approximate for the simulated camera (320x240)
        self.cam = PinholeCamera(
            width=320, height=240,
            fx=320.0, fy=320.0,  # Focal length
            cx=160.0, cy=120.0   # Principal point (center)
        )
        
        # Initialize visual odometry
        self.vo = VisualOdometry(self.cam, scale=0.1)
        
        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/rgb_camera_moving/image',
            self.image_callback,
            10
        )
        
        # Publishers for pose and path
        self.pose_pub = self.create_publisher(PoseStamped, '/vo/pose', 10)
        self.path_pub = self.create_publisher(Path, '/vo/path', 10)
        
        # Path message for trajectory
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'world'
        
        # Statistics
        self.frame_count = 0
        self.total_displacement = 0.0
        self.last_position = np.zeros((3, 1))
        
        self.get_logger().info('Visual Odometry ROS node started')
        self.get_logger().info('Subscribing to /rgb_camera_moving/image')
        self.get_logger().info('Publishing pose on /vo/pose and path on /vo/path')
        self.get_logger().info('Press Ctrl+C to quit')
        
    def image_callback(self, msg):
        """Process incoming images for visual odometry"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Update visual odometry
            cur_t = self.vo.update(gray)
            
            self.frame_count += 1
            
            # Only publish after initialization (stage >= DEFAULT_FRAME)
            if self.vo.frame_stage == STAGE_DEFAULT_FRAME and cur_t is not None:
                # Convert rotation matrix to quaternion
                rot = R.from_matrix(self.vo.cur_R)
                quat = rot.as_quat()  # Returns [x, y, z, w]
                euler = rot.as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw in degrees
                
                # Publish pose
                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = 'world'
                pose_msg.pose.position.x = float(cur_t[0, 0])
                pose_msg.pose.position.y = float(cur_t[1, 0])
                pose_msg.pose.position.z = float(cur_t[2, 0])
                
                # Set orientation as quaternion
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])
                
                self.pose_pub.publish(pose_msg)
                
                # Add to path and publish
                self.path_msg.header.stamp = msg.header.stamp
                self.path_msg.poses.append(pose_msg)
                
                # Keep only last 1000 poses
                if len(self.path_msg.poses) > 1000:
                    self.path_msg.poses.pop(0)
                    
                self.path_pub.publish(self.path_msg)
                
                # Calculate displacement
                displacement = np.linalg.norm(cur_t - self.last_position)
                self.total_displacement += displacement
                self.last_position = cur_t.copy()
                
                # Log periodically
                if self.frame_count % 30 == 0:
                    self.get_logger().info(
                        f'Frame {self.frame_count}: '
                        f'Position: [{cur_t[0,0]:.3f}, {cur_t[1,0]:.3f}, {cur_t[2,0]:.3f}], '
                        f'Orientation (RPY): [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°], '
                        f'Total distance: {self.total_displacement:.3f}m, '
                        f'Features: {len(self.vo.px_ref) if self.vo.px_ref is not None else 0}'
                    )
            
            # Visualize features on image
            if self.vo.px_ref is not None and len(self.vo.px_ref) > 0:
                for pt in self.vo.px_ref:
                    cv2.circle(cv_image, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                    
            # Add info overlay
            cv2.putText(cv_image, f'Frame: {self.frame_count}', 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if self.vo.px_ref is not None:
                cv2.putText(cv_image, f'Features: {len(self.vo.px_ref)}', 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if self.vo.frame_stage == STAGE_DEFAULT_FRAME and cur_t is not None:
                # Get orientation for display
                rot = R.from_matrix(self.vo.cur_R)
                euler = rot.as_euler('xyz', degrees=True)
                
                cv2.putText(cv_image, f'Pos: [{cur_t[0,0]:.2f}, {cur_t[1,0]:.2f}, {cur_t[2,0]:.2f}]', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(cv_image, f'Ori (RPY): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg', 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(cv_image, f'Distance: {self.total_displacement:.2f}m', 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            # Display
            cv2.imshow('Visual Odometry - Feature Tracking', cv_image)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit key pressed, shutting down...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    vo_node = VisualOdometryROS()
    
    try:
        rclpy.spin(vo_node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        vo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
