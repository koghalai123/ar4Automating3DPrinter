#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from ArucoDetector import ArucoDetectionViewer

class HandEyeCalibration(ArucoDetectionViewer):
    def __init__(self):
        super().__init__()

        # Calibration data storage
        self.camera_poses = []
        self.robot_poses = []

        # Define a series of robot poses for calibration
        base_poses = [
            [0.32, 0.0, 0.42, 0.0, 0.0, 0.0],   # Example pose 1 (x, y, z, roll, pitch, yaw)
        ]
        self.calibration_poses = []
        for pose in base_poses:
            self.calibration_poses.append(pose)
            for _ in range(50):
                offset = np.random.uniform(-0.1, 0.1, 6)
                offset[3:] = np.random.uniform(-0.08, 0.08, 3)
                self.calibration_poses.append(np.array(pose) + offset)

        # Wait for Aruco and Pose to initialize
        rclpy.spin_once(self, timeout_sec=2)

        # Start the calibration process
        self.collect_calibration_data()


    def move_robot(self, pose):
        self.move_to_pose(pose[:3], pose[3:])
        rclpy.spin_once(self, timeout_sec=5)

    def collect_calibration_data(self):
        """Collect camera and robot poses for calibration."""
        self.get_logger().info("Collecting calibration data...")

        for pose in self.calibration_poses:
            self.move_robot(pose)
            rclpy.spin_once(self, timeout_sec=2)

            # Get robot pose
            robot_pose = self.get_frame()
            self.get_logger().info(f"Robot Pose: {robot_pose}")

            # Get ArUco marker pose from camera
            if self.markerFromCamera is not None and self.cameraPose is not None:
                position_cam, euler_cam = self.markerFromCamera
                camera_pos, camera_euler = self.cameraPose

                self.camera_poses.append(self.pose_to_homogeneous_matrix(position_cam, euler_cam))
                self.robot_poses.append(self.pose_to_homogeneous_matrix(robot_pose[:3], robot_pose[3:]))

                self.get_logger().info(f"Camera Pose: {camera_pos, camera_euler} with respect to the world")
                self.get_logger().info(f"Marker Pose from Camera: {position_cam, euler_cam}")
            else:
                self.get_logger().warn("No marker detected, skipping pose.")
                continue

        self.perform_hand_eye_calibration()

    def pose_to_homogeneous_matrix(self, position, euler_angles):
        """Convert position and Euler angles to a 4x4 homogeneous transformation matrix."""
        rot = R.from_euler("XYZ", euler_angles, degrees=False)
        rotation_matrix = rot.as_matrix()

        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        homogeneous_matrix[:3, 3] = position

        return homogeneous_matrix

    def perform_hand_eye_calibration(self):
        """Perform hand-eye calibration using collected data."""
        self.get_logger().info("Performing hand-eye calibration...")

        if len(self.camera_poses) < 4 or len(self.robot_poses) < 4:
            self.get_logger().error("Not enough data to perform calibration.  Collect more poses.")
            return

        # Convert poses to rotation matrices and translation vectors
        camera_rots = [pose[:3, :3] for pose in self.camera_poses]
        camera_trans = [pose[:3, 3] for pose in self.camera_poses]
        robot_rots = [pose[:3, :3] for pose in self.robot_poses]
        robot_trans = [pose[:3, 3] for pose in self.robot_poses]

        # Perform hand-eye calibration
        rotation, translation = cv2.calibrateHandEye(
            robot_rots, robot_trans, camera_rots, camera_trans
        )

        # Optimize the calibration to minimize marker pose variance in base frame
        self.get_logger().info("Refining calibration with optimization...")
        rotation, translation = self.optimize_calibration(rotation, translation)

        # Convert rotation matrix to Euler angles
        r = R.from_matrix(rotation)
        euler = r.as_euler("XYZ", degrees=False)
        pose_str = f"X: {translation[0][0]:.4f}, Y: {translation[1][0]:.4f}, Z: {translation[2][0]:.4f}, R: {euler[0]:.4f}, P: {euler[1]:.4f}, Y: {euler[2]:.4f}"

        # Print the results
        self.get_logger().info("Hand-eye calibration successful:")
        self.get_logger().info(f"Rotation:\n{rotation}")
        self.get_logger().info(f"Translation:\n{translation}")
        self.get_logger().info(f"Readable Pose: {pose_str}")

        with open("hand_eye_calibration_result.txt", "w") as f:
            f.write(f"Rotation:\n{rotation}\nTranslation:\n{translation}\nReadable Pose: {pose_str}\n")

    def optimize_calibration(self, initial_rotation, initial_translation):
        """
        Optimize the hand-eye calibration by minimizing the distance between 
        calculated marker pose and true marker pose in the base frame.
        """
        # True marker position
        true_marker_pos = np.array([0.6, 0.0, 0.4])
        
        def cost_function(params):
            rx, ry, rz = params
            tx, ty, tz = 0.0, 0.0, 0.0
            
            # Construct X (Gripper -> Camera)
            rot = R.from_rotvec([rx, ry, rz])
            X = np.eye(4)
            X[:3, :3] = rot.as_matrix()
            X[:3, 3] = [tx, ty, tz]
            
            total_error = 0.0
            
            for i in range(len(self.robot_poses)):
                T_bg = self.robot_poses[i] # Base -> Gripper
                T_cm = self.camera_poses[i] # Camera -> Marker
                
                # T_bm = T_bg * X * T_cm
                T_bm = T_bg @ X @ T_cm
                
                # Calculate distance to true marker position
                dist = np.linalg.norm(T_bm[:3, 3] - true_marker_pos)
                total_error += dist
            
            return total_error

        best_res = None
        best_cost = float('inf')
        
        # Perform optimization with random restarts
        num_restarts = 500
        self.get_logger().info(f"Refining calibration with optimization ({num_restarts} random restarts)...")
        
        for i in range(num_restarts):
            # Random orientation [0, 2pi]
            random_euler = np.random.uniform(0, 2*np.pi, 3)
            r = R.from_euler('XYZ', random_euler, degrees=False)
            initial_rotvec = r.as_rotvec()
            
            # Zero translation as requested
            x0 = initial_rotvec
            
            res = minimize(cost_function, x0, method='Nelder-Mead', tol=1e-5)
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_res = res

        self.get_logger().info(f"Optimization result: {best_res.message}, Best Cost: {best_res.fun:.6f}")
        
        optimized_params = best_res.x
        rx, ry, rz = optimized_params
        tx, ty, tz = 0.0, 0.0, 0.0
        
        rot = R.from_rotvec([rx, ry, rz])
        return rot.as_matrix(), np.array([[tx], [ty], [tz]])

def main(args=None):
    rclpy.init(args=args)
    calibration_node = HandEyeCalibration()

    try:
        rclpy.spin(calibration_node)
    except KeyboardInterrupt:
        pass
    finally:
        calibration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()