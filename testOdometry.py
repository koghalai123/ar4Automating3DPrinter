#!/usr/bin/env python3

import time
import argparse
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from tf_transformations import (
    quaternion_from_euler, 
    euler_from_quaternion,
    quaternion_matrix,
    quaternion_multiply,
    quaternion_inverse
)

from pymoveit2 import MoveIt2


class OdometryTester(Node):
    def __init__(self):
        super().__init__("odometry_tester")

        # MoveIt2 interface for robot control
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 2.0

        # Subscribe to odometry pose
        self.odometry_sub = self.create_subscription(
            PoseStamped,
            '/rgbd_odometry/pose',
            self.odometry_callback,
            10
        )

        # Publisher to set initial odometry pose
        self.set_odom_pose_pub = self.create_publisher(
            PoseStamped,
            '/rgbd_odometry/set_initial_pose',
            10
        )

        # Store odometry data
        self.latest_odom_pose = None
        self.odom_history = deque(maxlen=1000)
        
        # Test results storage
        self.test_results = []

        self.get_logger().info("Odometry Tester initialized")

    def odometry_callback(self, msg: PoseStamped):
        """Store latest odometry pose"""
        self.latest_odom_pose = msg
        self.odom_history.append({
            'timestamp': time.time(),
            'pose': msg.pose
        })

    def get_current_robot_pose(self):
        """Get current robot end-effector pose from MoveIt2"""
        fk = self.moveit2.compute_fk()
        if fk is None:
            self.get_logger().warn("compute_fk returned None")
            return None, None
        
        pose_msg = fk[0] if isinstance(fk, list) else fk
        pos = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ], dtype=float)
        
        quat = [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        rpy = np.array([roll, pitch, yaw], dtype=float)
        
        return pos, rpy

    def get_odometry_pose(self):
        """Get latest odometry pose in position and RPY format"""
        if self.latest_odom_pose is None:
            return None, None
        
        pose = self.latest_odom_pose.pose
        pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float)
        
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        rpy = np.array([roll, pitch, yaw], dtype=float)
        
        return pos, rpy

    def get_odometry_quaternion(self):
        """Get latest odometry pose with quaternion"""
        if self.latest_odom_pose is None:
            return None, None
        
        pose = self.latest_odom_pose.pose
        pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float)
        
        quat = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ], dtype=float)
        
        return pos, quat

    @staticmethod
    def compute_rotation_delta(quat_initial, quat_final):
        """Compute the rotation delta between two quaternions and return as RPY
        
        Delta rotation: R_delta = R_final * R_initial^(-1)
        In quaternion form: q_delta = q_final * q_initial^(-1)
        """
        # Compute delta quaternion
        q_initial_inv = quaternion_inverse(quat_initial)
        q_delta = quaternion_multiply(quat_final, q_initial_inv)
        
        # Convert to euler angles
        euler = euler_from_quaternion(q_delta)
        return np.array(euler)

    def move_to_pose(self, pos_xyz, rpy, label=""):
        """Move robot to target pose"""
        q = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        q_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        self.get_logger().info(f"Moving to {label}: pos={pos_xyz}, rpy(deg)={np.rad2deg(rpy)}")
        
        self.moveit2.move_to_pose(
            position=Point(x=pos_xyz[0], y=pos_xyz[1], z=pos_xyz[2]),
            quat_xyzw=q_msg
        )
        self.moveit2.wait_until_executed()

    def move_delta(self, delta_pos=np.zeros(3), delta_rpy=np.zeros(3), label=""):
        """Move robot by a delta from current position"""
        curr_pos, curr_rpy = self.get_current_robot_pose()
        if curr_pos is None:
            self.get_logger().error("Could not read current pose")
            return False
        
        target_pos = curr_pos + np.asarray(delta_pos, dtype=float)
        target_rpy = curr_rpy + np.asarray(delta_rpy, dtype=float)
        
        self.move_to_pose(target_pos, target_rpy, label)
        return True

    def wait_for_stability(self, wait_time=2.0):
        """Wait for robot and odometry to stabilize"""
        self.get_logger().info(f"Waiting {wait_time}s for stability...")
        time.sleep(wait_time)

    def set_odometry_initial_pose(self):
        """Set the odometry's initial pose to match the robot's current pose"""
        curr_pos, curr_rpy = self.get_current_robot_pose()
        if curr_pos is None:
            self.get_logger().error("Could not read current robot pose")
            return False
        
        # Create PoseStamped message with robot's current pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"
        
        pose_msg.pose.position.x = curr_pos[0]
        pose_msg.pose.position.y = curr_pos[1]
        pose_msg.pose.position.z = curr_pos[2]
        
        q = quaternion_from_euler(curr_rpy[0], curr_rpy[1], curr_rpy[2])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        # Publish to set initial pose
        self.set_odom_pose_pub.publish(pose_msg)
        
        self.get_logger().info(
            f"Set odometry initial pose to robot pose: "
            f"pos={curr_pos}, rpy(deg)={np.rad2deg(curr_rpy)}"
        )
        
        # Wait for odometry to process the reset and reinitialize
        time.sleep(1.0)
        
        # Verify the pose was set correctly by checking current odometry pose
        odom_pos, odom_rpy = self.get_odometry_pose()
        if odom_pos is not None:
            pos_diff = np.linalg.norm(odom_pos - curr_pos)
            self.get_logger().info(
                f"Verification - Odometry now at: pos={odom_pos}, "
                f"diff from robot: {pos_diff:.3f}m"
            )
        
        return True

    def record_measurement(self, command_delta_pos, command_delta_rpy, initial_odom_pos, initial_odom_quat, label=""):
        """Record the measurement after a move"""
        # Get final poses
        final_robot_pos, final_robot_rpy = self.get_current_robot_pose()
        final_odom_pos, final_odom_quat = self.get_odometry_quaternion()
        
        if final_robot_pos is None or final_odom_pos is None:
            self.get_logger().error("Could not get final poses")
            return None
        
        # Calculate actual deltas from odometry
        actual_odom_delta_pos = final_odom_pos - initial_odom_pos
        
        # Compute rotation delta properly using quaternions
        actual_odom_delta_rpy = self.compute_rotation_delta(initial_odom_quat, final_odom_quat)
        final_odom_rpy = np.array(euler_from_quaternion(final_odom_quat))
        
        # Calculate errors
        pos_error = command_delta_pos - actual_odom_delta_pos
        rpy_error = command_delta_rpy - actual_odom_delta_rpy
        
        # Wrap angle errors to [-pi, pi]
        rpy_error = np.array([
            (rpy_error[i] + np.pi) % (2 * np.pi) - np.pi
            for i in range(3)
        ])
        
        # Calculate absolute pose errors
        abs_pos_error = final_robot_pos - final_odom_pos
        abs_rpy_error = final_robot_rpy - final_odom_rpy
        abs_rpy_error = np.array([
            (abs_rpy_error[i] + np.pi) % (2 * np.pi) - np.pi
            for i in range(3)
        ])
        
        result = {
            'label': label,
            'command_delta_pos': command_delta_pos,
            'command_delta_rpy': command_delta_rpy,
            'actual_odom_delta_pos': actual_odom_delta_pos,
            'actual_odom_delta_rpy': actual_odom_delta_rpy,
            'final_robot_pos': final_robot_pos,
            'final_robot_rpy': final_robot_rpy,
            'final_odom_pos': final_odom_pos,
            'final_odom_rpy': final_odom_rpy,
            'pos_error': pos_error,
            'rpy_error': rpy_error,
            'abs_pos_error': abs_pos_error,
            'abs_rpy_error': abs_rpy_error,
            'pos_error_magnitude': np.linalg.norm(pos_error),
            'rpy_error_magnitude': np.linalg.norm(rpy_error),
            'abs_pos_error_magnitude': np.linalg.norm(abs_pos_error),
            'abs_rpy_error_magnitude': np.linalg.norm(abs_rpy_error),
        }
        
        self.test_results.append(result)
        
        # Log results (condensed format)
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Test: {label}")
        self.get_logger().info(f"Robot   - Pos: [{final_robot_pos[0]:7.4f}, {final_robot_pos[1]:7.4f}, {final_robot_pos[2]:7.4f}] m, "
                              f"RPY: [{np.rad2deg(final_robot_rpy[0]):6.2f}, {np.rad2deg(final_robot_rpy[1]):6.2f}, {np.rad2deg(final_robot_rpy[2]):6.2f}] deg")
        self.get_logger().info(f"Odom    - Pos: [{final_odom_pos[0]:7.4f}, {final_odom_pos[1]:7.4f}, {final_odom_pos[2]:7.4f}] m, "
                              f"RPY: [{np.rad2deg(final_odom_rpy[0]):6.2f}, {np.rad2deg(final_odom_rpy[1]):6.2f}, {np.rad2deg(final_odom_rpy[2]):6.2f}] deg")
        self.get_logger().info(f"Abs Err - Pos: [{abs_pos_error[0]:7.4f}, {abs_pos_error[1]:7.4f}, {abs_pos_error[2]:7.4f}] m (|e|={result['abs_pos_error_magnitude']:.4f}), "
                              f"RPY: [{np.rad2deg(abs_rpy_error[0]):6.2f}, {np.rad2deg(abs_rpy_error[1]):6.2f}, {np.rad2deg(abs_rpy_error[2]):6.2f}] deg (|e|={np.rad2deg(result['abs_rpy_error_magnitude']):.2f})")
        self.get_logger().info(f"Cmd Δ   - Pos: [{command_delta_pos[0]:7.4f}, {command_delta_pos[1]:7.4f}, {command_delta_pos[2]:7.4f}] m, "
                              f"RPY: [{np.rad2deg(command_delta_rpy[0]):6.2f}, {np.rad2deg(command_delta_rpy[1]):6.2f}, {np.rad2deg(command_delta_rpy[2]):6.2f}] deg")
        self.get_logger().info(f"Odom Δ  - Pos: [{actual_odom_delta_pos[0]:7.4f}, {actual_odom_delta_pos[1]:7.4f}, {actual_odom_delta_pos[2]:7.4f}] m, "
                              f"RPY: [{np.rad2deg(actual_odom_delta_rpy[0]):6.2f}, {np.rad2deg(actual_odom_delta_rpy[1]):6.2f}, {np.rad2deg(actual_odom_delta_rpy[2]):6.2f}] deg")
        self.get_logger().info(f"Δ Error - Pos: [{pos_error[0]:7.4f}, {pos_error[1]:7.4f}, {pos_error[2]:7.4f}] m (|e|={result['pos_error_magnitude']:.4f}), "
                              f"RPY: [{np.rad2deg(rpy_error[0]):6.2f}, {np.rad2deg(rpy_error[1]):6.2f}, {np.rad2deg(rpy_error[2]):6.2f}] deg (|e|={np.rad2deg(result['rpy_error_magnitude']):.2f})")
        self.get_logger().info(f"{'='*60}\n")
        
        return result

    def run_test_sequence(self, test_points, stabilize_time=2.0, set_initial_pose=True):
        """Run through a sequence of test points"""
        self.get_logger().info(f"\nStarting test sequence with {len(test_points)} movements...")
        
        # Wait for initial odometry data
        self.get_logger().info("Waiting for odometry data...")
        timeout = 10.0
        start = time.time()
        while self.latest_odom_pose is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.latest_odom_pose is None:
            self.get_logger().error("No odometry data received! Is RGBD_odometry running?")
            return False
        
        self.get_logger().info("Odometry data received.")
        
        # Wait for odometry to stabilize and accumulate some frames
        self.get_logger().info("Waiting for odometry to stabilize...")
        self.wait_for_stability(3.0)
        
        # Set initial pose to align odometry with robot frame
        if set_initial_pose:
            self.get_logger().info("Setting initial odometry pose to match robot...")
            if not self.set_odometry_initial_pose():
                self.get_logger().error("Failed to set initial odometry pose")
                return False
            # Wait for the offset to be applied
            self.wait_for_stability(1.0)
        
        self.get_logger().info("Starting test...")
        
        # Initial stabilization
        self.wait_for_stability(stabilize_time)
        
        for i, test_point in enumerate(test_points):
            delta_pos = test_point['delta_pos']
            delta_rpy = test_point['delta_rpy']
            label = test_point.get('label', f"Move {i+1}")
            
            # Record initial odometry state (use quaternion for proper rotation computation)
            initial_odom_pos, initial_odom_quat = self.get_odometry_quaternion()
            
            # Execute movement
            success = self.move_delta(delta_pos, delta_rpy, label)
            if not success:
                self.get_logger().error(f"Failed to execute move: {label}")
                continue
            
            # Wait for stability
            self.wait_for_stability(stabilize_time)
            
            # Record measurement
            self.record_measurement(delta_pos, delta_rpy, initial_odom_pos, initial_odom_quat, label)
            
            # Spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return True

    def print_summary(self):
        """Print summary of all test results"""
        if not self.test_results:
            self.get_logger().warn("No test results to summarize")
            return
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("TEST SUMMARY")
        self.get_logger().info(f"{'='*60}")
        
        pos_errors = [r['pos_error_magnitude'] for r in self.test_results]
        rpy_errors = [r['rpy_error_magnitude'] for r in self.test_results]
        abs_pos_errors = [r['abs_pos_error_magnitude'] for r in self.test_results]
        abs_rpy_errors = [r['abs_rpy_error_magnitude'] for r in self.test_results]
        
        self.get_logger().info(f"Total tests: {len(self.test_results)}")
        
        self.get_logger().info(f"\nDelta Position errors (m):")
        self.get_logger().info(f"  Mean: {np.mean(pos_errors):.6f}, Std: {np.std(pos_errors):.6f}, Min: {np.min(pos_errors):.6f}, Max: {np.max(pos_errors):.6f}")
        
        self.get_logger().info(f"\nDelta Orientation errors (deg):")
        self.get_logger().info(f"  Mean: {np.rad2deg(np.mean(rpy_errors)):.3f}, Std: {np.rad2deg(np.std(rpy_errors)):.3f}, Min: {np.rad2deg(np.min(rpy_errors)):.3f}, Max: {np.rad2deg(np.max(rpy_errors)):.3f}")
        
        self.get_logger().info(f"\nAbsolute Position errors (m):")
        self.get_logger().info(f"  Mean: {np.mean(abs_pos_errors):.6f}, Std: {np.std(abs_pos_errors):.6f}, Min: {np.min(abs_pos_errors):.6f}, Max: {np.max(abs_pos_errors):.6f}")
        
        self.get_logger().info(f"\nAbsolute Orientation errors (deg):")
        self.get_logger().info(f"  Mean: {np.rad2deg(np.mean(abs_rpy_errors)):.3f}, Std: {np.rad2deg(np.std(abs_rpy_errors)):.3f}, Min: {np.rad2deg(np.min(abs_rpy_errors)):.3f}, Max: {np.rad2deg(np.max(abs_rpy_errors)):.3f}")
        
        self.get_logger().info(f"\nDetailed results:")
        for i, result in enumerate(self.test_results):
            self.get_logger().info(
                f"  {i+1}. {result['label']:20s}: "
                f"Δ_pos={result['pos_error_magnitude']:.4f}m, Δ_rpy={np.rad2deg(result['rpy_error_magnitude']):5.2f}deg | "
                f"Abs_pos={result['abs_pos_error_magnitude']:.4f}m, Abs_rpy={np.rad2deg(result['abs_rpy_error_magnitude']):5.2f}deg"
            )
        
        self.get_logger().info(f"{'='*60}\n")


def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description="Test odometry accuracy by commanding robot movements")
    parser.add_argument("--test", type=str, default="grid", 
                       choices=["grid", "linear", "rotation", "custom"],
                       help="Test pattern to execute")
    parser.add_argument("--step", type=float, default=0.05, 
                       help="Step size for grid/linear tests (m)")
    parser.add_argument("--rot_step", type=float, default=10.0, 
                       help="Rotation step size (degrees)")
    parser.add_argument("--stabilize", type=float, default=2.0, 
                       help="Stabilization time between moves (s)")
    parser.add_argument("--no_set_initial_pose", action="store_true",
                       help="Don't set initial odometry pose to match robot")
    args = parser.parse_args()

    # Create tester node (assumes RGBD_odometry is running externally)
    tester_node = OdometryTester()
    
    tester_node.get_logger().info("Odometry Tester started - expecting /rgbd_odometry/pose from external node")

    # Define test sequences
    test_sequences = {
        "grid": [
            # XY grid pattern
            {'delta_pos': np.array([args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': '+X'},
            {'delta_pos': np.array([0, args.step, 0]), 'delta_rpy': np.zeros(3), 'label': '+Y'},
            {'delta_pos': np.array([-args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': '-X'},
            {'delta_pos': np.array([0, -args.step, 0]), 'delta_rpy': np.zeros(3), 'label': '-Y'},
            {'delta_pos': np.array([0, 0, args.step]), 'delta_rpy': np.zeros(3), 'label': '+Z'},
            {'delta_pos': np.array([0, 0, -args.step]), 'delta_rpy': np.zeros(3), 'label': '-Z'},
        ],
        "linear": [
            # Linear motion in X direction
            {'delta_pos': np.array([args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': 'Linear +X step 1'},
            {'delta_pos': np.array([args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': 'Linear +X step 2'},
            {'delta_pos': np.array([args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': 'Linear +X step 3'},
            {'delta_pos': np.array([-3*args.step, 0, 0]), 'delta_rpy': np.zeros(3), 'label': 'Return to start'},
        ],
        "rotation": [
            # Pure rotation tests
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([np.deg2rad(args.rot_step), 0, 0]), 'label': '+Roll'},
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([-np.deg2rad(args.rot_step), 0, 0]), 'label': '-Roll'},
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([0, np.deg2rad(args.rot_step), 0]), 'label': '+Pitch'},
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([0, -np.deg2rad(args.rot_step), 0]), 'label': '-Pitch'},
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([0, 0, np.deg2rad(args.rot_step)]), 'label': '+Yaw'},
            {'delta_pos': np.zeros(3), 'delta_rpy': np.array([0, 0, -np.deg2rad(args.rot_step)]), 'label': '-Yaw'},
        ],
        "custom": [
            # Combined translation and rotation
            {'delta_pos': np.array([args.step, args.step, 0]), 'delta_rpy': np.array([0, 0, np.deg2rad(15)]), 'label': 'Diagonal XY + Yaw'},
            {'delta_pos': np.array([0, 0, args.step]), 'delta_rpy': np.array([np.deg2rad(10), 0, 0]), 'label': '+Z + Roll'},
            {'delta_pos': np.array([-args.step, -args.step, -args.step]), 'delta_rpy': np.array([0, -np.deg2rad(10), -np.deg2rad(15)]), 'label': 'Return diagonal'},
        ],
    }

    test_points = test_sequences[args.test]
    
    tester_node.get_logger().info(f"Running '{args.test}' test pattern with {len(test_points)} movements")
    
    try:
        success = tester_node.run_test_sequence(
            test_points, 
            stabilize_time=args.stabilize,
            set_initial_pose=not args.no_set_initial_pose
        )
        
        if success:
            tester_node.print_summary()
        else:
            tester_node.get_logger().error("Test sequence failed")
    finally:
        # Cleanup
        tester_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
