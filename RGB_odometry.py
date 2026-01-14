#!/usr/bin/env python3

import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nav_msgs.msg import Path
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from pymoveit2 import MoveIt2
import matplotlib.pyplot as plt
from collections import defaultdict


class VisualOdometryTest(Node):
    def __init__(self):
        super().__init__('visual_odometry_test')
        
        # MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        self.moveit2.max_velocity = 0.5  # Slower for better VO tracking
        self.moveit2.max_acceleration = 1.0
        
        # Subscribe to visual odometry pose
        self.vo_pose_sub = self.create_subscription(
            PoseStamped,
            '/vo/pose',
            self.vo_pose_callback,
            10
        )
        
        # Latest VO pose
        self.latest_vo_pose = None
        self.vo_pose_received = False
        
        # Data storage for comparison
        self.commanded_positions = []
        self.commanded_orientations = []
        self.vo_positions = []
        self.vo_orientations = []
        self.timestamps = []
        
        self.get_logger().info('Visual Odometry Test Node Initialized')
        self.get_logger().info('Waiting for visual odometry data...')
        
    def vo_pose_callback(self, msg):
        """Callback for visual odometry pose"""
        self.latest_vo_pose = msg
        self.vo_pose_received = True
        
    def get_current_robot_pose(self):
        """Get current end-effector pose from MoveIt2 FK"""
        fk = self.moveit2.compute_fk()
        if fk is None:
            self.get_logger().warn("compute_fk returned None")
            return None, None
        
        pose_msg = fk[0] if isinstance(fk, list) else fk
        
        position = np.array([
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
        rpy = np.array(euler_from_quaternion(quat), dtype=float)  # radians
        
        return position, rpy
        
    def move_to_pose(self, pos_xyz, rpy_rad):
        """Move robot to specified pose"""
        q = quaternion_from_euler(rpy_rad[0], rpy_rad[1], rpy_rad[2])
        q_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        self.moveit2.move_to_pose(
            position=Point(x=pos_xyz[0], y=pos_xyz[1], z=pos_xyz[2]),
            quat_xyzw=q_msg
        )
        self.moveit2.wait_until_executed()
        
    def record_measurement(self):
        """Record both commanded and VO measurements"""
        # Get commanded pose
        cmd_pos, cmd_rpy = self.get_current_robot_pose()
        
        if cmd_pos is None:
            self.get_logger().warn("Failed to get robot pose")
            return False
            
        # Wait a bit for VO to stabilize
        time.sleep(0.5)
        
        # Get VO pose
        if not self.vo_pose_received or self.latest_vo_pose is None:
            self.get_logger().warn("No VO data received yet")
            return False
            
        vo_pos = np.array([
            self.latest_vo_pose.pose.position.x,
            self.latest_vo_pose.pose.position.y,
            self.latest_vo_pose.pose.position.z,
        ])
        
        vo_quat = [
            self.latest_vo_pose.pose.orientation.x,
            self.latest_vo_pose.pose.orientation.y,
            self.latest_vo_pose.pose.orientation.z,
            self.latest_vo_pose.pose.orientation.w,
        ]
        vo_rpy = np.array(euler_from_quaternion(vo_quat), dtype=float)
        
        # Store measurements
        self.commanded_positions.append(cmd_pos.copy())
        self.commanded_orientations.append(cmd_rpy.copy())
        self.vo_positions.append(vo_pos.copy())
        self.vo_orientations.append(vo_rpy.copy())
        self.timestamps.append(time.time())
        
        return True
        
    def run_test_sequence(self):
        """Execute a series of movements and record data"""
        self.get_logger().info("Starting test sequence...")
        
        # Wait for VO to initialize
        self.get_logger().info("Waiting for visual odometry to initialize...")
        timeout = 10.0
        start = time.time()
        while not self.vo_pose_received and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        if not self.vo_pose_received:
            self.get_logger().error("Visual odometry not available. Make sure visual_odometry_ros.py is running!")
            return False
            
        self.get_logger().info("Visual odometry detected!")
        
        # Get initial pose
        init_pos, init_rpy = self.get_current_robot_pose()
        if init_pos is None:
            self.get_logger().error("Failed to get initial robot pose")
            return False
            
        self.get_logger().info(f"Initial position: {init_pos}")
        self.get_logger().info(f"Initial orientation (RPY): {np.rad2deg(init_rpy)}")
        
        # Reset VO by waiting for stable initial measurement
        time.sleep(2.0)
        
        # Define test movements (relative to initial pose)
        # Format: (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw)
        movements = [
            # Linear movements
            (0.05, 0.0, 0.0, 0.0, 0.0, 0.0),    # Move +X
            (0.0, 0.05, 0.0, 0.0, 0.0, 0.0),    # Move +Y
            (0.0, 0.0, 0.05, 0.0, 0.0, 0.0),    # Move +Z
            (0.0, 0.0, -0.05, 0.0, 0.0, 0.0),   # Move -Z
            (0.0, -0.05, 0.0, 0.0, 0.0, 0.0),   # Move -Y
            (-0.05, 0.0, 0.0, 0.0, 0.0, 0.0),   # Move -X (back to start)
            
            # Rotational movements
            (0.0, 0.0, 0.0, 0.1, 0.0, 0.0),     # Roll +5.7°
            (0.0, 0.0, 0.0, 0.0, 0.1, 0.0),     # Pitch +5.7°
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.1),     # Yaw +5.7°
            (0.0, 0.0, 0.0, 0.0, 0.0, -0.1),    # Yaw -5.7°
            (0.0, 0.0, 0.0, 0.0, -0.1, 0.0),    # Pitch -5.7°
            (0.0, 0.0, 0.0, -0.1, 0.0, 0.0),    # Roll -5.7° (back to start)
        ]
        
        # Record initial state
        self.get_logger().info("Recording initial state...")
        if not self.record_measurement():
            self.get_logger().error("Failed to record initial measurement")
            return False
            
        # Execute movements
        current_pos = init_pos.copy()
        current_rpy = init_rpy.copy()
        
        for i, (dx, dy, dz, dr, dp, dyaw) in enumerate(movements):
            self.get_logger().info(f"\n=== Movement {i+1}/{len(movements)} ===")
            self.get_logger().info(f"Delta: pos=[{dx:.3f}, {dy:.3f}, {dz:.3f}], "
                                 f"rpy=[{np.rad2deg(dr):.1f}°, {np.rad2deg(dp):.1f}°, {np.rad2deg(dyaw):.1f}°]")
            
            # Update target pose
            current_pos += np.array([dx, dy, dz])
            current_rpy += np.array([dr, dp, dyaw])
            
            # Execute movement
            self.get_logger().info(f"Moving to: pos={current_pos}, rpy={np.rad2deg(current_rpy)}")
            self.move_to_pose(current_pos, current_rpy)
            
            # Wait for movement to settle
            time.sleep(1.0)
            
            # Record measurement
            if not self.record_measurement():
                self.get_logger().warn(f"Failed to record measurement for movement {i+1}")
                continue
                
            self.get_logger().info(f"Movement {i+1} complete")
            
        self.get_logger().info("\n=== Test sequence complete! ===")
        return True
        
    def compute_deltas(self):
        """Compute deltas between consecutive measurements"""
        if len(self.commanded_positions) < 2:
            return None
            
        cmd_pos_deltas = []
        cmd_ori_deltas = []
        vo_pos_deltas = []
        vo_ori_deltas = []
        
        for i in range(1, len(self.commanded_positions)):
            # Commanded deltas
            cmd_pos_deltas.append(self.commanded_positions[i] - self.commanded_positions[i-1])
            cmd_ori_deltas.append(self.commanded_orientations[i] - self.commanded_orientations[i-1])
            
            # VO deltas
            vo_pos_deltas.append(self.vo_positions[i] - self.vo_positions[i-1])
            vo_ori_deltas.append(self.vo_orientations[i] - self.vo_orientations[i-1])
            
        return {
            'cmd_pos': np.array(cmd_pos_deltas),
            'cmd_ori': np.array(cmd_ori_deltas),
            'vo_pos': np.array(vo_pos_deltas),
            'vo_ori': np.array(vo_ori_deltas),
        }
        
    def analyze_results(self):
        """Analyze and print comparison results"""
        if len(self.commanded_positions) < 2:
            self.get_logger().error("Not enough data for analysis")
            return
            
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("ANALYSIS RESULTS")
        self.get_logger().info("="*60)
        
        # Convert to numpy arrays
        cmd_pos = np.array(self.commanded_positions)
        cmd_ori = np.array(self.commanded_orientations)
        vo_pos = np.array(self.vo_positions)
        vo_ori = np.array(self.vo_orientations)
        
        # Compute absolute errors at each measurement
        pos_errors = np.linalg.norm(cmd_pos - vo_pos, axis=1)
        ori_errors = np.linalg.norm(cmd_ori - vo_ori, axis=1)
        
        self.get_logger().info(f"\nNumber of measurements: {len(self.commanded_positions)}")
        self.get_logger().info(f"\nPosition Error Statistics:")
        self.get_logger().info(f"  Mean: {np.mean(pos_errors):.4f} m")
        self.get_logger().info(f"  Std:  {np.std(pos_errors):.4f} m")
        self.get_logger().info(f"  Max:  {np.max(pos_errors):.4f} m")
        self.get_logger().info(f"  Min:  {np.min(pos_errors):.4f} m")
        
        self.get_logger().info(f"\nOrientation Error Statistics:")
        self.get_logger().info(f"  Mean: {np.rad2deg(np.mean(ori_errors)):.2f}°")
        self.get_logger().info(f"  Std:  {np.rad2deg(np.std(ori_errors)):.2f}°")
        self.get_logger().info(f"  Max:  {np.rad2deg(np.max(ori_errors)):.2f}°")
        self.get_logger().info(f"  Min:  {np.rad2deg(np.min(ori_errors)):.2f}°")
        
        # Analyze deltas
        deltas = self.compute_deltas()
        if deltas:
            cmd_pos_mag = np.linalg.norm(deltas['cmd_pos'], axis=1)
            vo_pos_mag = np.linalg.norm(deltas['vo_pos'], axis=1)
            delta_pos_error = np.abs(cmd_pos_mag - vo_pos_mag)
            
            self.get_logger().info(f"\nDelta Movement Accuracy:")
            self.get_logger().info(f"  Position delta error mean: {np.mean(delta_pos_error):.4f} m")
            self.get_logger().info(f"  Position delta error std:  {np.std(delta_pos_error):.4f} m")
            
        # Print final comparison
        self.get_logger().info(f"\nFinal Poses:")
        self.get_logger().info(f"  Commanded Position: {cmd_pos[-1]}")
        self.get_logger().info(f"  VO Position:        {vo_pos[-1]}")
        self.get_logger().info(f"  Difference:         {cmd_pos[-1] - vo_pos[-1]}")
        self.get_logger().info(f"\n  Commanded Orientation (deg): {np.rad2deg(cmd_ori[-1])}")
        self.get_logger().info(f"  VO Orientation (deg):        {np.rad2deg(vo_ori[-1])}")
        self.get_logger().info(f"  Difference (deg):            {np.rad2deg(cmd_ori[-1] - vo_ori[-1])}")
        
    def plot_results(self):
        """Create visualization plots"""
        if len(self.commanded_positions) < 2:
            self.get_logger().error("Not enough data for plotting")
            return
            
        cmd_pos = np.array(self.commanded_positions)
        vo_pos = np.array(self.vo_positions)
        cmd_ori = np.array(self.commanded_orientations)
        vo_ori = np.array(self.vo_orientations)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(cmd_pos[:, 0], cmd_pos[:, 1], cmd_pos[:, 2], 'b-o', label='Commanded', linewidth=2)
        ax1.plot(vo_pos[:, 0], vo_pos[:, 1], vo_pos[:, 2], 'r--s', label='Visual Odometry', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Position components over time
        ax2 = fig.add_subplot(2, 3, 2)
        steps = range(len(cmd_pos))
        ax2.plot(steps, cmd_pos[:, 0], 'b-', label='Cmd X')
        ax2.plot(steps, vo_pos[:, 0], 'r--', label='VO X')
        ax2.plot(steps, cmd_pos[:, 1], 'g-', label='Cmd Y')
        ax2.plot(steps, vo_pos[:, 1], 'm--', label='VO Y')
        ax2.plot(steps, cmd_pos[:, 2], 'c-', label='Cmd Z')
        ax2.plot(steps, vo_pos[:, 2], 'y--', label='VO Z')
        ax2.set_xlabel('Measurement #')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position Components')
        ax2.legend()
        ax2.grid(True)
        
        # Position error over time
        ax3 = fig.add_subplot(2, 3, 3)
        pos_errors = np.linalg.norm(cmd_pos - vo_pos, axis=1)
        ax3.plot(steps, pos_errors, 'r-o', linewidth=2)
        ax3.set_xlabel('Measurement #')
        ax3.set_ylabel('Position Error (m)')
        ax3.set_title('Position Error Over Time')
        ax3.grid(True)
        
        # Orientation components over time
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(steps, np.rad2deg(cmd_ori[:, 0]), 'b-', label='Cmd Roll')
        ax4.plot(steps, np.rad2deg(vo_ori[:, 0]), 'r--', label='VO Roll')
        ax4.plot(steps, np.rad2deg(cmd_ori[:, 1]), 'g-', label='Cmd Pitch')
        ax4.plot(steps, np.rad2deg(vo_ori[:, 1]), 'm--', label='VO Pitch')
        ax4.plot(steps, np.rad2deg(cmd_ori[:, 2]), 'c-', label='Cmd Yaw')
        ax4.plot(steps, np.rad2deg(vo_ori[:, 2]), 'y--', label='VO Yaw')
        ax4.set_xlabel('Measurement #')
        ax4.set_ylabel('Orientation (deg)')
        ax4.set_title('Orientation Components')
        ax4.legend()
        ax4.grid(True)
        
        # Orientation error over time
        ax5 = fig.add_subplot(2, 3, 5)
        ori_errors = np.rad2deg(np.linalg.norm(cmd_ori - vo_ori, axis=1))
        ax5.plot(steps, ori_errors, 'b-o', linewidth=2)
        ax5.set_xlabel('Measurement #')
        ax5.set_ylabel('Orientation Error (deg)')
        ax5.set_title('Orientation Error Over Time')
        ax5.grid(True)
        
        # XY trajectory (top-down view)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(cmd_pos[:, 0], cmd_pos[:, 1], 'b-o', label='Commanded', linewidth=2)
        ax6.plot(vo_pos[:, 0], vo_pos[:, 1], 'r--s', label='Visual Odometry', linewidth=2)
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_title('XY Trajectory (Top View)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')
        
        plt.tight_layout()
        plt.savefig('/home/koghalai/ar4_ws/vo_test_results.png', dpi=150)
        self.get_logger().info("\nPlot saved to: /home/koghalai/ar4_ws/vo_test_results.png")
        plt.show()


def main():
    rclpy.init()
    
    node = VisualOdometryTest()
    
    try:
        # Run the test sequence
        success = node.run_test_sequence()
        
        if success:
            # Analyze results
            node.analyze_results()
            
            # Plot results
            node.plot_results()
        else:
            node.get_logger().error("Test sequence failed")
            
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
