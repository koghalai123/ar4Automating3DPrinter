#!/usr/bin/env python3
"""
Script to manually set the odometry initial pose.
This aligns the odometry frame with a specified robot pose.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import argparse
import sys


class OdometryPoseSetter(Node):
    def __init__(self):
        super().__init__('odometry_pose_setter')
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/rgbd_odometry/set_initial_pose',
            10
        )
        
    def set_pose(self, x, y, z, roll, pitch, yaw):
        """Set odometry initial pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"
        
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        
        q = quaternion_from_euler(roll, pitch, yaw)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        # Publish multiple times to ensure it's received
        for _ in range(5):
            self.pose_pub.publish(pose_msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info(
            f"Set odometry pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Set initial odometry pose to align with robot frame"
    )
    parser.add_argument('--x', type=float, default=0.0, help='X position (m)')
    parser.add_argument('--y', type=float, default=0.0, help='Y position (m)')
    parser.add_argument('--z', type=float, default=0.0, help='Z position (m)')
    parser.add_argument('--roll', type=float, default=0.0, help='Roll angle (radians)')
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch angle (radians)')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw angle (radians)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    node = OdometryPoseSetter()
    
    try:
        node.set_pose(args.x, args.y, args.z, args.roll, args.pitch, args.yaw)
        print("Initial pose set successfully!")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
