#!/usr/bin/env python3
"""
Example of using MoveIt 2 Servo to perform a circular motion.
- ros2 run pymoveit2 ex_servo.py
"""


from math import cos, sin

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import TwistStamped
from rclpy.parameter import Parameter


def main():
    rclpy.init()

    # Create node for this example
    node = Node("ex_servo")
    # Align with Gazebo/MoveIt sim time if enabled
    node.set_parameters([Parameter(name="use_sim_time", value=True)])

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # AR4 publishes TwistStamped directly to Servo's input topic
    frame_id = "base_link"  # AR4 robot base frame
    twist_pub = node.create_publisher(
        TwistStamped,
        "/delta_twist_cmds",
        qos_profile=QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_ALL,
        ),
        callback_group=callback_group,
    )

    def servo_circular_motion():
        """Move in a circular motion using Servo"""

        now_sec = node.get_clock().now().nanoseconds * 1e-9
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        # Use modest speeds (m/s) and match Servo expectations
        speed = 0.05
        msg.twist.linear.x = speed * sin(now_sec)
        msg.twist.linear.y = speed * cos(now_sec)
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        twist_pub.publish(msg)
        # Debug: report subscriber count
        sub_count = twist_pub.get_subscription_count()
        node.get_logger().info(f"Publishing Twist to /delta_twist_cmds; subscribers: {sub_count}")

    # Create timer for moving in a circular motion (match Servo publish_period ~0.04)
    node.create_timer(0.04, servo_circular_motion)

    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()
    exit(0)


if __name__ == "__main__":
    main()
