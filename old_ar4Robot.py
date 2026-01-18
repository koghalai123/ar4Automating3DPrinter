#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from geometry_msgs.msg import Point, Quaternion, Pose
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
import time

def create_transformation_matrix(translation, euler_angles):
    """Create a 4x4 homogeneous transformation matrix."""
    transform = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
    transform[:3, 3] = translation
    return transform


class AR4_ROBOT(Node):
    def __init__(self, use_joint_positions):
        rclpy.init()
        super().__init__("ar4_robot_commander")
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        self.moveit2.max_velocity = 1.5
        self.moveit2.max_acceleration = 3
        
        self.use_joint_positions = use_joint_positions
        
        if self.use_joint_positions:
            self.get_logger().info("Joint commander initialized. Waiting for connection to Gazebo/Physical Robot.")
        else:
            self.get_logger().info("Pose commander initialized. Waiting for connection to Gazebo/Physical Robot.")

        # Reference transformation for the end effector
        self.reference_translation = [0.0, 0.0, 0.0]  # Example translation (x, y, z)
        self.reference_rotation = [0.0, 0.0, np.pi * 1 / 2]  # Example rotation (roll, pitch, yaw)
        self.transformation_matrix = create_transformation_matrix(self.reference_translation, self.reference_rotation)
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        self.angle_offsets = {
            "roll": 0,  # Example offset in radians
            "pitch": 1.571,
            "yaw": 1.571
        }
        
        self.pos_offsets = {
            "x": 0.32783003,  # Example offset in radians
            "y": -0.00699888,
            "z": 0.47477099
        }

    def resetErrors(self):
        """Reset MoveIt2 state and ensure joint states are available"""
        print("Resetting MoveIt2 state...")
        
        self.moveit2.reset_new_joint_state_checker()
        self.moveit2.force_reset_executing_state()
        
        '''# Wait for joint states to be available
        attempts = 0
        while self.moveit2.joint_state is None and attempts < 5:
            print("Waiting for joint states...")
            time.sleep(0.1)
            attempts += 1
        
        if self.moveit2.joint_state is None:
            print("Failed to get joint states after reset")
            return False
            
        print("MoveIt2 state reset successful")'''
        return True
    def toMyPreferredFrame(self, position, eulerAngles, reference_frame="base_link"):
        # Apply transformation to the pose. Euler Angles should be in the order: roll, pith, yaw
        # My preferred frame has the robot's forward direction as x, leftwards direction as y, and upwards direction as z. 
        #At the home position, the end effector has 0 rotations for all euler angles
        position = np.array([position[0], position[1], position[2], 1.0])
        global_position = np.dot(self.transformation_matrix, position)[:3]
        transformed_orientation = euler_from_matrix(
            np.dot(self.transformation_matrix[:3, :3], euler_matrix(eulerAngles[0], eulerAngles[1], eulerAngles[2], axes='sxyz')[:3, :3]),
            axes='sxyz'
        )
        global_orientation = np.array([
            transformed_orientation[1] + self.angle_offsets["roll"],
            transformed_orientation[0] + self.angle_offsets["pitch"],
            transformed_orientation[2] + self.angle_offsets["yaw"]
        ])
        
        homeRefFramePos = global_position- np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])
        if reference_frame=="base_link":
            return global_position, global_orientation
        else:
            return homeRefFramePos, global_orientation
    
    def get_ik(self, position, eulerAngles, frame="base_link"):
        
        
        transformed_position, transformed_orientation = self.fromMyPreferredFrame(position, eulerAngles, old_reference_frame=frame, new_reference_frame="base_link")
        
        quat = quaternion_from_euler(transformed_orientation[0], transformed_orientation[1], transformed_orientation[2])
        
        target_pose = Pose()
        target_pose.position.x = transformed_position[0]
        target_pose.position.y = transformed_position[1]
        target_pose.position.z = transformed_position[2]
        
        
        
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]

        # Compute inverse kinematics
        jointPositions = self.moveit2.compute_ik(
            position=target_pose.position,
            quat_xyzw=target_pose.orientation,
        )

        # Check if IK was successful
        if jointPositions is not None:
            print("Joint angles found")
            #for name, position in zip(jointPositions.name, jointPositions.position):
            #    print(f"{name}: {position}")
            return np.array(jointPositions.position[:6])
        else:
            print("Failed to compute IK.")
            return None
        
    
    def get_current_pose(self, reference_frame="base_link"):
        """Get the current end effector pose."""
        self.moveit2.reset_new_joint_state_checker()
        fk_result = self.moveit2.compute_fk()
        
        if fk_result is None:
            self.get_logger().warn("Could not compute current pose")
            return None
            
        if isinstance(fk_result, list):
            current_pose = fk_result[0]  # Take first result if multiple
        else:
            current_pose = fk_result

        # Convert quaternion to Euler angles
        quat = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        eulerAngles = np.array([roll, pitch, yaw])
        
        position = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
        position, orientation = self.toMyPreferredFrame(position, eulerAngles, reference_frame)
        
        return position, orientation
        '''self.transformedOrientation = transformed_orientation
        self.globalOrientation = global_orientation
        self.homeRefFramePos = homeRefFramePos
        self.homeRefFrameOrientation = global_orientation'''
        
    def fromMyPreferredFrame(self, position, eulerAngles, old_reference_frame="base_link", new_reference_frame="base_link"):
        if new_reference_frame == old_reference_frame:
            position = np.array([position[0], position[1], position[2], 1.0])
        elif new_reference_frame == "end_effector_link" and old_reference_frame == "base_link":
            position = np.array([position[0], position[1], position[2], 1.0]) - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"],0.0])        
        elif new_reference_frame == "base_link" and old_reference_frame == "end_effector_link":
            position = np.array([position[0], position[1], position[2], 1.0]) + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"],0.0])        

        transformed_position = (np.dot(self.inverse_transformation_matrix, position)[:3])
        
        roll = eulerAngles[1] - self.angle_offsets["pitch"]
        pitch = eulerAngles[0] - self.angle_offsets["roll"]
        yaw = eulerAngles[2] - self.angle_offsets["yaw"]
        
        transformed_orientation_matrix = np.dot(self.inverse_transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3])
        transformed_orientation = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')

        return transformed_position, transformed_orientation

    def move_pose(self, position, eulerAngles, reference_frame="base_link"):
        """Move to a specific pose."""

        transformed_position, transformed_orientation = self.fromMyPreferredFrame(position, eulerAngles, reference_frame)

        # Convert transformed orientation to quaternion
        quat = quaternion_from_euler(transformed_orientation[0], transformed_orientation[1], transformed_orientation[2])
        orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        #self.moveit2.force_reset_executing_state()
        self.moveit2.move_to_pose(position=Point(x=transformed_position[0], y=transformed_position[1], z=transformed_position[2]), quat_xyzw=orientation)
        self.moveit2.wait_until_executed()

    def move_to_joint_positions(self, joint_positions):
        """Move to specific joint positions."""
        self.moveit2.move_to_configuration(joint_positions)
        self.moveit2.wait_until_executed()