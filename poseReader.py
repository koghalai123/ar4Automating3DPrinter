#!/usr/bin/env python3

import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

import tf2_ros
from rclpy.time import Time

def quat_to_euler(x: float, y: float, z: float, w: float):
	roll, pitch, yaw = R.from_quat([x, y, z, w]).as_euler("xyz", degrees=False)
	return roll, pitch, yaw

import sys
sys.path.insert(0, "/home/koghalai/ar4_ws/src/ar4Automating3DPrinter")
from moveit2 import MoveIt2

class PoseReader(Node):
	"""ROS 2 node that prints the gripper pose every second using pymoveit2."""

	def __init__(self, node_name: Optional[str] = None, enable_pose_print: bool = True):
		super().__init__(node_name or "gripper_pose_reader")

		# Robot specifics (AR4 / MoveIt config defaults)
		joint_names = [
			"joint_1",
			"joint_2",
			"joint_3",
			"joint_4",
			"joint_5",
			"joint_6",
		]
		base_link_name = "base_link"
		end_effector_name = "link_6"  # AR4 end-effector link
		group_name = "ar_manipulator"  # AR4 MoveIt group

		# Initialize MoveIt2 helper
		self._cb_group = ReentrantCallbackGroup()
		self.moveit2 = MoveIt2(
			node=self,
			joint_names=joint_names,
			base_link_name=base_link_name,
			end_effector_name=end_effector_name,
			group_name=group_name,
			use_move_group_action=False,
			callback_group=self._cb_group,
		)

		# Keep local references for frames
		self.base_link_name = base_link_name
		self.end_effector_name = end_effector_name

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		# Subscribe directly to joint states
		self._last_joint_msg = None  # list[float] ordered by self.moveit2.joint_names
		self._fk_future = None
		# Hardcoded baseline orientation (home) so printed rpy is (0,0,0) at home
		self.pose = np.array([-1,-1,-1,-1,-1,-1])
		self.quat = np.array([-1, -1, -1, -1])
		self.frame = ""

		self.enable_pose_print = enable_pose_print

		self.create_subscription(
			JointState,
			"joint_states",
			self._on_joint_states,
			10,
		)
		# Always update pose every 0.5s, but only print if enabled
		self._timer = self.create_timer(0.5, self._on_timer)

		self.get_logger().info(
			f"PoseReader started; base='{base_link_name}', eef='{end_effector_name}'"
		)
		self.frameAngles = np.array([0, 0, np.pi/2])  # Rotation from Bad Frame to Good Frame

	def to_good_frame(self, bad_position, bad_euler_angles):
		# Transformation from Bad Frame to Good Frame (BF to GF)

		R_BF_GF_Vec = R.from_euler('XYZ', self.frameAngles, degrees=False)
		R_BF_GF = R_BF_GF_Vec.as_matrix()
		H_BF_GF = np.eye(4)
		H_BF_GF[:3, :3] = R_BF_GF

		# Create rotation matrix from Euler angles
		RBadFrameVec = R.from_euler('XYZ', bad_euler_angles, degrees=False)
		RBadFrame = RBadFrameVec.as_matrix()
		HBadFrame = np.eye(4)
		HBadFrame[:3, :3] = RBadFrame
		HBadFrame[:3, 3] = bad_position

		HGoodFrame = H_BF_GF @ HBadFrame
		good_position = HGoodFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ('XYZ' order)
		good_euler_angles_vec = R.from_matrix(HGoodFrame[:3, :3])
		good_euler_angles = good_euler_angles_vec.as_euler('XYZ', degrees=False)


		return good_position, good_euler_angles

	def to_bad_frame(self, good_position, good_euler_angles):
		"""
		Inverse transformation: from Good Frame back to Bad Frame.
		Args:
			good_position: np.array([x, y, z]) in Good Frame
			good_euler_angles: np.array([roll, pitch, yaw]) in Good Frame ('XYZ' order)
		Returns:
			bad_position, bad_euler_angles in Bad Frame
		"""
		# Inverse rotation from Good Frame to Bad Frame
		R_GF_BF_Vec = R.from_euler('XYZ', -self.frameAngles, degrees=False)
		R_GF_BF = R_GF_BF_Vec.as_matrix()
		H_GF_BF = np.eye(4)
		H_GF_BF[:3, :3] = R_GF_BF

		# Create rotation matrix from Euler angles in Good Frame
		RGoodFrameVec = R.from_euler('XYZ', good_euler_angles, degrees=False)
		RGoodFrame = RGoodFrameVec.as_matrix()
		HGoodFrame = np.eye(4)
		HGoodFrame[:3, :3] = RGoodFrame
		HGoodFrame[:3, 3] = good_position

		# Apply inverse transformation
		HBadFrame = H_GF_BF @ HGoodFrame
		bad_position = HBadFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ('XYZ' order)
		bad_euler_angles_vec = R.from_matrix(HBadFrame[:3, :3])
		bad_euler_angles = bad_euler_angles_vec.as_euler('XYZ', degrees=False)

		return bad_position, bad_euler_angles


	def _on_joint_states(self, msg: JointState):
		# Store joints mapped to the planning group order
		self.jointAngles = msg.position[2:8]
		self.linkNames = msg.name[2:8]

		try:
			self._last_joint_msg = [
				float(msg.position[msg.name.index(j)]) for j in self.moveit2.joint_names
			]
		except ValueError:
			# Missing joints in this message; skip
			return
	
	def get_frame(self, frame = "ee_link"):
		temp = self.tf_buffer.lookup_transform(
			"world",  # target (base)
			frame,                  # source (camera/link)
			Time()
		)
		position = temp.transform.translation
		x, y, z = position.x, position.y, position.z
		
		# Extract rotation (quaternion)
		rotation = temp.transform.rotation
		qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
		
		# Convert quaternion to Euler angles (roll, pitch, yaw)
		roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)

		good_pos, good_euler = self.to_good_frame(np.array([x, y, z]), np.array([roll, pitch, yaw]))

		#bad_pos, bad_euler = self.to_bad_frame(good_pos, good_euler)
		goodPose = np.array([good_pos[0], good_pos[1], good_pos[2], good_euler[0], good_euler[1], good_euler[2]])
		#badPose = np.array([bad_pos[0], bad_pos[1], bad_pos[2], bad_euler[0], bad_euler[1], bad_euler[2]])
		return goodPose

	def get_fk(self):
		# Synchronous FK via MoveIt2.compute_fk()
		js = JointState()
		js.name = list(self.moveit2.joint_names)
		js.position = list(self._last_joint_msg)
		pose_stamped = self.moveit2.compute_fk(
			joint_state=js,
			fk_link_names=[self.end_effector_name],
		)
		if pose_stamped is None:
			self.get_logger().warn("FK failed or returned empty result")
			return
		if isinstance(pose_stamped, list):
			pose_stamped = pose_stamped[0] if pose_stamped else None
			if pose_stamped is None:
				self.get_logger().warn("FK returned empty list")
				return

		p = pose_stamped.pose.position
		q = pose_stamped.pose.orientation
		# Compute relative orientation to hardcoded home quaternion so home is (0,0,0)
		qx, qy, qz, qw = q.x, q.y, q.z, q.w
		
		roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)
		frame = pose_stamped.header.frame_id or self.base_link_name

		self.pose = np.array([p.x, p.y, p.z, roll, pitch, yaw])
		self.quat = np.array([qx, qy, qz, qw])
		self.frame = frame
		#print("Computed fk (sync)")


	def _on_timer(self):
		# Ensure joint states have been received at least once
		if not self._last_joint_msg:
			self.get_logger().warn("Waiting for joint_states...")
			return
		# Always update pose
		self.pose = self.get_frame()
		# Only print if enabled
		if self.enable_pose_print:
			print(
				f"[GripperPose] frame={self.frame} pos=({self.pose[0]:.4f}, {self.pose[1]:.4f}, {self.pose[2]:.4f}) "
				f"quat=({self.quat[0]:.4f}, {self.quat[1]:.4f}, {self.quat[2]:.4f}, {self.quat[3]:.4f}) "
				f"rpy=({self.pose[3]:.4f}, {self.pose[4]:.4f}, {self.pose[5]:.4f})"
			)


def main(argv=None):
	rclpy.init(args=argv)
	node = PoseReader()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)

