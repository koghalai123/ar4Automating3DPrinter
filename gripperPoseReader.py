#!/usr/bin/env python3

import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def quat_to_euler(x: float, y: float, z: float, w: float):
	roll, pitch, yaw = R.from_quat([x, y, z, w]).as_euler("xyz", degrees=False)
	return roll, pitch, yaw

import sys
sys.path.insert(0, "/home/koghalai/ar4_ws/src/ar4Automating3DPrinter")
from moveit2 import MoveIt2

class GripperPoseReader(Node):
	"""ROS 2 node that prints the gripper pose every second using pymoveit2."""

	def __init__(self):
		super().__init__("gripper_pose_reader")

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
		self.moveit2 = MoveIt2(
			node=self,
			joint_names=joint_names,
			base_link_name=base_link_name,
			end_effector_name=end_effector_name,
			group_name=group_name,
			use_move_group_action=False,
		)

		# Keep local references for frames
		self.base_link_name = base_link_name
		self.end_effector_name = end_effector_name

		# Subscribe directly to joint states
		self._last_joint_state = None  # list[float] ordered by self.moveit2.joint_names
		self._fk_future = None
		# Hardcoded baseline orientation (home) so printed rpy is (0,0,0) at home
		self._home_quat = [-0.5001, 0.5000, -0.4999, 0.5001]  # [x, y, z, w]
		self.create_subscription(
			JointState,
			"joint_states",
			self._on_joint_states,
			10,
		)
        
		# Periodic timer to print pose
		self._timer = self.create_timer(2.0, self._on_timer)

		self.get_logger().info(
			f"GripperPoseReader started; base='{base_link_name}', eef='{end_effector_name}'"
		)

	def _on_joint_states(self, msg: JointState):
		# Store joints mapped to the planning group order
		try:
			self._last_joint_state = [
				float(msg.position[msg.name.index(j)]) for j in self.moveit2.joint_names
			]
		except ValueError:
			# Missing joints in this message; skip
			return

	def _on_timer(self):
		# Ensure joint states have been received at least once
		if not self._last_joint_state:
			self.get_logger().warn("Waiting for joint_states...")
			return

		# If a previous FK future completed, consume and print it
		if self._fk_future is not None and self._fk_future.done():
			pose_stamped = self.moveit2.get_compute_fk_result(
				self._fk_future, fk_link_names=[self.end_effector_name]
			)
			self._fk_future = None
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
			# q_rel = q_current * conj(q_home)
			qx0, qy0, qz0, qw0 = self._home_quat
			# conjugate of unit quaternion
			cx, cy, cz, cw = -qx0, -qy0, -qz0, qw0
			# quaternion multiply (x,y,z,w) * (cx,cy,cz,cw)
			rx = qw * cx + qx * cw + qy * cz - qz * cy
			ry = qw * cy - qx * cz + qy * cw + qz * cx
			rz = qw * cz + qx * cy - qy * cx + qz * cw
			rw = qw * cw - qx * cx - qy * cy - qz * cz
			# normalize to guard against drift
			norm = math.sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
			if norm > 0:
				rx, ry, rz, rw = rx/norm, ry/norm, rz/norm, rw/norm
			roll, pitch, yaw = quat_to_euler(rx, ry, rz, rw)
			frame = pose_stamped.header.frame_id or self.base_link_name
			print(
				f"[GripperPose] frame={frame} pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) "
				f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}) "
				f"rpy=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
			)

		# If no FK request is in flight, start a new one using JointState form to avoid private attr
		if self._fk_future is None:
			js = JointState()
			js.name = list(self.moveit2.joint_names)
			js.position = list(self._last_joint_state)
			self._fk_future = self.moveit2.compute_fk_async(
				joint_state=js,
				fk_link_names=[self.end_effector_name],
			)


def main(argv=None):
	rclpy.init(args=argv)
	node = GripperPoseReader()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)

