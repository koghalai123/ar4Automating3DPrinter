import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from showVideoFeed import CameraViewer
from poseReader import PoseReader
import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
import tf2_ros
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from poseReader import PoseReader 
import sys
import time
from geometry_msgs.msg import Point, Quaternion, Pose
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from moveit_msgs.msg import PositionConstraint, BoundingVolume, SolidPrimitive


class constrainedMover(Node):
	def __init__(self, poseReader: PoseReader):
		super().__init__('constrained_mover')
		self.poseReader = poseReader
		self.pose1 = np.array([0.35, 0.05, 0.45, 0, 0, 0])
		self.pose2 = np.array([0.35, -0.05, 0.45, 0, 0, 0])

		self.poseReader.moveit2.allowed_planning_time = 30.0
		#self.poseReader.moveit2.pipeline_id = "ompl"
		#self.poseReader.moveit2.planner_id = "RRTConnectkConfigDefault"
		time.sleep(4.0)  # Wait for MoveIt2 to be ready
		self.moveBetweenPoses()

	def moveBetweenPoses(self):
		for i in range(10):
			pos1 = self.pose1[0:3]
			ori1 = self.pose1[3:6]
			pos2 = self.pose2[0:3]
			ori2 = self.pose2[3:6]
			self.move_to_pose_constrained(pos1, ori1)
			time.sleep(0.5)  # Pause between steps
			self.move_to_pose_constrained(pos2, ori2)
			time.sleep(0.5)  # Pause between steps
		return
	def move_to_pose_constrained(self, pos, euler):
		# Transform to bad frame and get quaternion
		bad_pos, bad_euler = self.poseReader.to_bad_frame(pos, euler)
		q = quaternion_from_euler(bad_euler[0], bad_euler[1], bad_euler[2])
		q_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

		# Set orientation constraint (constant orientation)
		self.poseReader.moveit2.set_path_orientation_constraint(
			quat_xyzw=q,
			frame_id="base_link",  # or the appropriate frame
			target_link=self.poseReader.end_effector_name,
			tolerance=0.15  # Increase tolerance for path constraints
		)

		# Move to pose with constraint
		self.poseReader.moveit2.move_to_pose(position=Point(x=bad_pos[0], y=bad_pos[1], z=bad_pos[2]), quat_xyzw=q_msg)
		self.poseReader.moveit2.wait_until_executed()

		# Clear path constraints after movement
		self.poseReader.moveit2.clear_path_constraints()

	def add_movement_constraint(self, constraint_type, value):
		"""
		Add a path constraint without moving.
		constraint_type: 'orientation' or 'x', 'y', 'z'
		value: for 'orientation', tuple (x,y,z,w) quaternion
			   for 'x','y','z', float value to constrain to
		"""
		if constraint_type == 'orientation':
			# Assume value is (x,y,z,w) quaternion tuple
			self.poseReader.moveit2.set_path_orientation_constraint(
				quat_xyzw=value,
				frame_id="base_link",
				target_link=self.poseReader.end_effector_name,
				tolerance=0.15
			)
		elif constraint_type in ['x', 'y', 'z']:
			# Get current position
			self.poseReader.get_fk()
			current_pos = self.poseReader.pose[:3]  # [x, y, z]
			axis_index = {'x': 0, 'y': 1, 'z': 2}[constraint_type]
			
			# Set constrained position
			constrained_pos = current_pos.copy()
			constrained_pos[axis_index] = value
			
			# Create position constraint with box for anisotropic tolerance
			constraint = PositionConstraint()
			constraint.header.frame_id = "base_link"
			constraint.link_name = self.poseReader.end_effector_name
			constraint.constraint_region = BoundingVolume()
			constraint.constraint_region.primitives.append(SolidPrimitive())
			constraint.constraint_region.primitives[0].type = 1  # BOX
			# Dimensions: half-sizes
			tol_small = 0.001  # tight constraint on the axis
			tol_large = 10.0   # loose on others
			dimensions = [tol_large, tol_large, tol_large]
			dimensions[axis_index] = tol_small
			constraint.constraint_region.primitives[0].dimensions = dimensions
			constraint.constraint_region.primitive_poses.append(Pose())
			constraint.constraint_region.primitive_poses[0].position.x = constrained_pos[0]
			constraint.constraint_region.primitive_poses[0].position.y = constrained_pos[1]
			constraint.constraint_region.primitive_poses[0].position.z = constrained_pos[2]
			constraint.weight = 1.0
			
			# Append to path constraints
			self.poseReader.moveit2._MoveIt2__move_action_goal.request.path_constraints.position_constraints.append(constraint)
		else:
			self.get_logger().error(f"Unknown constraint type: {constraint_type}")
	
	rclpy.init(args=argv)
	
	poseReader = PoseReader(enable_pose_print=False)
	node = constrainedMover(poseReader)
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)