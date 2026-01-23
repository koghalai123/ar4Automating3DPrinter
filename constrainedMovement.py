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


class constrainedMover(Node):
	def __init__(self, poseReader: PoseReader):
		super().__init__('constrained_mover')
		self.poseReader = poseReader
		self.pose1 = np.array([0.35, 0.05, 0.45, 0, 0, 0])
		self.pose2 = np.array([0.35, -0.05, 0.45, 0, 0, 0])

		self.poseReader.moveit2.allowed_planning_time = 10.0
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
			tolerance=0.1  # Increase tolerance for path constraints
		)

		# Move to pose with constraint
		self.poseReader.moveit2.move_to_pose(position=Point(x=bad_pos[0], y=bad_pos[1], z=bad_pos[2]), quat_xyzw=q_msg)
		self.poseReader.moveit2.wait_until_executed()

		# Clear path constraints after movement
		self.poseReader.moveit2.clear_path_constraints()


















def main(argv=None):
	
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