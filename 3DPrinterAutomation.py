from ArucoDetector import ArucoDetectionViewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from showVideoFeed import CameraViewer
from poseReader import PoseReader
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
import tf2_ros
from rclpy.time import Time




class printerAutomation(ArucoDetectionViewer):
    def __init__(self, marker_size=[0.01, 0.05], aruco_dict=['DICT_4X4_50', 'DICT_6X6_50'], calibration_mode=False):
        super().__init__()
        # Override parameters after super().__init__()
        self.marker_size = marker_size
        self.aruco_dict = [self._get_aruco_dict(d) for d in aruco_dict]
        self.get_logger().info(f"printerAutomation initialized with marker_size={marker_size}, aruco_dict={aruco_dict}, calibration_mode={calibration_mode}")
        self.declare_parameter('marker_size', marker_size)
        self.declare_parameter('aruco_dict', aruco_dict)
        self.declare_parameter('calibration_mode', calibration_mode)

        self.timer = self.create_timer(10.0, self.pickupPlate)
    def pickupPlate(self, markerID=0):
        self.moveToMarker(markerID)
        self.liftPlate(markerID)
    

    def moveToMarker(self, markerID=0):
        offsetPos = np.array([0.0, 0.0, 0.15])  #position offset from marker
        offsetOri = np.array([0.0, 0.0, 0.0])  # No rotation offset
        if hasattr(self, 'marker_poses') and self.marker_poses is not None:
            for entry in self.marker_poses:
                if entry['id'] == markerID:
                    marker_pos = entry['positionInWorld']
                    temp = entry['orientInWorld']
                    marker_ori = np.array([temp['roll'], temp['pitch'], temp['yaw']])
                    tf2Name = entry['tf2Name']

                    badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame = "base_link", target_frame = tf2Name)
                    goodPos, goodEuler = self.to_good_frame(badPos,badEuler)


                    self.get_logger().info(f'Moving to marker ID {markerID} at pose: {marker_pos}, {marker_ori}')
                    self.move_to_pose(goodPos, goodEuler)
                    # Implement MoveIt2 movement logic here
                    return
    def liftPlate(self, markerID=0):
        offsetPos = np.array([0.0, 0.1, 0.15])  #position offset from marker
        offsetOri = np.array([0.0, 0.0, 0.0])  # No rotation offset
        if hasattr(self, 'marker_poses') and self.marker_poses is not None:
            for entry in self.marker_poses:
                if entry['id'] == markerID:
                    marker_pos = entry['positionInWorld']
                    temp = entry['orientInWorld']
                    marker_ori = np.array([temp['roll'], temp['pitch'], temp['yaw']])
                    tf2Name = entry['tf2Name']

                    badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame = "base_link", target_frame = tf2Name)
                    goodPos, goodEuler = self.to_good_frame(badPos,badEuler)


                    self.get_logger().info(f'Moving to marker ID {markerID} at pose: {marker_pos}, {marker_ori}')
                    self.move_to_pose(goodPos, goodEuler)
                    # Implement MoveIt2 movement logic here
                    return


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
	



def main():
    # Preprogrammed values
    
    
    rclpy.init()
    node = printerAutomation(marker_size=[0.03, 0.05], aruco_dict=['DICT_4X4_50', 'DICT_6X6_50'], calibration_mode=False)
    
    texture_path = 'materials/textures/marker6x6_0.png'
    marker_size = 0.05
    x, y, z = 0.05, -0.5, 0.45
    roll, pitch, yaw = 0.0, 0.0, 1.57
    spawn_aruco_marker(node, texture_path, marker_size, x, y, z, roll, pitch, yaw)
    
    texture_path = 'materials/textures/marker4x4_0.png'
    marker_size = 0.03
    x, y, z = 0.05, -0.55, 0.45
    roll, pitch, yaw = 0.0, 0.0, 1.57
    spawn_aruco_marker(node, texture_path, marker_size, x, y, z, roll, pitch, yaw)
    
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()