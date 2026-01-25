from ArucoDetector import ArucoDetectionViewer
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
import tempfile
import os




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
	


def delete_aruco_marker(n: Node, name: str):
    
    delete_cmd = [
        'gz', 'service', '-s', '/world/default/remove',
        '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
        '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
    ]
    try:
        result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
        msg = result.stdout.strip() or 'Existing marker deleted successfully'
        n.get_logger().info(f'Aruco delete: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        n.get_logger().warn(f'Aruco delete failed (possibly no existing marker): {err}')
    except Exception as e:
        n.get_logger().warn(f'Aruco delete exception: {e}')

def spawn_aruco_marker(n: Node, texture_path: str, marker_size: float, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    # Determine the name from the texture file
    
    name = os.path.basename(texture_path).split('.')[0]

    delete_aruco_marker(n, name)
    # Modify the SDF file with the new texture and size
    sdf_path = '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf'
    with open(sdf_path, 'r') as f:
        sdf_content = f.read()
    # Replace the albedo_map line
    old_texture_line = '              <albedo_map>materials/textures/marker.png</albedo_map>'
    new_texture_line = f'              <albedo_map>{texture_path}</albedo_map>'
    modified_sdf = sdf_content.replace(old_texture_line, new_texture_line)
    # Replace the size line
    old_size_line = f'            <size>0.0001 {0.05} {0.05}</size>'
    new_size_line = f'            <size>0.0001 {marker_size} {marker_size}</size>'
    modified_sdf = modified_sdf.replace(old_size_line, new_size_line)
    # Write to a temporary file in the same directory as the model
    model_dir = '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/'
    with tempfile.NamedTemporaryFile(dir=model_dir, mode='w', suffix='.sdf', delete=False) as temp_file:
        temp_file.write(modified_sdf)
        temp_sdf_path = temp_file.name

    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-file', temp_sdf_path,
        '-name', name,
        '-x', str(x), '-y', str(y), '-z', str(z),
        '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        msg = result.stdout.strip() or 'Spawn command executed successfully'
        n.get_logger().info(f'Aruco spawn: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        n.get_logger().error(f'Aruco spawn failed: {err}')
    except Exception as e:
        n.get_logger().error(f'Aruco spawn exception: {e}')
    finally:
        # Clean up the temp file
        os.unlink(temp_sdf_path)


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