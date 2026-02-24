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
from simulated3DPrinter import Simulated3DPrinter
import time


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
        
        # Estimated marker frame name prefix
        self.estimatedMarkerPrefix = "estimated_marker_"
        
    def scanLocationForMarkers(self, estimated_pos, estimated_orient=[0,0,0], viewing_distance=0.15, frame_name=None):
        """
        Move the camera to face an estimated marker location.
        
        This creates a temporary TF frame at the estimated location and moves
        the robot to view it, similar to moveToMarker but for positions where
        we expect to find a marker.
        
        Parameters:
            estimated_pos: [x, y, z] estimated marker position in base_link frame
            estimated_orient: [roll, pitch, yaw] estimated marker orientation (radians).
                             If None, assumes marker faces toward robot origin.
            viewing_distance: Distance from marker to position the camera (meters)
            frame_name: Optional custom frame name. If None, uses "estimated_marker_0"
        """
        estimated_pos = np.array(estimated_pos)        
        
        # Create frame name
        if frame_name is None:
            frame_name = f"{self.estimatedMarkerPrefix}0"
        
        # Define offset from marker (position camera at viewing_distance in front, facing the marker)
        # The offset is in the marker frame: negative Y to be in front of marker
        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = np.array([0.0, 0.0, 0.0])
        
        # Transform offset from base_link to the estimated marker frame with retry
        markerBadPos, markerBadEuler = self.to_bad_frame(estimated_pos, estimated_orient)

        for attempt in range(5):
            # Keep broadcasting while we try to look up
            self.broadcast_marker_transform(
                markerBadPos, 
                markerBadEuler, 
                parent_frame="base_link", 
                child_frame=frame_name
            )
            badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame="base_link", target_frame=frame_name)
            if badPos is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if badPos is None:
            self.get_logger().error(f"Failed to get transform to {frame_name}")
            return False
        
        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        
        self.get_logger().info(f'Scanning for markers at estimated position: {estimated_pos}')
        self.get_logger().info(f'Moving to viewing position: {goodPos}')
        
        self.move_to_pose(goodPos, goodEuler)
        return True
    
    def scanMultipleLocations(self, locations, viewing_distance=0.15, pause_duration=2.0):
        """
        Scan multiple estimated marker locations sequentially.
        
        Parameters:
            locations: List of [x, y, z] positions or tuples of (pos, orient)
            viewing_distance: Distance from each marker to position the camera
            pause_duration: Time to pause at each location for marker detection (seconds)
        """
        import time
        
        for i, location in enumerate(locations):
            # Handle both position-only and (position, orientation) formats
            if isinstance(location, tuple) and len(location) == 2:
                pos, orient = location
            else:
                pos = location
                orient = None
            
            frame_name = f"{self.estimatedMarkerPrefix}{i}"
            self.get_logger().info(f"Scanning location {i+1}/{len(locations)}: {pos}")
            
            success = self.scanLocationForMarkers(
                estimated_pos=pos,
                estimated_orient=orient,
                viewing_distance=viewing_distance,
                frame_name=frame_name
            )
            
            if success:
                # Pause to allow marker detection
                time.sleep(pause_duration)
                
                # Check if any markers were detected
                if hasattr(self, 'marker_poses') and self.marker_poses:
                    self.get_logger().info(f"Detected {len(self.marker_poses)} markers at location {i+1}")
                else:
                    self.get_logger().info(f"No markers detected at location {i+1}")

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
    printer = Simulated3DPrinter()
    printer.spawn_complete()

    node = printerAutomation(marker_size=[0.03, 0.05], aruco_dict=['DICT_4X4_50', 'DICT_6X6_50'], calibration_mode=False)
    
    # Use a one-shot timer to run scanLocationForMarkers after the node is fully initialized
    def delayed_scan():
        node.get_logger().info("Starting delayed scan for markers...")
        node.scanLocationForMarkers(
            estimated_pos=[0.5, -0.0, 0.15],      # Where we expect the marker
            estimated_orient=[0,0,0],                # Auto-compute orientation facing robot
            viewing_distance=0.15                 # 15cm from marker
        )
        # Cancel the timer after running once
        scan_timer.cancel()
    
    # Wait 3 seconds for the system to fully initialize (joint_states, TF, etc.)
    scan_timer = node.create_timer(3.0, delayed_scan)
    node.pickupPlate()

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