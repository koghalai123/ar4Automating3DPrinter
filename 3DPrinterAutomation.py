from ArucoDetector import ArucoDetectionViewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from poseReader import PoseReader
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
import tf2_ros
from rclpy.time import Time
from simulated3DPrinter import Simulated3DPrinter
import time
import threading


class printerAutomation(ArucoDetectionViewer):
    def __init__(self, calibration_mode=False):
        super().__init__()
        self.get_logger().info(f"printerAutomation initialized, calibration_mode={calibration_mode}")

        # Estimated marker frame name prefix
        self.estimatedMarkerPrefix = "estimated_marker_"
        
        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

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
        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = np.array([0.0, 0.0, 0.0])
        
        # Transform offset from base_link to the estimated marker frame with retry
        markerBadPos, markerBadEuler = self.to_bad_frame(estimated_pos, estimated_orient)

        # Ensure the TF broadcaster exists before the retry loop
        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

        badPos = None
        badEuler = None
        for attempt in range(20):
            # Broadcast the estimated marker frame
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = frame_name
            t.transform.translation.x = float(markerBadPos[0])
            t.transform.translation.y = float(markerBadPos[1])
            t.transform.translation.z = float(markerBadPos[2])
            q = R.from_euler("XYZ", markerBadEuler, degrees=False).as_quat()
            t.transform.rotation.x = float(q[0])
            t.transform.rotation.y = float(q[1])
            t.transform.rotation.z = float(q[2])
            t.transform.rotation.w = float(q[3])
            self.tf2_broadcaster.sendTransform(t)
            
            # Give the TF buffer time to receive the broadcast via the executor
            time.sleep(0.2)
            
            try:
                badPos, badEuler = self.applyFrameChange(
                    offsetPos, offsetOri,
                    source_frame="base_link", target_frame=frame_name
                )
                if badPos is not None:
                    self.get_logger().info(f"TF lookup succeeded on attempt {attempt+1}")
                    break
            except Exception as e:
                self.get_logger().warn(f"Attempt {attempt+1}/20: TF lookup for '{frame_name}' failed: {e}")
        
        if badPos is None:
            self.get_logger().error(f"Failed to get transform to {frame_name} after 20 attempts")
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


def _print_menu():
    """Print the interactive command menu."""
    print("\n" + "=" * 50)
    print("  3D Printer Automation - Command Menu")
    print("=" * 50)
    print("  1) Scan location for markers")
    print("  2) Scan multiple locations")
    print("  3) Move to marker")
    print("  4) Lift plate")
    print("  5) Pickup plate (move + lift)")
    print("  6) Move to pose (manual)")
    print("  7) List detected markers")
    print("  8) Add movement constraint")
    print("  9) Print current end-effector pose")
    print("  0) Quit")
    print("=" * 50)


def _parse_floats(prompt, count=None):
    """Prompt for space-separated floats. Returns list of floats or None on error."""
    try:
        raw = input(prompt).strip()
        values = [float(v) for v in raw.split()]
        if count is not None and len(values) != count:
            print(f"  Expected {count} values, got {len(values)}")
            return None
        return values
    except ValueError:
        print("  Invalid input. Enter space-separated numbers.")
        return None


def _input_thread(node):
    """
    Runs in a background thread. Reads user commands from stdin
    and dispatches them on the node (which is spinning in the main thread).
    """
    # Wait for the system to initialize
    time.sleep(5.0)
    print("\n[INFO] System ready. Type a command number.")

    while rclpy.ok():
        _print_menu()
        try:
            choice = input(">> ").strip()
        except EOFError:
            break

        if choice == "1":
            pos = _parse_floats("  Enter estimated pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter estimated orient (roll pitch yaw) [0 0 0]: ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanLocationForMarkers at {pos}")
            node.scanLocationForMarkers(
                estimated_pos=pos,
                estimated_orient=orient,
                viewing_distance=dist,
            )

        elif choice == "2":
            n = _parse_floats("  How many locations? ", 1)
            if n is None:
                continue
            n = int(n[0])
            locations = []
            for i in range(n):
                pos = _parse_floats(f"  Location {i+1} pos (x y z): ", 3)
                if pos is None:
                    break
                locations.append(pos)
            if len(locations) == n:
                dist = _parse_floats("  Viewing distance [0.15]: ", 1)
                dist = dist[0] if dist else 0.15
                pause = _parse_floats("  Pause duration (s) [2.0]: ", 1)
                pause = pause[0] if pause else 2.0
                node.get_logger().info(f"User requested scanMultipleLocations ({n} locations)")
                node.scanMultipleLocations(locations, viewing_distance=dist, pause_duration=pause)

        elif choice == "3":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested moveToMarker({mid})")
            node.moveToMarker(markerID=mid)

        elif choice == "4":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested liftPlate({mid})")
            node.liftPlate(markerID=mid)

        elif choice == "5":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested pickupPlate({mid})")
            node.pickupPlate(markerID=mid)

        elif choice == "6":
            pos = _parse_floats("  Enter target pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter target orient (roll pitch yaw): ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            node.get_logger().info(f"User requested move_to_pose({pos}, {orient})")
            node.move_to_pose(np.array(pos), np.array(orient))

        elif choice == "7":
            if hasattr(node, 'marker_poses') and node.marker_poses:
                print(f"\n  Detected {len(node.marker_poses)} marker(s):")
                for entry in node.marker_poses:
                    pos = entry.get('positionInWorld', 'N/A')
                    ori = entry.get('orientInWorld', 'N/A')
                    print(f"    ID {entry['id']}: pos={pos}, orient={ori}")
            else:
                print("  No markers currently detected.")

        elif choice == "8":
            ctype = input("  Constraint type (orientation / x / y / z): ").strip()
            if ctype == "orientation":
                val = _parse_floats("  Quaternion (x y z w): ", 4)
                if val:
                    node.add_movement_constraint('orientation', tuple(val))
            elif ctype in ['x', 'y', 'z']:
                val = _parse_floats(f"  {ctype} value: ", 1)
                if val:
                    node.add_movement_constraint(ctype, val[0])
            else:
                print(f"  Unknown constraint type: {ctype}")

        elif choice == "9":
            if hasattr(node, 'pose') and node.pose is not None:
                print(f"  Current EEF pose: {node.pose}")
            else:
                print("  Pose not yet available (waiting for joint_states).")

        elif choice == "0":
            print("  Shutting down...")
            rclpy.shutdown()
            break

        else:
            print("  Unknown option. Try again.")


def main():
    rclpy.init()
    
    # Create the node FIRST so we can pass it to the printer
    node = printerAutomation(calibration_mode=False)

    printer = Simulated3DPrinter(
        node=node,
        pos=[0.0, -0.67, 0.20],
        orient=[0.0, 0.0, np.pi],
    )
    printer.spawn_fast()

    # Spin both the ROS node and the stream's internal ROS node
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.stream._ros_node)

    # Start the executor in a background thread so TF / joint_states / callbacks work
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for joint_states to arrive before scanning
    node.get_logger().info("Waiting for joint_states before initial scan...")
    for _ in range(50):  # up to 5 seconds
        if hasattr(node, 'pose') and node.pose is not None:
            break
        time.sleep(0.1)

    # Run the initial scan (blocks until move_to_pose completes)
    node.get_logger().info("Starting initial scan for markers...")
    node.scanLocationForMarkers(
        estimated_pos=[0.0, -0.67, 0.35],
        estimated_orient=[0, 0, 0],  # marker Z-axis points +Y (toward robot)
        viewing_distance=0.25
    )
    node.get_logger().info("Initial scan complete.")

    # Now start the interactive input loop on the main thread
    #_input_thread(node)



if __name__ == '__main__':
    main()