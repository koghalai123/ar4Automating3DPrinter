import os
import subprocess
import tempfile
import time
import math
import threading
import re
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PointStamped, PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import tf_transformations
import tf2_geometry_msgs
import tf2_ros
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


def rotate_vector_by_quaternion(v, q):
    """Rotate a 3D vector by a quaternion."""
    rot_matrix = tf_transformations.quaternion_matrix(q)[:3, :3]
    return rot_matrix @ v


class Simulated3DPrinter:
    """
    A simulated 3D printer with walls, a swinging door, and ArUco markers.
    
    Parameters:
        node: ROS2 node for communication
        pos: Position [x, y, z] in world frame
        orient: Orientation [roll, pitch, yaw] in radians
        width: Printer width (default 0.3)
        depth: Printer depth (default 0.3)
        height: Printer height (default 0.3)
        wall_thickness: Wall thickness (default 0.01)
        door_frequency: Door oscillation frequency in Hz (default 0.2)
        door_amplitude: Door swing amplitude in radians (default pi/2)
        model_dir: Directory for temporary SDF files
    """
    
    def __init__(self, node, pos, orient, width=0.3, depth=0.3, height=0.3,
                 wall_thickness=0.01, door_frequency=0.2, door_amplitude=math.pi/2,
                 model_dir='/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/',
                 aruco_sdf_path='/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf'):
        
        self.node = node
        self.pos = np.array(pos)
        self.orient = np.array(orient)
        self.width = width
        self.depth = depth
        self.height = height
        self.wall_thickness = wall_thickness
        self.door_frequency = door_frequency
        self.door_amplitude = door_amplitude
        self.model_dir = model_dir
        self.aruco_sdf_path = aruco_sdf_path
        
        # TF components
        self.tf_broadcaster = TransformBroadcaster(node)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        
        # Service client for setting poses (will be set up later)
        self.set_pose_client = None
        
        # Compute quaternions
        self.q = tf_transformations.quaternion_from_euler(
            float(self.orient[0]), float(self.orient[1]), float(self.orient[2])
        )
        self.q = [float(x) for x in self.q]
        self.q_marker = tf_transformations.quaternion_multiply(
            self.q, tf_transformations.quaternion_from_euler(0, 0, math.pi/2)
        )
        self.q_marker = [float(x) for x in self.q_marker]
        
        # Store spawned entity names for cleanup
        self.spawned_entities = []
        self.markers = {}
        self.walls = {}
        
        # Publishers for pose tracking
        self.door_pose_pub = None
        self.marker_pose_pub = None
        
        # Animation thread
        self.animation_thread = None
        self.running = False

    def setup_pose_service(self, set_pose_client):
        """Set the service client for pose updates."""
        self.set_pose_client = set_pose_client

    def _broadcast_printer_frame(self):
        """Broadcast the printer's TF frame."""
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = f"printer_frame"
        transform.transform.translation.x = float(self.pos[0])
        transform.transform.translation.y = float(self.pos[1])
        transform.transform.translation.z = float(self.pos[2])
        transform.transform.rotation.x = self.q[0]
        transform.transform.rotation.y = self.q[1]
        transform.transform.rotation.z = self.q[2]
        transform.transform.rotation.w = self.q[3]
        self.tf_broadcaster.sendTransform(transform)
        return transform

    def _get_transform_with_retry(self, target_frame, source_frame, timeout_sec=5.0):
        """Get a TF transform with retry logic."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self._broadcast_printer_frame()
                rclpy.spin_once(self.node, timeout_sec=0.1)
        raise tf2_ros.LookupException(
            f"Could not get transform from {source_frame} to {target_frame} after {timeout_sec}s"
        )

    def _transform_point_to_world(self, local_x, local_y, local_z):
        """Transform a point from printer frame to world frame."""
        point_stamped = PointStamped()
        point_stamped.header.frame_id = "printer_frame"
        point_stamped.header.stamp = self.node.get_clock().now().to_msg()
        point_stamped.point.x = float(local_x)
        point_stamped.point.y = float(local_y)
        point_stamped.point.z = float(local_z)
        
        transform = self._get_transform_with_retry("world", "printer_frame")
        transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return transformed.point.x, transformed.point.y, transformed.point.z

    def _delete_entity(self, name):
        """Delete an entity from Gazebo."""
        delete_cmd = [
            'gz', 'service', '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
        ]
        try:
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or f'{name} deleted successfully'
            self.node.get_logger().debug(f'Delete: {msg}')
        except subprocess.CalledProcessError as e:
            pass  # Entity might not exist
        except Exception as e:
            self.node.get_logger().warn(f'Delete exception for {name}: {e}')

    def spawn_aruco_marker(self, texture_path, marker_size, local_pos, name_prefix="marker"):
        """
        Spawn an ArUco marker relative to the printer.
        
        Parameters:
            texture_path: Path to the marker texture
            marker_size: Size of the marker
            local_pos: [x, y, z] position in printer frame
            name_prefix: Prefix for the marker name
        
        Returns:
            name: The spawned marker's name
        """
        name = os.path.basename(texture_path).split('.')[0]
        self._delete_entity(name)
        
        # Read and modify SDF
        with open(self.aruco_sdf_path, 'r') as f:
            sdf_content = f.read()
        
        # Replace texture
        old_texture_line = '              <albedo_map>materials/textures/marker.png</albedo_map>'
        new_texture_line = f'              <albedo_map>{texture_path}</albedo_map>'
        modified_sdf = sdf_content.replace(old_texture_line, new_texture_line)
        
        # Replace size
        old_size_line = f'            <size>0.0001 {0.05} {0.05}</size>'
        new_size_line = f'            <size>0.0001 {marker_size} {marker_size}</size>'
        modified_sdf = modified_sdf.replace(old_size_line, new_size_line)
        
        # Make non-static and remove collisions
        modified_sdf = modified_sdf.replace('<static>true</static>', '<static>false</static>')
        modified_sdf = re.sub(r'<collision[^>]*>.*?</collision>', '', modified_sdf, flags=re.DOTALL)
        
        # Write temp file and spawn
        with tempfile.NamedTemporaryFile(dir=self.model_dir, mode='w', suffix='.sdf', delete=False) as temp_file:
            temp_file.write(modified_sdf)
            temp_sdf_path = temp_file.name
        
        # Transform position to world
        x, y, z = self._transform_point_to_world(*local_pos)
        roll, pitch = float(self.orient[0]), float(self.orient[1])
        yaw = float(self.orient[2]) + math.pi/2
        
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', temp_sdf_path, '-name', name,
            '-x', str(x), '-y', str(y), '-z', str(z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.node.get_logger().info(f'Spawned marker: {name}')
            self.spawned_entities.append(name)
            self.markers[name] = {'local_pos': local_pos, 'size': marker_size, 'world_pos': [x, y, z]}
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'Failed to spawn marker {name}: {e.stderr or e.stdout}')
        finally:
            os.unlink(temp_sdf_path)
        
        return name

    def spawn_wall(self, name, size, local_pos):
        """
        Spawn a wall relative to the printer.
        
        Parameters:
            name: Wall name
            size: [size_x, size_y, size_z]
            local_pos: [x, y, z] position in printer frame
        
        Returns:
            World position [x, y, z]
        """
        self._delete_entity(name)
        
        # Create SDF (no collision for visual-only walls)
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{size[0]} {size[1]} {size[2]}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        
        with tempfile.NamedTemporaryFile(dir=self.model_dir, mode='w', suffix='.sdf', delete=False) as temp_file:
            temp_file.write(sdf_content)
            temp_sdf_path = temp_file.name
        
        x, y, z = self._transform_point_to_world(*local_pos)
        roll, pitch, yaw = float(self.orient[0]), float(self.orient[1]), float(self.orient[2])
        
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', temp_sdf_path, '-name', name,
            '-x', str(x), '-y', str(y), '-z', str(z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.node.get_logger().info(f'Spawned wall: {name}')
            self.spawned_entities.append(name)
            self.walls[name] = {'local_pos': local_pos, 'size': size, 'world_pos': [x, y, z]}
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'Failed to spawn wall {name}: {e.stderr or e.stdout}')
        finally:
            os.unlink(temp_sdf_path)
        
        return [x, y, z]

    def spawn_all_walls(self):
        """Spawn all printer walls (bottom, top, left, right, front, back)."""
        t = self.wall_thickness
        w, d, h = self.width, self.depth, self.height
        
        wall_configs = [
            ("bottom", [w, d, t], [0, 0, -h/2]),
            ("top", [w, d, t], [0, 0, h/2]),
            ("left", [t, d, h], [-w/2, 0, 0]),
            ("right", [t, d, h], [w/2, 0, 0]),
            ("front", [w, t, h], [0, -d/2, 0]),
            ("back", [w, t, h], [0, d/2, 0]),
        ]
        
        for name, size, local_pos in wall_configs:
            self.spawn_wall(name, size, local_pos)

    def spawn_door_marker(self, texture_path, marker_size):
        """
        Spawn a marker attached to the front door.
        
        Parameters:
            texture_path: Path to the marker texture
            marker_size: Size of the marker
        
        Returns:
            Marker name
        """
        # Position at center of front face, slightly outside
        local_pos = [0, -self.depth/2 - self.wall_thickness/2, 0]
        return self.spawn_aruco_marker(texture_path, marker_size, local_pos, "door_marker")

    def _animate_door(self, door_name, marker_name, local_marker_pos):
        """Animation loop for the swinging door."""
        offset_to_center = np.array([self.width/2, 0, 0])  # from hinge to door center
        
        # Compute hinge position in world
        hinge_local = [-self.width/2, -self.depth/2, 0]
        hinge_world = list(self._transform_point_to_world(*hinge_local))
        
        # Broadcast hinge frame
        hinge_transform = TransformStamped()
        hinge_transform.header.frame_id = "world"
        hinge_transform.child_frame_id = "hinge_frame"
        hinge_transform.transform.translation.x = hinge_world[0]
        hinge_transform.transform.translation.y = hinge_world[1]
        hinge_transform.transform.translation.z = hinge_world[2]
        hinge_transform.transform.rotation.x = self.q[0]
        hinge_transform.transform.rotation.y = self.q[1]
        hinge_transform.transform.rotation.z = self.q[2]
        hinge_transform.transform.rotation.w = self.q[3]
        
        while self.running:
            t = time.time()
            # Oscillate between 0 and -amplitude
            delta_yaw = -self.door_amplitude * (math.sin(2 * math.pi * self.door_frequency * t) + 1) / 2
            
            # Compute rotated offset
            cos_y, sin_y = math.cos(delta_yaw), math.sin(delta_yaw)
            rotated_offset = np.array([
                cos_y * offset_to_center[0] - sin_y * offset_to_center[1],
                sin_y * offset_to_center[0] + cos_y * offset_to_center[1],
                offset_to_center[2]
            ])
            
            # Door position and orientation in world
            door_pos = np.array(hinge_world) + rotate_vector_by_quaternion(rotated_offset, self.q)
            q_delta = tf_transformations.quaternion_from_euler(0, 0, delta_yaw)
            door_ori = tf_transformations.quaternion_multiply(self.q, q_delta)
            door_ori = [float(x) for x in door_ori]
            
            # Marker position and orientation (with extra 90° rotation)
            marker_pos = door_pos + rotate_vector_by_quaternion(local_marker_pos, door_ori)
            q_marker_offset = tf_transformations.quaternion_from_euler(0, 0, math.pi/2)
            marker_ori = tf_transformations.quaternion_multiply(door_ori, q_marker_offset)
            marker_ori = [float(x) for x in marker_ori]
            
            # Update Gazebo poses via service
            if self.set_pose_client:
                try:
                    # Door pose
                    door_request = SetEntityPose.Request()
                    door_request.entity = Entity()
                    door_request.entity.name = door_name
                    door_request.entity.type = Entity.MODEL
                    door_request.pose = Pose()
                    door_request.pose.position = Point(x=float(door_pos[0]), y=float(door_pos[1]), z=float(door_pos[2]))
                    door_request.pose.orientation = Quaternion(x=float(door_ori[0]), y=float(door_ori[1]), z=float(door_ori[2]), w=float(door_ori[3]))
                    self.set_pose_client.call_async(door_request)
                    
                    # Marker pose
                    marker_request = SetEntityPose.Request()
                    marker_request.entity = Entity()
                    marker_request.entity.name = marker_name
                    marker_request.entity.type = Entity.MODEL
                    marker_request.pose = Pose()
                    marker_request.pose.position = Point(x=float(marker_pos[0]), y=float(marker_pos[1]), z=float(marker_pos[2]))
                    marker_request.pose.orientation = Quaternion(x=float(marker_ori[0]), y=float(marker_ori[1]), z=float(marker_ori[2]), w=float(marker_ori[3]))
                    self.set_pose_client.call_async(marker_request)
                except Exception as e:
                    self.node.get_logger().warn(f'Failed to set pose: {e}')
            
            # Publish poses for other nodes
            if self.door_pose_pub:
                door_msg = PoseStamped()
                door_msg.header.stamp = self.node.get_clock().now().to_msg()
                door_msg.header.frame_id = "world"
                door_msg.pose.position = Point(x=float(door_pos[0]), y=float(door_pos[1]), z=float(door_pos[2]))
                door_msg.pose.orientation = Quaternion(x=float(door_ori[0]), y=float(door_ori[1]), z=float(door_ori[2]), w=float(door_ori[3]))
                self.door_pose_pub.publish(door_msg)
            
            if self.marker_pose_pub:
                marker_msg = PoseStamped()
                marker_msg.header.stamp = self.node.get_clock().now().to_msg()
                marker_msg.header.frame_id = "world"
                marker_msg.pose.position = Point(x=float(marker_pos[0]), y=float(marker_pos[1]), z=float(marker_pos[2]))
                marker_msg.pose.orientation = Quaternion(x=float(marker_ori[0]), y=float(marker_ori[1]), z=float(marker_ori[2]), w=float(marker_ori[3]))
                self.marker_pose_pub.publish(marker_msg)
            
            # Broadcast TF frames
            hinge_transform.header.stamp = self.node.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(hinge_transform)
            
            door_transform = TransformStamped()
            door_transform.header.stamp = self.node.get_clock().now().to_msg()
            door_transform.header.frame_id = "hinge_frame"
            door_transform.child_frame_id = "door_frame"
            door_transform.transform.translation.x = float(rotated_offset[0])
            door_transform.transform.translation.y = float(rotated_offset[1])
            door_transform.transform.translation.z = float(rotated_offset[2])
            door_transform.transform.rotation.x = float(q_delta[0])
            door_transform.transform.rotation.y = float(q_delta[1])
            door_transform.transform.rotation.z = float(q_delta[2])
            door_transform.transform.rotation.w = float(q_delta[3])
            self.tf_broadcaster.sendTransform(door_transform)
            
            marker_transform = TransformStamped()
            marker_transform.header.stamp = self.node.get_clock().now().to_msg()
            marker_transform.header.frame_id = "door_frame"
            marker_transform.child_frame_id = "marker_frame"
            marker_transform.transform.translation.x = float(local_marker_pos[0])
            marker_transform.transform.translation.y = float(local_marker_pos[1])
            marker_transform.transform.translation.z = float(local_marker_pos[2])
            marker_transform.transform.rotation.x = 0.0
            marker_transform.transform.rotation.y = 0.0
            marker_transform.transform.rotation.z = 0.0
            marker_transform.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(marker_transform)
            
            time.sleep(0.02)  # 50Hz

    def start_door_animation(self, marker_name):
        """Start the door animation with attached marker."""
        if self.animation_thread and self.animation_thread.is_alive():
            self.node.get_logger().warn('Animation already running')
            return
        
        # Create pose publishers
        self.door_pose_pub = self.node.create_publisher(PoseStamped, '/door_world_pose', 10)
        self.marker_pose_pub = self.node.create_publisher(PoseStamped, '/marker_world_pose', 10)
        
        # Local marker position relative to door center
        local_marker_pos = np.array([0, -self.wall_thickness/2, 0])
        
        self.running = True
        self.animation_thread = threading.Thread(
            target=self._animate_door,
            args=("front", marker_name, local_marker_pos),
            daemon=True
        )
        self.animation_thread.start()
        self.node.get_logger().info('Door animation started')

    def stop_animation(self):
        """Stop the door animation."""
        self.running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
        self.node.get_logger().info('Door animation stopped')

    def spawn(self):
        """Spawn the complete printer with walls and initialize TF."""
        # Broadcast printer frame and wait for TF
        for _ in range(10):
            self._broadcast_printer_frame()
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        # Spawn all walls
        self.spawn_all_walls()
        
        self.node.get_logger().info(f'Printer spawned at position {self.pos}')

    def cleanup(self):
        """Remove all spawned entities."""
        self.stop_animation()
        for name in self.spawned_entities:
            self._delete_entity(name)
        self.spawned_entities.clear()
        self.markers.clear()
        self.walls.clear()


def main():
    rclpy.init()
    node = Node('simulated_printer')
    
    # Start the ros_gz_bridge for the set_pose service
    bridge_proc = subprocess.Popen(
        ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
         '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    node.get_logger().info('Started ros_gz_bridge for set_pose service')
    
    # Create service client
    set_pose_client = node.create_client(SetEntityPose, '/world/default/set_pose')
    node.get_logger().info('Waiting for set_pose service...')
    if set_pose_client.wait_for_service(timeout_sec=10.0):
        node.get_logger().info('SetEntityPose service available')
    else:
        node.get_logger().warn('SetEntityPose service not available')
    
    # ============================================================
    # CONFIGURATION - Edit these values to customize the printer
    # ============================================================
    printer_config = {
        'pos': [0.0, -0.7, 0.1],
        'orient': [0.0, 0.0, 0.5],  # roll, pitch, yaw in radians
        'width': 0.3,
        'depth': 0.3,
        'height': 0.3,
        'wall_thickness': 0.01,
        'door_frequency': 0.2,  # Hz
        'door_amplitude': math.pi / 2,  # radians
    }
    
    door_marker_config = {
        'texture_path': 'materials/textures/marker4x4_0.png',
        'marker_size': 0.03,
    }
    
    # Optional: additional static marker
    static_marker_config = {
        'texture_path': 'materials/textures/marker6x6_0.png',
        'marker_size': 0.05,
        'local_pos': [0, -0.15 - 0.02, 0],  # Outside front wall
    }
    # ============================================================
    
    # Create and spawn the printer
    printer = Simulated3DPrinter(node, **printer_config)
    printer.setup_pose_service(set_pose_client)
    printer.spawn()
    
    # Spawn optional static marker
    if static_marker_config:
        printer.spawn_aruco_marker(
            static_marker_config['texture_path'],
            static_marker_config['marker_size'],
            static_marker_config['local_pos']
        )
    
    # Spawn door marker and start animation
    door_marker_name = printer.spawn_door_marker(
        door_marker_config['texture_path'],
        door_marker_config['marker_size']
    )
    printer.start_door_animation(door_marker_name)
    
    # Keep node alive
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        printer.cleanup()
        bridge_proc.terminate()
        bridge_proc.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
