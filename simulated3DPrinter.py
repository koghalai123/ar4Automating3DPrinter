import os
import subprocess
import tempfile
import time
import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PointStamped
from std_msgs.msg import Header
import tf_transformations
import tf2_geometry_msgs
import tf2_ros


def rotate_vector_by_quaternion(v, q):
    rot_matrix = tf_transformations.quaternion_matrix(q)[:3, :3]
    return rot_matrix @ v


class Simulated3DPrinter():
    def __init__(self, pos, orient):
        self.pos = pos
        self.orient = orient
        self.width = 0.2
        self.height = 0.2
        self.depth = 0.2

    def delete_aruco_marker(self, name: str):
        
        delete_cmd = [
            'gz', 'service', '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
        ]
        try:
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or 'Existing marker deleted successfully'
            print(f'Aruco delete: {msg}')
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            print(f'Aruco delete failed (possibly no existing marker): {err}')
        except Exception as e:
            print(f'Aruco delete exception: {e}')

    def spawn_aruco_marker(self, texture_path: str, marker_size: float, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        # Determine the name from the texture file
        
        name = os.path.basename(texture_path).split('.')[0]

        self.delete_aruco_marker(name)
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
        # Make it non-static
        modified_sdf = modified_sdf.replace('<static>true</static>', '<static>false</static>')
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
            print(f'Aruco spawn: {msg}')
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            print(f'Aruco spawn failed: {err}')
        except Exception as e:
            print(f'Aruco spawn exception: {e}')
        finally:
            # Clean up the temp file
            os.unlink(temp_sdf_path)
        return name
def spawn_wall(name, size_x, size_y, size_z, x, y, z, roll, pitch, yaw):
    # Delete existing wall if any
    delete_cmd = [
        'gz', 'service', '-s', '/world/default/remove',
        '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
        '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
    ]
    try:
        result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
        msg = result.stdout.strip() or 'Existing wall deleted successfully'
        print(f'Wall delete: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        print(f'Wall delete failed (possibly no existing wall): {err}')
    except Exception as e:
        print(f'Wall delete exception: {e}')
    
    # Create a simple box SDF
    sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
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
    # Write to a temporary file
    model_dir = '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/'  # reuse dir
    with tempfile.NamedTemporaryFile(dir=model_dir, mode='w', suffix='.sdf', delete=False) as temp_file:
        temp_file.write(sdf_content)
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
        msg = result.stdout.strip() or 'Wall spawn command executed successfully'
        print(f'Wall spawn: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        print(f'Wall spawn failed: {err}')
    except Exception as e:
        print(f'Wall spawn exception: {e}')
    finally:
        # Clean up the temp file
        os.unlink(temp_sdf_path)
    def rotate_marker_back_and_forth(name, x, y, z, initial_yaw):
        amplitude = 0.5  # radians
        frequency = 0.5  # Hz
        while True:
            t = time.time()
            delta_yaw = amplitude * math.sin(2 * math.pi * frequency * t)
            yaw = initial_yaw + delta_yaw
            sin_half = math.sin(yaw / 2)
            cos_half = math.cos(yaw / 2)
            set_cmd = [
                'gz', 'service', '-s', '/world/default/set_pose',
                '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000', '--req', f'name: "{name}" position {{ x: {x} y: {y} z: {z} }} orientation {{ x: 0 y: 0 z: {sin_half} w: {cos_half} }}'
            ]
            try:
                subprocess.run(set_cmd, capture_output=True, text=True, check=True)
                time.sleep(0.05)  # 20 fps
            except Exception as e:
                print(f'Error: {e}')
                break

def rotate_front_wall(name, initial_yaw, marker_name, local_marker_pos, pivot_offset, hinge_pos, hinge_ori, tf_broadcaster, node):
    amplitude = math.pi / 2  # from 0 to pi/2 relative
    frequency = 0.2  # Hz
    offset_to_center = -pivot_offset  # from hinge to door center
    while True:
        t = time.time()
        delta_yaw = amplitude * (math.sin(2 * math.pi * frequency * t) + 1) / 2  # oscillate between 0 and pi/2
        # Compute rotated offset in hinge frame
        cos_y = math.cos(delta_yaw)
        sin_y = math.sin(delta_yaw)
        rotated_offset = np.array([cos_y * offset_to_center[0] - sin_y * offset_to_center[1],
                                   sin_y * offset_to_center[0] + cos_y * offset_to_center[1],
                                   offset_to_center[2]])
        rotated_offset = [float(x) for x in rotated_offset]
        # Door position in world: hinge_pos + rotate rotated_offset by hinge_ori
        door_pos = np.array(hinge_pos) + rotate_vector_by_quaternion(rotated_offset, hinge_ori)
        # Door orientation: hinge_ori * q_delta
        q_delta = tf_transformations.quaternion_from_euler(0, 0, delta_yaw)
        q_delta = [float(x) for x in q_delta]
        door_ori = tf_transformations.quaternion_multiply(hinge_ori, q_delta)
        door_ori = [float(x) for x in door_ori]
        
        # Set door pose
        set_cmd = [
            'gz', 'service', '-s', '/world/default/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{name}" position {{ x: {float(door_pos[0])} y: {float(door_pos[1])} z: {float(door_pos[2])} }} orientation {{ x: {float(door_ori[0])} y: {float(door_ori[1])} z: {float(door_ori[2])} w: {float(door_ori[3])} }}'
        ]
        subprocess.run(set_cmd, capture_output=True, text=True, check=True)

        # Compute marker position and orientation
        marker_pos = door_pos + rotate_vector_by_quaternion(local_marker_pos, door_ori)
        marker_ori = door_ori  # relative identity
        
        # Set marker pose
        marker_set_cmd = [
            'gz', 'service', '-s', '/world/default/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{marker_name}" position {{ x: {float(marker_pos[0])} y: {float(marker_pos[1])} z: {float(marker_pos[2])} }} orientation {{ x: {float(marker_ori[0])} y: {float(marker_ori[1])} z: {float(marker_ori[2])} w: {float(marker_ori[3])} }}'
        ]
        subprocess.run(marker_set_cmd, capture_output=True, text=True, check=True)

        # Optionally broadcast tf frames for visualization
        door_transform = TransformStamped()
        door_transform.header.stamp = node.get_clock().now().to_msg()
        door_transform.header.frame_id = "hinge_frame"
        door_transform.child_frame_id = "door_frame"
        door_transform.transform.translation.x = rotated_offset[0]
        door_transform.transform.translation.y = rotated_offset[1]
        door_transform.transform.translation.z = rotated_offset[2]
        door_transform.transform.rotation.x = q_delta[0]
        door_transform.transform.rotation.y = q_delta[1]
        door_transform.transform.rotation.z = q_delta[2]
        door_transform.transform.rotation.w = q_delta[3]
        tf_broadcaster.sendTransform(door_transform)

        marker_transform = TransformStamped()
        marker_transform.header.stamp = node.get_clock().now().to_msg()
        marker_transform.header.frame_id = "door_frame"
        marker_transform.child_frame_id = "marker_frame"
        marker_transform.transform.translation.x = float(local_marker_pos[0])
        marker_transform.transform.translation.y = float(local_marker_pos[1])
        marker_transform.transform.translation.z = float(local_marker_pos[2])
        marker_transform.transform.rotation.x = 0
        marker_transform.transform.rotation.y = 0
        marker_transform.transform.rotation.z = 0
        marker_transform.transform.rotation.w = 1
        tf_broadcaster.sendTransform(marker_transform)

        time.sleep(0.02)

def main():
    rclpy.init()
    node = Node('wall_spawner')
    tf_broadcaster = TransformBroadcaster(node)
    buffer = Buffer()
    listener = TransformListener(buffer, node)

    pos = [0.0, -0.7, 0.1]
    orient = [0.0, 0.0, 0.5]
    printerObj = Simulated3DPrinter(pos, orient)

    # Broadcast printer frame
    transform = TransformStamped()
    transform.header.stamp = node.get_clock().now().to_msg()
    transform.header.frame_id = "world"
    transform.child_frame_id = "printer_frame"
    transform.transform.translation.x = float(pos[0])
    transform.transform.translation.y = float(pos[1])
    transform.transform.translation.z = float(pos[2])
    q = tf_transformations.quaternion_from_euler(float(orient[0]), float(orient[1]), float(orient[2]))
    q = [float(x) for x in q]
    q_marker = tf_transformations.quaternion_multiply(q, tf_transformations.quaternion_from_euler(0, 0, math.pi/2))
    q_marker = [float(x) for x in q_marker]
    transform.transform.rotation.x = q[0]
    transform.transform.rotation.y = q[1]
    transform.transform.rotation.z = q[2]
    transform.transform.rotation.w = q[3]
    tf_broadcaster.sendTransform(transform)

    # Wait for tf to propagate
    rclpy.spin_once(node, timeout_sec=0.1)

    # Define printer dimensions
    printer_width = 0.3
    printer_depth = 0.3
    printer_height = 0.3
    thickness = 0.01

    # Spawn ArUco markers relative to printer
    texture_path = 'materials/textures/marker6x6_0.png'
    marker_size = 0.05
    roll = float(orient[0])
    pitch = float(orient[1])
    yaw = float(orient[2]) + math.pi/2
    # Outside front wall
    ox = 0
    oy = -printer_depth/2 - 0.02  # outside
    oz = 0
    point_stamped = PointStamped()
    point_stamped.header.frame_id = "printer_frame"
    point_stamped.header.stamp = node.get_clock().now().to_msg()
    point_stamped.point.x = ox
    point_stamped.point.y = oy
    point_stamped.point.z = oz
    transform = buffer.lookup_transform("world", "printer_frame", rclpy.time.Time())
    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
    x = transformed.point.x
    y = transformed.point.y
    z = transformed.point.z
    name1 = printerObj.spawn_aruco_marker(texture_path, marker_size, x, y, z, roll, pitch, yaw)
    # Broadcast marker frame
    marker_transform = TransformStamped()
    marker_transform.header.stamp = node.get_clock().now().to_msg()
    marker_transform.header.frame_id = "world"
    marker_transform.child_frame_id = "marker6x6_frame"
    marker_transform.transform.translation.x = x
    marker_transform.transform.translation.y = y
    marker_transform.transform.translation.z = z
    marker_transform.transform.rotation.x = q_marker[0]
    marker_transform.transform.rotation.y = q_marker[1]
    marker_transform.transform.rotation.z = q_marker[2]
    marker_transform.transform.rotation.w = q_marker[3]
    tf_broadcaster.sendTransform(marker_transform)

    rclpy.spin_once(node, timeout_sec=0.01)
    
    texture_path = 'materials/textures/marker4x4_0.png'
    marker_size = 0.03
    # At the center of the front face
    ox = 0
    oy = -printer_depth/2 - thickness/2  # center of front face
    oz = 0
    point_stamped.point.x = ox
    point_stamped.point.y = oy
    point_stamped.point.z = oz
    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
    x = transformed.point.x
    y = transformed.point.y
    z = transformed.point.z
    name2 = printerObj.spawn_aruco_marker(texture_path, marker_size, x, y, z, roll, pitch, yaw)
    # Broadcast marker frame
    marker_transform.child_frame_id = "marker4x4_frame"
    marker_transform.transform.translation.x = x
    marker_transform.transform.translation.y = y
    marker_transform.transform.translation.z = z
    marker_transform.transform.rotation.x = q_marker[0]
    marker_transform.transform.rotation.y = q_marker[1]
    marker_transform.transform.rotation.z = q_marker[2]
    marker_transform.transform.rotation.w = q_marker[3]
    tf_broadcaster.sendTransform(marker_transform)

    rclpy.spin_once(node, timeout_sec=0.01)

    # Spawn printer walls
    roll = float(orient[0])
    pitch = float(orient[1])
    yaw = float(orient[2])

    walls = [
        ("bottom", 0, 0, -printer_height/2, printer_width, printer_depth, thickness),
        ("top", 0, 0, printer_height/2, printer_width, printer_depth, thickness),
        ("left", -printer_width/2, 0, 0, thickness, printer_depth, printer_height),
        ("right", printer_width/2, 0, 0, thickness, printer_depth, printer_height),
        ("front", 0, -printer_depth/2, 0, printer_width, thickness, printer_height),
        ("back", 0, printer_depth/2, 0, printer_width, thickness, printer_height),
    ]

    front_x = front_y = front_z = front_yaw = 0

    for name, ox, oy, oz, sx, sy, sz in walls:
        # Transform offset from printer_frame to world
        point_stamped = PointStamped()
        point_stamped.header.frame_id = "printer_frame"
        point_stamped.header.stamp = node.get_clock().now().to_msg()
        point_stamped.point.x = ox
        point_stamped.point.y = oy
        point_stamped.point.z = oz
        try:
            transform = buffer.lookup_transform("world", "printer_frame", rclpy.time.Time())
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            wx = transformed.point.x
            wy = transformed.point.y
            wz = transformed.point.z
            spawn_wall(name, sx, sy, sz, wx, wy, wz, roll, pitch, yaw)
            if name == "front":
                front_x, front_y, front_z, front_yaw = wx, wy, wz, yaw
        except Exception as e:
            print(f"TF error for {name}: {e}")

    # Broadcast hinge frame for front wall
    hinge_pos = np.array([front_x - printer_width/2, front_y, front_z])  # left edge
    hinge_pos = [float(x) for x in hinge_pos]
    hinge_transform = TransformStamped()
    hinge_transform.header.stamp = node.get_clock().now().to_msg()
    hinge_transform.header.frame_id = "world"
    hinge_transform.child_frame_id = "hinge_frame"
    hinge_transform.transform.translation.x = hinge_pos[0]
    hinge_transform.transform.translation.y = hinge_pos[1]
    hinge_transform.transform.translation.z = hinge_pos[2]
    hinge_transform.transform.rotation.x = q[0]
    hinge_transform.transform.rotation.y = q[1]
    hinge_transform.transform.rotation.z = q[2]
    hinge_transform.transform.rotation.w = q[3]
    tf_broadcaster.sendTransform(hinge_transform)

    rclpy.spin_once(node, timeout_sec=0.01)

    hinge_pos = [hinge_transform.transform.translation.x,
                 hinge_transform.transform.translation.y,
                 hinge_transform.transform.translation.z]
    hinge_ori = q

    # Start rotating the front wall
    pivot_offset = np.array([-printer_width/2, 0, 0])  # rotate about left edge
    local_marker_pos = np.array([0, -thickness/2, 0])  # marker at center of front face relative to wall
    threading.Thread(target=rotate_front_wall, args=("front", front_yaw, name2, local_marker_pos, pivot_offset, hinge_pos, hinge_ori, tf_broadcaster, node)).start()

    # Keep the node alive
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()