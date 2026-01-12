#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class RTABMapVisualizer(Node):
    def __init__(self):
        super().__init__('rtabmap_visualizer')
        
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.image_sub = self.create_subscription(
            Image, '/rgb_camera_moving/image', self.image_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/rtabmap/odom', self.odom_callback, qos_profile)
        
        self.path_sub = self.create_subscription(
            Path, '/rtabmap/mapPath', self.path_callback, qos_profile)
        
        self.camera_position = np.array([0.0, 0.0, 0.0])
        self.camera_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.trajectory = []
        self.frame_count = 0
        self.current_frame = None
        self.total_displacement = 0.0
        self.last_displacement = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.rtabmap_running = False
        self.rtabmap_process = None
        
        self.get_logger().info('RTABMap Visualizer started')
        self.get_logger().info('Press "q" to quit, "r" to reset, "s" to start RTABMap')
        
    def odom_callback(self, msg):
        try:
            pos = msg.pose.pose.position
            new_position = np.array([pos.x, pos.y, pos.z])
            displacement = new_position - self.last_position
            displacement_magnitude = np.linalg.norm(displacement)
            
            if displacement_magnitude > 0.001:
                self.last_displacement = displacement
                self.total_displacement += displacement_magnitude
                self.last_position = new_position.copy()
            
            self.camera_position = new_position
            orient = msg.pose.pose.orientation
            self.camera_orientation = np.array([orient.x, orient.y, orient.z, orient.w])
        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {str(e)}')
    
    def path_callback(self, msg):
        try:
            self.trajectory = []
            for pose_stamped in msg.poses:
                pos = pose_stamped.pose.position
                self.trajectory.append(np.array([pos.x, pos.y, pos.z]))
            
            if len(self.trajectory) > 100:
                self.trajectory = self.trajectory[-100:]
        except Exception as e:
            self.get_logger().error(f'Error in path callback: {str(e)}')
    
    def start_rtabmap(self):
        if self.rtabmap_running:
            self.get_logger().warn('RTABMap is already running')
            return
        
        self.get_logger().info('Starting RTABMap SLAM...')
        try:
            cmd = ['ros2', 'run', 'rtabmap_odom', 'rgbd_odometry', '--ros-args',
                   '-r', 'rgb/image:=/rgb_camera_moving/image',
                   '-r', 'rgb/camera_info:=/rgb_camera_moving/camera_info',
                   '-p', 'frame_id:=ee_camera_link', '-p', 'approx_sync:=false']
            
            self.rtabmap_process = subprocess.Popen(cmd)
            self.rtabmap_running = True
            self.get_logger().info('RTABMap started')
        except Exception as e:
            self.get_logger().error(f'Error starting RTABMap: {str(e)}')
    
    def stop_rtabmap(self):
        if not self.rtabmap_running:
            return
        
        self.get_logger().info('Stopping RTABMap...')
        try:
            if self.rtabmap_process:
                self.rtabmap_process.terminate()
                self.rtabmap_process.wait(timeout=5)
                self.rtabmap_running = False
                self.get_logger().info('RTABMap stopped')
        except Exception as e:
            self.get_logger().error(f'Error stopping RTABMap: {str(e)}')
        
    def spawn_objects(self, scene_type='random', count=20, seed=None):
        self.get_logger().info(f'Spawning {count} {scene_type} objects...')
        try:
            cmd = ['python3', '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/object_generator.py',
                   '--scene', scene_type, '--count', str(count)]
            if seed is not None:
                cmd.extend(['--seed', str(seed)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.get_logger().info('Objects spawned successfully!')
            else:
                self.get_logger().error(f'Failed to spawn objects: {result.stderr}')
        except Exception as e:
            self.get_logger().error(f'Error spawning objects: {str(e)}')
    
    def draw_overlay(self, frame):
        height, width = frame.shape[:2]
        traj_size = 200
        traj_display = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)
        
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]
            
            if np.ptp(x_coords) > 0 and np.ptp(y_coords) > 0:
                x_norm = ((x_coords - np.min(x_coords)) / np.ptp(x_coords) * (traj_size - 40) + 20).astype(int)
                y_norm = ((y_coords - np.min(y_coords)) / np.ptp(y_coords) * (traj_size - 40) + 20).astype(int)
                
                for i in range(1, len(x_norm)):
                    cv2.line(traj_display, (x_norm[i-1], y_norm[i-1]), (x_norm[i], y_norm[i]), (0, 255, 0), 2)
                
                cv2.circle(traj_display, (x_norm[-1], y_norm[-1]), 5, (0, 0, 255), -1)
        
        if width > traj_size and height > traj_size:
            frame[10:10+traj_size, width-traj_size-10:width-10] = traj_display
        
        y_offset = 20
        line_height = 25
        
        cv2.rectangle(frame, (5, 5), (380, 210), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (380, 210), (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if self.rtabmap_running else (0, 0, 255)
        status_text = "RUNNING" if self.rtabmap_running else "STOPPED"
        cv2.putText(frame, f"RTABMap: {status_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        y_offset += line_height
        
        cv2.putText(frame, "Position (m):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  X: {self.camera_position[0]:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Y: {self.camera_position[1]:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  Z: {self.camera_position[2]:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, "Displacement (m/frame):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  dX: {self.last_displacement[0]:.4f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"  dY: {self.last_displacement[1]:.4f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = height - 30
        cv2.putText(frame, f"Total Distance: {self.total_displacement:.3f} m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset = height - 10
        cv2.putText(frame, "Press: 's' start RTABMap, 'q' quit, 'r' reset", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1
            self.current_frame = cv_image
            display_frame = self.draw_overlay(cv_image.copy())
            cv2.imshow('RTABMap SLAM - Camera Feed', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit key pressed')
                self.stop_rtabmap()
                rclpy.shutdown()
            elif key == ord('r'):
                self.get_logger().info('Resetting SLAM...')
                self.camera_position = np.array([0.0, 0.0, 0.0])
                self.camera_orientation = np.array([0.0, 0.0, 0.0, 1.0])
                self.trajectory = []
                self.total_displacement = 0.0
                self.last_position = np.array([0.0, 0.0, 0.0])
                self.last_displacement = np.array([0.0, 0.0, 0.0])
            elif key == ord('s'):
                if not self.rtabmap_running:
                    threading.Thread(target=self.start_rtabmap).start()
                else:
                    self.get_logger().info('RTABMap is already running')
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    slam = RTABMapVisualizer()
    spawn_thread = threading.Thread(target=slam.spawn_objects)
    spawn_thread.start()
    
    try:
        rclpy.spin(slam)
    except KeyboardInterrupt:
        pass
    finally:
        slam.stop_rtabmap()
        cv2.destroyAllWindows()
        slam.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
