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
from tf2_geometry_msgs import do_transform_pose




class printerAutomation(ArucoDetectionViewer):
    def __init__(self, marker_size=0.05, aruco_dict='DICT_4X4_50', calibration_mode=False):
        super().__init__()
        # Override parameters after super().__init__()
        self.marker_size = marker_size
        self.aruco_dict = self._get_aruco_dict(aruco_dict)
        self.get_logger().info(f"printerAutomation initialized with marker_size={marker_size}, aruco_dict={aruco_dict}, calibration_mode={calibration_mode}")
        self.declare_parameter('marker_size', marker_size)
        self.declare_parameter('aruco_dict', aruco_dict)
        self.declare_parameter('calibration_mode', calibration_mode)

        self.timer = self.create_timer(5.0, self.moveToMarker)

    def moveToMarker(self, markerID=0):
        offsetPos = np.array([0.0, 0.0, 0.15])  # 10 cm above the marker
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



def main(args=None):
    rclpy.init(args=args)
    node = printerAutomation(marker_size=0.05, aruco_dict='DICT_4X4_50', calibration_mode=False)
    def spawn_aruco_marker(n: Node):
        delete_cmd = [
            'gz', 'service', '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', 'name: "aruco_marker" type: MODEL'
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


        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', '/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/aruco_marker/model.sdf',
            '-name', 'aruco_marker',
            '-x', '0.05', '-y', '-0.5', '-z', '0.45',
            '-R', '0', '-P', '0', '-Y', '1.57'
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

    spawn_aruco_marker(node)
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