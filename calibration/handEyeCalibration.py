#!/usr/bin/env python3
"""Safe eye-in-hand calibration for the xArm wrist camera and ChArUco board.

The board stays fixed. Move the robot manually with the commissioned GUI or
UFACTORY Studio and press C for each sample. This program never commands the
robot. The result is the measured link_eef -> camera optical TF.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as Rotation
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from xarm_msgs.srv import SetInt16
from xarm_msgs.msg import RobotMsg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ar4_automation.robot_config import get_robot_config
from calibration.CameraCalibration import (
    ARUCO_DICT, MARKER_LENGTH, SQUARE_LENGTH, SQUARES_X, SQUARES_Y)
from calibration.charuco_utils import (
    create_board, detect, detector_parameters, estimate_pose)
from ar4_automation.web_video_server import select_camera


def matrix(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def transform_from_msg(msg):
    t, q = msg.transform.translation, msg.transform.rotation
    return matrix(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                  [t.x, t.y, t.z])


def calibration_error(base_to_gripper, camera_to_board,
                      gripper_to_camera):
    board_poses = [bg @ gripper_to_camera @ cb
                   for bg, cb in zip(base_to_gripper, camera_to_board)]
    positions = np.array([pose[:3, 3] for pose in board_poses])
    centre = positions.mean(axis=0)
    translation_rmse = float(np.sqrt(np.mean(np.sum(
        (positions - centre) ** 2, axis=1))))
    rotations = Rotation.from_matrix([pose[:3, :3] for pose in board_poses])
    mean_rotation = rotations.mean()
    angle_errors = (mean_rotation.inv() * rotations).magnitude()
    return {
        "translation_rmse_m": translation_rmse,
        "translation_max_m": float(np.max(np.linalg.norm(
            positions - centre, axis=1))),
        "rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(
            angle_errors ** 2)))),
        "board_position_in_base_m": centre.tolist(),
    }


def solve_hand_eye(base_to_gripper, camera_to_board):
    if len(base_to_gripper) < 8:
        raise ValueError("At least 8 captures are required (15-25 recommended)")
    positions = np.array([pose[:3, 3] for pose in base_to_gripper])
    translation_span = max(
        np.linalg.norm(a - b) for a in positions for b in positions)
    rotations = [Rotation.from_matrix(pose[:3, :3])
                 for pose in base_to_gripper]
    rotation_span = max(
        (a.inv() * b).magnitude() for a in rotations for b in rotations)
    if translation_span < 0.05 or rotation_span < np.radians(20.0):
        raise ValueError(
            "Capture poses lack diversity: need at least 0.05 m translation "
            "span and 20 deg rotation span")
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    candidates = []
    for name, method in methods.items():
        try:
            r, t = cv2.calibrateHandEye(
                [x[:3, :3] for x in base_to_gripper],
                [x[:3, 3] for x in base_to_gripper],
                [x[:3, :3] for x in camera_to_board],
                [x[:3, 3] for x in camera_to_board], method=method)
            tf = matrix(r, t)
            metrics = calibration_error(base_to_gripper, camera_to_board, tf)
            if np.isfinite(tf).all():
                candidates.append((metrics["translation_rmse_m"], name,
                                   tf, metrics))
        except cv2.error:
            continue
    if not candidates:
        raise RuntimeError("OpenCV could not solve hand-eye calibration")
    return min(candidates, key=lambda item: item[0])


class Collector(Node):
    def __init__(self, robot, camera_index=None, camera_keyword=None,
                 camera_calibration=None):
        super().__init__("charuco_hand_eye_collector")
        config = get_robot_config(robot)
        self.base_frame = config["base_link"]
        self.gripper_frame = config["end_effector_link"]
        self.bridge = CvBridge()
        self.image = None
        self.camera_matrix = None
        self.distortion = None
        self.camera_frame = None
        self.got_image = False
        self.got_camera_info = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cap = None
        self.controller_mode = None
        namespace = config["xarm_safety"]["namespace"].rstrip("/")
        self.create_subscription(RobotMsg, f"{namespace}/robot_states",
                                 self.on_robot_state, 10)
        if camera_index is not None or camera_keyword is not None:
            if camera_index is None:
                camera_index = select_camera(
                    preset_keyword=camera_keyword or "GENERAL WEBCAM")
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open USB camera index {camera_index}")
            calibration = np.load(camera_calibration)
            self.camera_matrix = calibration["camera_matrix"]
            self.distortion = calibration["dist_coeffs"]
            self.camera_frame = "usb_camera_optical_frame"
            self.got_camera_info = True
            self.get_logger().info(
                f"USB camera index {camera_index}; calibration={camera_calibration}")
        else:
            self.create_subscription(Image, config["color_topic"],
                                     self.on_image, 1)
            self.create_subscription(CameraInfo, config["camera_info_topic"],
                                     self.on_info, 10)

    def update_usb_image(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.image = frame
            self.got_image = True

    def on_robot_state(self, msg):
        self.controller_mode = int(msg.mode)

    def on_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.camera_frame = msg.header.frame_id.lstrip("/")
        self.got_image = True

    def on_info(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        self.distortion = np.asarray(msg.d, dtype=float)
        self.got_camera_info = True

    def base_to_gripper(self):
        msg = self.tf_buffer.lookup_transform(
            self.base_frame, self.gripper_frame, Time())
        return transform_from_msg(msg)

    def set_controller(self, namespace, mode):
        """Set UFACTORY mode and START state, waiting for both responses."""
        for service, value in (("set_mode", mode), ("set_state", 0)):
            client = self.create_client(SetInt16, f"{namespace}/{service}")
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"service unavailable: {namespace}/{service}")
            request = SetInt16.Request()
            request.data = value
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or future.result() is None:
                raise RuntimeError(f"no response from {namespace}/{service}")
            response = future.result()
            if int(response.ret) != 0:
                raise RuntimeError(
                    f"{namespace}/{service} rejected {value}: "
                    f"ret={response.ret} message={response.message}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("lite6", "xarm6"), default="xarm6")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("calibration/xarm6_hand_eye.json"))
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument(
        "--teach-mode", action="store_true",
        help="put the physical arm in UFACTORY mode 2 and restore mode 0 on exit")
    parser.add_argument("--usb-camera", action="store_true",
                        help="open a traditional USB webcam with OpenCV")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--camera-keyword", default="GENERAL WEBCAM")
    parser.add_argument(
        "--camera-calibration", type=pathlib.Path,
        default=pathlib.Path("calibration/camera_matrix.npz"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init()
    if args.camera_index is not None:
        args.usb_camera = True
    if args.usb_camera and not args.camera_calibration.is_file():
        raise FileNotFoundError(
            f"camera calibration not found: {args.camera_calibration}")
    node = Collector(
        args.robot,
        camera_index=args.camera_index,
        camera_keyword=args.camera_keyword if args.usb_camera else None,
        camera_calibration=str(args.camera_calibration) if args.usb_camera else None)
    namespace = get_robot_config(args.robot)["xarm_safety"]["namespace"].rstrip("/")
    teach_enabled = False
    restore_mode = None
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = create_board(SQUARES_X, SQUARES_Y, SQUARE_LENGTH,
                         MARKER_LENGTH, dictionary)
    parameters = detector_parameters()
    robot_poses, board_poses = [], []
    print("Keep the ChArUco board fixed. Move the arm between captures.")
    print("C=capture, S=solve/save, Q=quit. Target 15-25 varied poses.")
    try:
        cv2.namedWindow("xArm ChArUco hand-eye calibration",
                        cv2.WINDOW_NORMAL)
        cv2.resizeWindow("xArm ChArUco hand-eye calibration", 960, 540)
        if args.teach_mode:
            deadline = time.monotonic() + 3.0
            while node.controller_mode is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.controller_mode is None:
                raise RuntimeError(
                    f"no controller telemetry on {namespace}/robot_states")
            restore_mode = node.controller_mode
            print("Enabling UFACTORY teach mode (mode 2)...")
            node.set_controller(namespace, 2)
            teach_enabled = True
            print("Teach mode active. Support the arm before moving it.")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            node.update_usb_image()
            if node.image is None:
                view = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(view, "Waiting for camera image...", (35, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
                config = get_robot_config(args.robot)
                cv2.putText(view, config["color_topic"], (35, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
                cv2.imshow("xArm ChArUco hand-eye calibration", view)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            view = node.image.copy()
            if node.camera_matrix is None:
                cv2.putText(view, "Waiting for CameraInfo...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("xArm ChArUco hand-eye calibration", view)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            mc, mi, _, cc, ci = detect(gray, board, dictionary, parameters)
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(view, mc, mi)
            valid = cc is not None and len(cc) >= args.min_corners
            pose = None
            if valid:
                ok, rvec, tvec = estimate_pose(
                    cc, ci, board, node.camera_matrix, node.distortion)
                if ok:
                    pose = matrix(cv2.Rodrigues(rvec)[0], tvec)
                    cv2.drawFrameAxes(view, node.camera_matrix,
                                      node.distortion, rvec, tvec, 0.04)
            cv2.putText(view, f"samples={len(robot_poses)} corners={0 if cc is None else len(cc)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if pose is not None else (0, 0, 255), 2)
            cv2.imshow("xArm ChArUco hand-eye calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if pose is None:
                    print("Capture rejected: board pose is not reliable")
                    continue
                try:
                    robot_poses.append(node.base_to_gripper())
                    board_poses.append(pose)
                    print(f"Captured sample {len(robot_poses)}")
                except Exception as exc:
                    print(f"Capture rejected: TF unavailable: {exc}")
            if key == ord("s"):
                score, method, tf, metrics = solve_hand_eye(
                    robot_poses, board_poses)
                rotation = Rotation.from_matrix(tf[:3, :3])
                result = {
                    "schema_version": 1,
                    "robot": args.robot,
                    "parent_frame": node.gripper_frame,
                    "child_frame": node.camera_frame,
                    "method": method,
                    "sample_count": len(robot_poses),
                    "translation_m": tf[:3, 3].tolist(),
                    "quaternion_xyzw": rotation.as_quat().tolist(),
                    "rpy_rad": rotation.as_euler("xyz").tolist(),
                    "matrix": tf.tolist(),
                    "metrics": metrics,
                    "board": {"squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                              "square_length_m": SQUARE_LENGTH,
                              "marker_length_m": MARKER_LENGTH,
                              "dictionary": "DICT_4X4_50"},
                }
                quality_ok = (metrics["translation_rmse_m"] <= 0.01 and
                              metrics["rotation_rmse_deg"] <= 3.0)
                result["quality_passed"] = quality_ok
                if not quality_ok:
                    print(json.dumps(result, indent=2))
                    print("Calibration rejected: require translation RMSE <= "
                          "0.010 m and rotation RMSE <= 3.0 deg")
                    continue
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2))
                print(f"Saved {args.output}")
                break
    except KeyboardInterrupt:
        print("Calibration interrupted; no incomplete result was saved.")
    finally:
        if teach_enabled:
            print(f"Restoring original UFACTORY mode ({restore_mode})...")
            try:
                node.set_controller(namespace, restore_mode)
            except Exception as exc:
                print(f"WARNING: could not restore mode {restore_mode}: {exc}")
                print(f"Run: ros2 service call {namespace}/set_mode "
                      f"xarm_msgs/srv/SetInt16 '{{data: {restore_mode}}}'")
        if node.cap is not None:
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        # SIGINT may already have shut the default context down. Calling it a
        # second time raises RCLError and used to hide the useful shutdown
        # diagnostics behind a traceback.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
