"""Persistent, GUI-friendly ArUco and ChArUco commissioning workflow."""

from __future__ import annotations

import json
import pathlib
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.CameraCalibration import (
    ARUCO_DICT, MARKER_LENGTH, SQUARE_LENGTH, SQUARES_X, SQUARES_Y)
from calibration.charuco_utils import (
    create_board, detect, detector_parameters, estimate_pose)


def _matrix(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


class VisionCommissioning:
    """Own calibration samples and named observation poses for one robot.

    The class deliberately does not command motion.  RosRobotBackend applies
    the normal safety interlocks before replaying a saved observation pose.
    """

    def __init__(self, node, robot, repo_root):
        self.node = node
        self.robot = robot
        self.root = pathlib.Path(repo_root)
        self.state_path = self.root / "data" / "vision_commissioning.json"
        self.calibration_path = (
            self.root / "calibration" / f"{robot}_hand_eye.json")
        self.lock = threading.RLock()
        self.samples = []
        self.observation_poses = []
        self.marker_roles = {}
        self.last_solve = None
        self.session_active = False
        self.last_detection = {
            "valid": False, "corner_count": 0, "error": None}
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.dictionary = dictionary
        self.board = create_board(
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
        self.parameters = detector_parameters()
        self._load()

    def _load(self):
        if not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text())
            self.samples = list(data.get("hand_eye_samples", []))
            self.observation_poses = list(data.get("observation_poses", []))
            self.marker_roles = dict(data.get("marker_roles", {}))
            self.last_solve = data.get("last_solve")
        except Exception as exc:
            self.last_detection["error"] = f"could not load session: {exc}"

    def _save(self):
        data = {
            "schema_version": 1,
            "robot": self.robot,
            "updated_at": time.time(),
            "hand_eye_samples": self.samples,
            "observation_poses": self.observation_poses,
            "marker_roles": self.marker_roles,
            "last_solve": self.last_solve,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(self.state_path)

    def start(self, clear=False):
        with self.lock:
            if clear:
                self.samples = []
            self.session_active = True
            self._save()
            return self.snapshot()

    def stop(self):
        with self.lock:
            self.session_active = False
            return self.snapshot()

    def _raw_frame(self):
        stream = self.node.stream
        with stream.lock:
            return None if stream.raw_frame is None else stream.raw_frame.copy()

    def inspect_board(self, frame=None, draw=False):
        stream = self.node.stream
        frame = self._raw_frame() if frame is None else frame.copy()
        if frame is None:
            raise RuntimeError("no camera frame is available")
        if stream.camera_matrix is None or stream.dist_coeffs is None:
            raise RuntimeError("camera intrinsic calibration is unavailable")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mc, mi, _, cc, ci = detect(
            gray, self.board, self.dictionary, self.parameters)
        corners = 0 if cc is None else len(cc)
        pose = None
        rvec = tvec = None
        if cc is not None and corners >= 12:
            ok, rvec, tvec = estimate_pose(
                cc, ci, self.board, stream.camera_matrix,
                stream.dist_coeffs)
            if ok:
                pose = np.eye(4)
                pose[:3, :3] = cv2.Rodrigues(rvec)[0]
                pose[:3, 3] = np.asarray(tvec).reshape(3)
        if draw:
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(frame, mc, mi)
            if cc is not None:
                cv2.aruco.drawDetectedCornersCharuco(frame, cc, ci)
            if pose is not None:
                cv2.drawFrameAxes(frame, stream.camera_matrix,
                                  stream.dist_coeffs, rvec, tvec, 0.04)
        self.last_detection = {
            "valid": pose is not None,
            "corner_count": int(corners),
            "error": None if pose is not None else
                     "at least 12 ChArUco corners are required",
        }
        return pose, frame

    def preview(self, fallback):
        """Update ChArUco health without hiding the normal ArUco overlay.

        ``fallback`` is the detector's annotated camera frame.  Replacing it
        with a raw ChArUco frame during a session used to make the colored
        ArUco axes disappear exactly when an operator wanted to register a
        workspace marker.
        """
        if not self.session_active:
            return fallback
        try:
            self.inspect_board(draw=False)
        except Exception as exc:
            self.last_detection = {
                "valid": False, "corner_count": 0, "error": str(exc)}
        return fallback

    def capture_sample(self):
        with self.lock:
            if not self.session_active:
                raise RuntimeError("start a calibration session first")
            camera_to_board, _ = self.inspect_board(draw=False)
            if camera_to_board is None:
                raise RuntimeError(self.last_detection["error"])
            position, quaternion = self.node._eef_pose_truth()
            if position is None:
                raise RuntimeError("base-to-gripper TF is unavailable")
            # A hand-eye pair is only meaningful when image and robot pose are
            # sampled from a stationary arm.  This also prevents captures
            # while the operator is still guiding the wrist.
            time.sleep(0.15)
            check_position, check_quaternion = self.node._eef_pose_truth()
            if check_position is None:
                raise RuntimeError("base-to-gripper TF became unavailable")
            moved = np.linalg.norm(check_position - position)
            turned = (Rotation.from_quat(quaternion).inv() *
                      Rotation.from_quat(check_quaternion)).magnitude()
            if moved > 0.001 or turned > np.radians(1.0):
                raise RuntimeError(
                    "robot is still moving; hold it stationary and capture again")
            position, quaternion = check_position, check_quaternion
            base_to_gripper = _matrix(position, quaternion)
            if self.samples:
                previous = np.asarray(
                    self.samples[-1]["base_to_gripper"]["matrix"])
                translation = np.linalg.norm(
                    base_to_gripper[:3, 3] - previous[:3, 3])
                rotation = Rotation.from_matrix(
                    previous[:3, :3].T @ base_to_gripper[:3, :3]).magnitude()
                if translation < 0.01 and rotation < np.radians(5):
                    raise RuntimeError(
                        "pose is too similar to the previous sample; move at "
                        "least 10 mm or rotate the wrist at least 5 degrees")
            sample = {
                "timestamp": time.time(),
                "joint_positions": [float(x) for x in
                                    (self.node._last_joint_msg
                                     if self.node._last_joint_msg is not None
                                     else [])],
                "base_to_gripper": {"matrix": base_to_gripper.tolist()},
                "camera_to_charuco": {"matrix": camera_to_board.tolist()},
                "detected_corner_count": self.last_detection["corner_count"],
            }
            self.samples.append(sample)
            self._save()
            return {"sample_count": len(self.samples), "sample": sample}

    def discard_last(self):
        with self.lock:
            if not self.samples:
                raise RuntimeError("there are no calibration samples")
            discarded = self.samples.pop()
            self._save()
            return {"sample_count": len(self.samples),
                    "discarded": discarded}

    def solve(self):
        # Import lazily so normal robot startup does not pull calibration-only
        # ROS message dependencies into mock/server tests.
        from calibration.handEyeCalibration import solve_hand_eye
        with self.lock:
            robot_poses = [np.asarray(
                x["base_to_gripper"]["matrix"], dtype=float)
                for x in self.samples]
            board_poses = [np.asarray(
                x["camera_to_charuco"]["matrix"], dtype=float)
                for x in self.samples]
            _, method, tf, metrics = solve_hand_eye(robot_poses, board_poses)
            rotation = Rotation.from_matrix(tf[:3, :3])
            quality = (metrics["translation_rmse_m"] <= 0.01 and
                       metrics["rotation_rmse_deg"] <= 3.0)
            result = {
                "schema_version": 1,
                "robot": self.robot,
                "parent_frame": self.node.end_effector_name,
                "child_frame": self.node.stream.camera_frame_id,
                "method": method,
                "sample_count": len(self.samples),
                "translation_m": tf[:3, 3].tolist(),
                "quaternion_xyzw": rotation.as_quat().tolist(),
                "rpy_rad": rotation.as_euler("xyz").tolist(),
                "matrix": tf.tolist(),
                "metrics": metrics,
                "quality_passed": quality,
                "board": {
                    "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                    "square_length_m": SQUARE_LENGTH,
                    "marker_length_m": MARKER_LENGTH,
                    "dictionary": "DICT_4X4_50",
                },
            }
            # Keep rejected results too.  They are valuable feedback for the
            # operator (usually insufficient pose diversity), while only a
            # passing result is allowed to become the active calibration.
            self.last_solve = result
            self._save()
            if quality:
                self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
                self.calibration_path.write_text(
                    json.dumps(result, indent=2) + "\n")
                # Apply the accepted transform immediately.  Previously a
                # newly generated file was only consumed on the next backend
                # restart, which made a successful browser calibration look
                # ineffective until the operator restarted the whole GUI.
                self.node.hand_eye_transform = tf
                self.node.hand_eye_file = str(self.calibration_path)
            return result

    def save_observation(self, name, marker_id=None, role="other"):
        name = str(name).strip()
        if not name:
            raise ValueError("observation pose name is required")
        position, quaternion = self.node._eef_pose_truth()
        if position is None:
            raise RuntimeError("current end-effector TF is unavailable")
        good_position, good_euler = self.node.to_good_frame(
            position, Rotation.from_quat(quaternion).as_euler("XYZ"))
        item = {
            "name": name,
            "marker_id": None if marker_id is None else int(marker_id),
            "role": str(role),
            "timestamp": time.time(),
            "robot": self.robot,
            "joint_positions": [float(x) for x in
                                (self.node._last_joint_msg
                                 if self.node._last_joint_msg is not None
                                 else [])],
            "eef_pose": {
                "position": np.asarray(position).tolist(),
                "quaternion_xyzw": np.asarray(quaternion).tolist(),
                "good_position": np.asarray(good_position).tolist(),
                "good_euler": np.asarray(good_euler).tolist(),
                "frame": self.node.base_link_name,
            },
        }
        with self.lock:
            self.observation_poses = [
                x for x in self.observation_poses if x.get("name") != name]
            self.observation_poses.append(item)
            self._save()
        return item

    def observation(self, name):
        with self.lock:
            for item in self.observation_poses:
                if item.get("name") == name:
                    return item
        raise KeyError(f"unknown observation pose '{name}'")

    def confirm_marker(self, marker_id, role="other"):
        marker_id = int(marker_id)
        marker = next((m for m in self.node.marker_poses
                       if int(m.get("id", -1)) == marker_id), None)
        if marker is None or marker.get("estimated", False):
            raise RuntimeError(
                f"ArUco {marker_id} has not been measured by the camera")
        self.marker_roles[str(marker_id)] = str(role)
        self._save()
        self.node.save_state()
        return {"marker_id": marker_id, "role": role, "saved": True}

    def snapshot(self):
        calibration = None
        if self.calibration_path.is_file():
            try:
                calibration = json.loads(self.calibration_path.read_text())
            except Exception:
                calibration = {"quality_passed": False, "error": "invalid file"}
        return {
            "session_active": self.session_active,
            "sample_count": len(self.samples),
            "recommended_sample_count": 15,
            "last_detection": dict(self.last_detection),
            "observation_poses": list(self.observation_poses),
            "marker_roles": dict(self.marker_roles),
            "hand_eye": calibration,
            "last_solve": self.last_solve,
            "state_path": str(self.state_path),
            "calibration_path": str(self.calibration_path),
        }
