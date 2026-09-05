"""Persistent, GUI-friendly ArUco and ChArUco commissioning workflow."""

from __future__ import annotations

import json
import pathlib
import shutil
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.CameraCalibration import (
    ARUCO_DICT, MARKER_LENGTH, SQUARE_LENGTH, SQUARES_X, SQUARES_Y)
from calibration.charuco_utils import (
    calibrate_camera, create_board, detect, detector_parameters, estimate_pose)


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
        # Intrinsic samples intentionally live only for the current session.
        # They contain OpenCV arrays and should not be mixed across cameras or
        # resolutions by accidentally resuming an old browser session.
        self.intrinsic_samples = []
        self.intrinsic_session_active = False
        self.last_intrinsic = None
        self.last_intrinsic_detection = {
            "valid": False, "corner_count": 0, "error": None}
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
            # Hand-eye needs the pose associated with the *measured* joints.
            # TF can lag while the physical xArm is in teach mode, leaving the
            # backend to reject every post-first sample as a duplicate.
            joints, joint_source = self.node._calibration_joint_positions()
            position, quaternion = self.node._eef_pose_from_joint_state(joints)
            if position is None:
                raise RuntimeError("base-to-gripper FK is unavailable")
            # A hand-eye pair is only meaningful when image and robot pose are
            # sampled from a stationary arm.  This also prevents captures
            # while the operator is still guiding the wrist.
            time.sleep(0.15)
            check_joints, check_source = self.node._calibration_joint_positions()
            check_position, check_quaternion = (
                self.node._eef_pose_from_joint_state(check_joints))
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
                previous_sample = self.samples[-1]
                previous = np.asarray(
                    previous_sample["base_to_gripper"]["matrix"])
                translation = np.linalg.norm(
                    base_to_gripper[:3, 3] - previous[:3, 3])
                rotation = Rotation.from_matrix(
                    previous[:3, :3].T @ base_to_gripper[:3, :3]).magnitude()
                if translation < 0.01 and rotation < np.radians(5):
                    current_joints = np.asarray(check_joints or [], dtype=float)
                    previous_joints = np.asarray(
                        previous_sample.get("joint_positions") or [],
                        dtype=float)
                    max_joint_delta = None
                    if len(current_joints) == len(previous_joints) and \
                            len(current_joints):
                        wrapped = (current_joints - previous_joints + np.pi) % \
                                  (2 * np.pi) - np.pi
                        max_joint_delta = float(np.degrees(np.max(np.abs(wrapped))))
                    joint_detail = ("unavailable" if max_joint_delta is None
                                    else f"{max_joint_delta:.2f} deg")
                    raise RuntimeError(
                        "pose is too similar to the previous sample; move at "
                        "least 10 mm or rotate the wrist at least 5 degrees "
                        f"(FK delta: {translation * 1000:.1f} mm, "
                        f"{np.degrees(rotation):.2f} deg; measured max joint "
                        f"delta: {joint_detail})")
            sample = {
                "timestamp": time.time(),
                "joint_positions": [float(x) for x in
                                    (check_joints or [])],
                "joint_source": check_source or joint_source,
                "base_to_gripper": {"matrix": base_to_gripper.tolist()},
                "camera_to_charuco": {"matrix": camera_to_board.tolist()},
                "detected_corner_count": self.last_detection["corner_count"],
            }
            self.samples.append(sample)
            self._save()
            return {"sample_count": len(self.samples), "sample": sample}

    # ---- Camera intrinsics -------------------------------------------------

    def start_intrinsic(self, clear=False):
        """Start a short, camera-only ChArUco calibration session.

        This is deliberately independent of hand-eye samples: intrinsics need
        many different *views of the board in the image*, while hand-eye also
        needs corresponding robot poses.  Keeping the two sets separate makes
        it impossible to feed unsuitable frames into the hand-eye solver.
        """
        with self.lock:
            if clear:
                self.intrinsic_samples = []
            self.intrinsic_session_active = True
            return self.snapshot()

    def stop_intrinsic(self):
        with self.lock:
            self.intrinsic_session_active = False
            return self.snapshot()

    def capture_intrinsic(self):
        with self.lock:
            if not self.intrinsic_session_active:
                raise RuntimeError("start an intrinsic calibration session first")
            frame = self._raw_frame()
            if frame is None:
                raise RuntimeError("no camera frame is available")
            stream = self.node.stream
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mc, mi, _, cc, ci = detect(
                gray, self.board, self.dictionary, self.parameters)
            corner_count = 0 if cc is None else len(cc)
            marker_count = 0 if mi is None else len(mi)
            valid = marker_count >= 4 and corner_count >= 20
            self.last_intrinsic_detection = {
                "valid": valid, "corner_count": int(corner_count),
                "marker_count": int(marker_count),
                "error": None if valid else
                "show at least 4 markers and 20 ChArUco corners",
            }
            if not valid:
                raise RuntimeError(self.last_intrinsic_detection["error"])
            size = (int(gray.shape[1]), int(gray.shape[0]))
            if self.intrinsic_samples and size != self.intrinsic_samples[0]["size"]:
                raise RuntimeError(
                    "camera resolution changed; start over before capturing")
            self.intrinsic_samples.append({
                "corners": np.asarray(cc, dtype=np.float32).copy(),
                "ids": np.asarray(ci, dtype=np.int32).copy(),
                "size": size,
                "timestamp": time.time(),
            })
            return {"sample_count": len(self.intrinsic_samples),
                    "corner_count": int(corner_count), "size": size}

    def discard_intrinsic(self):
        with self.lock:
            if not self.intrinsic_samples:
                raise RuntimeError("there are no intrinsic calibration captures")
            self.intrinsic_samples.pop()
            return {"sample_count": len(self.intrinsic_samples)}

    def solve_intrinsic(self):
        """Solve and atomically apply a calibrated camera model.

        We retain the old .npz as ``camera_matrix.previous.npz`` and only
        replace the active calibration when RMS is reasonably low.  A bad
        capture sequence therefore cannot silently degrade later ArUco poses.
        """
        with self.lock:
            if len(self.intrinsic_samples) < 12:
                raise RuntimeError("at least 12 intrinsic captures are required")
            corners = [sample["corners"] for sample in self.intrinsic_samples]
            ids = [sample["ids"] for sample in self.intrinsic_samples]
            size = self.intrinsic_samples[0]["size"]
            flags = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
            rms, matrix, distortion, _rvecs, _tvecs = calibrate_camera(
                corners, ids, self.board, size, flags=flags)
            # 1 px is a conservative acceptance threshold for an ordinary USB
            # webcam.  A result above it remains visible to the operator but
            # does not replace the known calibration.
            quality = bool(np.isfinite(rms) and rms <= 1.0)
            result = {
                "sample_count": len(self.intrinsic_samples),
                "image_size": list(size),
                "rms_px": float(rms),
                "quality_passed": quality,
                "max_rms_px": 1.0,
                "camera_index": getattr(self.node.stream, "camera_index", None),
            }
            self.last_intrinsic = result
            if not quality:
                return result

            path = pathlib.Path(getattr(self.node.stream, "calibration_file", "")
                                or self.root / "calibration" / "camera_matrix.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = path.with_name("camera_matrix.previous.npz")
            if path.is_file():
                shutil.copy2(path, backup)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, camera_matrix=matrix, dist_coeffs=distortion,
                     image_size=np.asarray(size),
                     camera_index=-1 if result["camera_index"] is None
                                  else result["camera_index"],
                     rms_px=float(rms))
            temporary.replace(path)
            self.node.stream.set_calibration(matrix, distortion)
            result["path"] = str(path)
            result["backup_path"] = str(backup) if backup.is_file() else None
            return result

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
            "intrinsic": {
                "session_active": self.intrinsic_session_active,
                "sample_count": len(self.intrinsic_samples),
                "recommended_sample_count": 20,
                "last_detection": dict(self.last_intrinsic_detection),
                "last_solve": self.last_intrinsic,
            },
        }
