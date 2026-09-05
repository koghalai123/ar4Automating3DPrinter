import warnings
# scipy's gimbal lock warning is benign (rotation fine, euler decomposition
# non-unique at +/-90). filter here so every entry script inherits it.
warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)

from .aruco_detector import ArucoDetectionViewer
import rclpy
import numpy as np
import os
import json
import csv
import datetime
import functools
from pymoveit2 import GripperInterface
from .printerclass import BambuPrinter
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
import tf2_ros
from .simulated3DPrinter import Simulated3DPrinter
import time
import threading


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_LOG_DIR = os.path.join(_DATA_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


def _timed(method):
    """Log wall-clock duration + call chain for a public method."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        call_chain = " > ".join(self._timing_call_stack + [method.__name__])
        self._timing_call_stack.append(method.__name__)
        paused_at_start = self._timing_total_paused
        t0 = time.perf_counter()
        try:
            result = method(self, *args, **kwargs)
        finally:
            elapsed = round((time.perf_counter() - t0) - (self._timing_total_paused - paused_at_start), 4)
            self._timing_call_stack.pop()
            self._record_timing(call_chain, elapsed)
        return result
    return wrapper


class printerAutomation(ArucoDetectionViewer):
    def __init__(self, calibration_mode=False, stream_source="webcam", camera_index=None, camera_keyword="GENERAL WEBCAM",
                 color_topic=None, depth_topic=None, camera_info_topic=None,
                 feed_rotation_deg=0.0, marker_sizes=None, robot='ar4',
                 hand_eye_file=None):
        self._startup_start = time.perf_counter()
        # camera topic defaults resolve from the robot config in ArucoDetectionViewer
        super().__init__(source=stream_source,
                         camera_index=camera_index,
                         camera_keyword=camera_keyword,
                         color_topic=color_topic,
                         depth_topic=depth_topic,
                         camera_info_topic=camera_info_topic,
                         feed_rotation_deg=feed_rotation_deg,
                         marker_sizes=marker_sizes,
                         calibration_file=os.path.join(_REPO_ROOT, "calibration", "camera_matrix.npz"),
                         hand_eye_file=hand_eye_file,
                         robot=robot)
        self.get_logger().info(f"printerAutomation initialized, robot={robot}, calibration_mode={calibration_mode}")

        self.estimatedMarkerPrefix = "estimated_marker_"

        # gripper no-op switch, sim workaround
        self.gripper_disabled = False
        # add noise to estimated markers, for scan robustness testing
        self.randomize_estimated_markers = False
        # 10s observation window instead of 1s, for noise data collection
        self.collect_orientation_noise_data = False
        # extra scan passes re-aim at the fresh detection for a head-on measurement
        self.scan_passes = 1

        # Offset configs: per printer type, one waypoint list per procedure
        # (pickup/place/scrape) in the marker's local frame. Each procedure
        # runs exactly its own list, so every action is visible here.
        # Entry kinds:
        #   {'pos': [x,y,z], 'angle_deg': tilt about marker X (None = untilted)}
        #   {'scan': viewing_distance_m}   retried closer if the move fails
        #   {'gripper': 'open'|'close'}    close records joints for the replay
        # descriptions are for humans only. Keep pre-grasp and grasp aligned in
        # marker X/Y so the grasp move is pure marker Z.
        self.offset_configs = {
            # Printer with the handle above the marker
            'printer_offset_old_old': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.07, 0.15]), 'angle_deg': None},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.07, 0.077]), 'angle_deg': None},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.17, 0.077]), 'angle_deg': None},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.17, 0.077]), 'angle_deg': None},
                    {'description': 'descend back to the grasp pose',
                     'pos': np.array([0.0, 0.07, 0.077]), 'angle_deg': None},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.07, 0.15]), 'angle_deg': None},
                ],
                'scrape': None,
            },
            'printer_offset_old': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.082, 0.15]), 'angle_deg': -10.0},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.082, 0.047]), 'angle_deg': -10.0},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.19, 0.057]), 'angle_deg': -10.0},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.19, 0.057]), 'angle_deg': -10.0},
                    {'description': 'partial descent, tucked toward the marker',
                     'pos': np.array([0.0, 0.12, 0.057]), 'angle_deg': -10.0},
                    {'description': 'shift out along marker Z',
                     'pos': np.array([0.0, 0.12, 0.032]), 'angle_deg': -10.0},
                    {'description': 'set down',
                     'pos': np.array([0.0, 0.09, 0.032]), 'angle_deg': -10.0},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.082, 0.15]), 'angle_deg': -10.0},
                ],
                'scrape': None,
            },
            'printer_offset': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.06, 0.15]), 'angle_deg': 0.0},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.06, 0.1]), 'angle_deg': 0.0},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.14, 0.11]), 'angle_deg': 0.0},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.14, 0.11]), 'angle_deg': -10.0},
                    {'description': 'partial descent, tucked toward the marker',
                     'pos': np.array([0.0, 0.105, 0.11]), 'angle_deg': -10.0},
                    {'description': 'shift out along marker Z',
                     'pos': np.array([0.0, 0.105, 0.09]), 'angle_deg': -10.0},
                    {'description': 'set down',
                     'pos': np.array([0.0, 0.06, 0.09]), 'angle_deg': -10.0},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.06, 0.15]), 'angle_deg': 0.0},
                ],
                'scrape': None,
            },
            # Printer with the marker to the side; also the scrape fixture
            'box_offset': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.03, 0.15]), 'angle_deg': None},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.03, 0.102]), 'angle_deg': None},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.13, 0.102]), 'angle_deg': None},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.13, 0.102]), 'angle_deg': None},
                    {'description': 'descend back to the grasp pose',
                     'pos': np.array([0.0, 0.03, 0.102]), 'angle_deg': None},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.03, 0.15]), 'angle_deg': None},
                ],
                'scrape': [
                    {'description': 'scrape standoff along marker Z',
                     'pos': np.array([0.0, 0.092, 0.29]), 'angle_deg': 5.0},
                    {'description': 'full scrape depth',
                     'pos': np.array([0.0, 0.092, 0.13]), 'angle_deg': 5.0},
                    {'description': 'retract to standoff',
                     'pos': np.array([0.0, 0.092, 0.29]), 'angle_deg': 5.0},
                ],
            },
            'box_offset_old': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.05, 0.15]), 'angle_deg': None},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.05, 0.102]), 'angle_deg': None},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.15, 0.102]), 'angle_deg': None},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.15, 0.102]), 'angle_deg': None},
                    {'description': 'descend back to the grasp pose',
                     'pos': np.array([0.0, 0.05, 0.102]), 'angle_deg': None},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.05, 0.15]), 'angle_deg': None},
                ],
                'scrape': None,
            },
        }
        ## marker_id -> config name, unlisted ids fall back to 'box_offset'
        self.marker_offset_config = {}

        # tool orientation used to face a marker, and the camera's vertical
        # mount offset (m, base Z), both per robot
        self.offsetOri = self.robot_config['offset_ori']
        self.camera_z_offset = self.robot_config['camera_z_offset']

        # state file, saved every 5s, loaded at startup
        self._state_save_path = os.path.join(_DATA_DIR, "printer_state.json")
        self.create_timer(5.0, self._auto_save_state)

        # timing log, one row per public-method call
        _timing_dir = os.path.join(_DATA_DIR, "timing")
        os.makedirs(_timing_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timing_csv_path = os.path.join(_timing_dir, f"timing_{_ts}.csv")
        self._timing_file = open(self._timing_csv_path, "w", newline="")
        self._timing_writer = csv.writer(self._timing_file)
        self._timing_call_stack = []
        self._timing_total_paused = 0.0
        self._timing_pause_start = None
        self._timing_writer.writerow(["timestamp", "call_chain", "duration_s"])
        self._timing_file.flush()

        # raw scan log, one row per frame, truncated on restart
        self._scan_log_path = os.path.join(_LOG_DIR, "scan_raw_measurements.csv")
        self._scan_log_marker_id = None   # set by scanToMarker while active
        self._scan_log_distance = None
        self._scan_log_movement_id = 0     # bumped per observation window
        self._scan_log_file = open(self._scan_log_path, 'w', newline='')
        self._scan_log_writer = csv.writer(self._scan_log_file)
        self._scan_log_writer.writerow([
            'marker_id', 'scan_distance', 'movement_id',
            'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw',
            'cam_px', 'cam_py', 'cam_pz', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw',
        ])
        self._scan_log_file.flush()

        # marker_id -> BambuPrinter, filled by register_bambu_printer()
        self._bambu_printers: dict = {}

        # recorded pickup joints ({'marker','grasp','lift'}) so placePlate can
        # replay them instead of using flip-prone pose IK
        self._pickup_replay = None

        # Gripper interface (robots without one configured run gripper-disabled)
        gripper_cfg = self.robot_config['gripper']
        self._lite6_gripper_clients = {}
        if gripper_cfg is None:
            self.gripper = None
            self.gripper_disabled = True
            self.get_logger().info(
                f"No gripper configured for robot '{robot}'; gripper commands are skipped."
            )
        elif gripper_cfg.get('type') == 'lite6_service':
            try:
                from xarm_msgs.srv import Call
                namespace = gripper_cfg['namespace'].rstrip('/')
                self._lite6_gripper_clients = {
                    'open': self.create_client(
                        Call, f"{namespace}/open_lite6_gripper"),
                    'close': self.create_client(
                        Call, f"{namespace}/close_lite6_gripper"),
                }
                # Non-None marks a configured physical gripper for the web
                # backend's manipulation-hardware guard.
                self.gripper = 'lite6_service'
            except ImportError:
                self.gripper = None
                self.gripper_disabled = True
                self.get_logger().error(
                    "xarm_msgs unavailable; Lite 6 gripper commands disabled")
        else:
            self.gripper = GripperInterface(
                node=self,
                callback_group=self._cb_group,
                **gripper_cfg,
            )

    def _record_timing(self, call_chain: str, duration_s: float):
        """Append one timing row to the session CSV."""
        self._timing_writer.writerow(
            [datetime.datetime.now().isoformat(), call_chain, duration_s]
        )
        self._timing_file.flush()

    def record_startup_time(self):
        """Log time from __init__ to now as a 'startup' timing row."""
        elapsed = round(time.perf_counter() - self._startup_start, 4)
        self._record_timing("startup", elapsed)

    def pause_timing(self):
        """Pause the timing clock; paused time is excluded from active timers."""
        if self._timing_pause_start is None:
            self._timing_pause_start = time.perf_counter()

    def resume_timing(self):
        """Resume the timing clock."""
        if self._timing_pause_start is not None:
            self._timing_total_paused += time.perf_counter() - self._timing_pause_start
            self._timing_pause_start = None

    # ---- State persistence ----

    def save_state(self):
        """Dump marker poses, offset config, and printer configs to JSON."""
        data = {
            "marker_offset_config": {str(k): v for k, v in self.marker_offset_config.items()},
            "markers": [],
            "printers": getattr(self, '_saved_printer_configs', []),
        }
        for entry in self.stream.found_markers.values():
            if 'positionInBase' not in entry or 'eulerInBase' not in entry:
                continue
            data["markers"].append({
                "id": int(entry['id']),
                "positionInBase": entry['positionInBase'].tolist(),
                "eulerInBase": entry['eulerInBase'].tolist(),
                "dict_name": entry.get('dict_name', 'unknown'),
                "marker_size": float(entry.get('marker_size', 0.03)),
                "estimated": bool(entry.get('estimated', False)),
            })
        try:
            with open(self._state_save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save_state: could not write {self._state_save_path}: {e}")

    def register_printers(self, printers):
        """Store printer configs (dicts with marker_id/pos/orient/door_marker_texture) so save_state includes them."""
        self._saved_printer_configs = [
            {
                "marker_id": int(p["marker_id"]),
                "pos": list(p["pos"]),
                "orient": list(p["orient"]),
                "door_marker_texture": p["door_marker_texture"],
            }
            for p in printers
        ]

    def load_state(self):
        """Restore marker poses and offset config from the save file; markers
        come back as estimates so the next scan overwrites them."""
        if not os.path.exists(self._state_save_path):
            self.get_logger().info(f"load_state: no save file at {self._state_save_path} — starting fresh")
            return False
        try:
            with open(self._state_save_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().warn(f"load_state: could not read {self._state_save_path}: {e}")
            return False

        # json keys are strings
        for k, v in data.get("marker_offset_config", {}).items():
            self.marker_offset_config[int(k)] = v

        for m in data.get("markers", []):
            marker_id = int(m["id"])
            pos = np.array(m["positionInBase"], dtype=float)
            euler = np.array(m["eulerInBase"], dtype=float)
            self.register_estimated_marker(marker_id=marker_id, bad_pos=pos, bad_euler=euler)
            # keep real detections marked as real so they don't get replaced
            # by a geometric fallback later
            if not m.get("estimated", True):
                self.stream.found_markers[marker_id]['estimated'] = False

        if data.get("printers"):
            self._saved_printer_configs = data["printers"]

        n = len(data.get("markers", []))
        self.get_logger().info(
            f"load_state: restored {n} marker(s) and offset config from {self._state_save_path}"
        )
        return True

    def _auto_save_state(self):
        self.save_state()

    # ---- Raw measurement logging ----

    def _on_raw_marker_measurement(self, marker_id, pos_in_base, quat_in_base,
                                    pos_in_camera, quat_in_camera):
        """Per-frame raw detection hook; writes a CSV row immediately, but
        only while a scanToMarker observation window is active."""
        if self._scan_log_marker_id is None:
            return
        if marker_id != self._scan_log_marker_id:
            return
        row = [
            marker_id,
            round(self._scan_log_distance, 6) if self._scan_log_distance is not None else '',
            self._scan_log_movement_id,
            round(float(pos_in_base[0]), 6),
            round(float(pos_in_base[1]), 6),
            round(float(pos_in_base[2]), 6),
            round(float(quat_in_base[0]), 6),
            round(float(quat_in_base[1]), 6),
            round(float(quat_in_base[2]), 6),
            round(float(quat_in_base[3]), 6),
            round(float(pos_in_camera[0]), 6),
            round(float(pos_in_camera[1]), 6),
            round(float(pos_in_camera[2]), 6),
            round(float(quat_in_camera[0]), 6),
            round(float(quat_in_camera[1]), 6),
            round(float(quat_in_camera[2]), 6),
            round(float(quat_in_camera[3]), 6),
        ]
        self._scan_log_writer.writerow(row)
        self._scan_log_file.flush()

    # ---- Helpers ----

    def _find_marker_entry(self, marker_id):
        """Look up a marker by ID from marker_poses. Returns entry dict or None."""
        for m in self.marker_poses:
            if m['id'] == marker_id:
                return m
        return None

    def _broadcast_static_tf(self, bad_pos, bad_euler, child_frame):
        """Broadcast a static TF for a marker pose in base_link."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_link_name
        t.child_frame_id = child_frame
        t.transform.translation.x = float(bad_pos[0])
        t.transform.translation.y = float(bad_pos[1])
        t.transform.translation.z = float(bad_pos[2])
        q = R.from_euler("XYZ", bad_euler, degrees=False).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf2_static_broadcaster.sendTransform(t)

    def _apply_offset_in_marker_frame(self, marker_pos, marker_euler, offset_pos, offset_ori):
        """Apply an offset in the marker's local frame, return (pos, euler) in base_link."""
        R_marker = R.from_euler("XYZ", marker_euler, degrees=False)
        target_pos = marker_pos + R_marker.apply(offset_pos)
        target_euler = (R_marker * R.from_euler("XYZ", offset_ori, degrees=False)).as_euler("XYZ", degrees=False)
        return target_pos, target_euler

    def _tilted_offset_ori(self, angle_deg):
        """offsetOri tilted angle_deg about marker X (premultiplied, so the
        approach still runs along marker Z). None for zero angle."""
        if not angle_deg:
            return None
        tilt = R.from_euler("x", np.radians(angle_deg))
        return (tilt * R.from_euler("XYZ", self.offsetOri, degrees=False)).as_euler("XYZ", degrees=False)

    def _move_to_marker_offset(self, marker_id, offset_pos, offset_ori=None):
        """Find the marker, apply the offset, move there."""
        if offset_ori is None:
            offset_ori = self.offsetOri

        entry = self._find_marker_entry(marker_id)
        if entry is None:
            self.get_logger().warn(f"Marker ID {marker_id} not found in detected marker poses.")
            available_ids = [m['id'] for m in self.marker_poses]
            self.get_logger().info(f"Available marker IDs: {available_ids}")
            return False

        bad_pos = entry['positionInBase']
        bad_euler = entry['eulerInBase']

        badPos, badEuler = self._apply_offset_in_marker_frame(bad_pos, bad_euler, offset_pos, offset_ori)

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f'Moving to marker ID {marker_id} — marker centre: {bad_pos}, target: {badPos}')
        self.freeze_markers()
        move_ok = self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        return move_ok

    def _get_waypoints_for_marker(self, marker_id, procedure):
        """Waypoint list for 'pickup'/'place'/'scrape', or None."""
        config_name = self.marker_offset_config.get(marker_id, 'box_offset')
        waypoints = self.offset_configs[config_name].get(procedure)
        return waypoints if waypoints else None

    def _preflight_waypoints(self, marker_id, waypoints, caller):
        """Validate the complete routine geometry before its first motion.

        MoveIt still performs self/environment collision checking for every
        planned segment.  This earlier pass catches missing markers, bad
        calibration, NaN values and targets outside the commissioned software
        workspace before a routine can partially execute.
        """
        if not waypoints:
            raise RuntimeError(f"{caller}: routine has no configured waypoints")
        self.assert_motion_safe()
        entry = self._find_marker_entry(marker_id)
        if entry is None:
            raise RuntimeError(
                f"{caller}: marker {marker_id} has no registered pose")
        marker_pos = np.asarray(entry['positionInBase'], dtype=float)
        marker_euler = np.asarray(entry['eulerInBase'], dtype=float)
        if (marker_pos.shape != (3,) or marker_euler.shape != (3,)
                or not np.all(np.isfinite(marker_pos))
                or not np.all(np.isfinite(marker_euler))):
            raise RuntimeError(
                f"{caller}: marker {marker_id} pose is invalid")

        checked = 0
        for index, waypoint in enumerate(waypoints, start=1):
            if 'scan' in waypoint:
                offset = np.array([0.0, 0.0, float(waypoint['scan'])])
                bad_pos, bad_euler = self._apply_offset_in_marker_frame(
                    marker_pos, marker_euler, offset, self.offsetOri)
                bad_pos += np.array([0.0, 0.0, self.camera_z_offset])
            elif 'pos' in waypoint:
                orientation = (
                    self._tilted_offset_ori(waypoint.get('angle_deg'))
                    if waypoint.get('angle_deg') else self.offsetOri)
                bad_pos, bad_euler = self._apply_offset_in_marker_frame(
                    marker_pos, marker_euler,
                    np.asarray(waypoint['pos'], dtype=float), orientation)
            else:
                continue
            good_pos, _ = self.to_good_frame(bad_pos, bad_euler)
            try:
                self.validate_pose_target(good_pos)
            except ValueError as exc:
                raise RuntimeError(
                    f"{caller}: waypoint {index} failed workspace "
                    f"validation: {exc}") from exc
            checked += 1
        self.get_logger().info(
            f"{caller}: preflight passed for marker {marker_id} "
            f"({checked} Cartesian targets; MoveIt collision checks remain active)")
        return True

    def _follow_waypoints(self, markerID, waypoints, caller, tolerant_last=False):
        """Run the entries in order: moves, scans, gripper actions (a 'close'
        records joints for the pickup replay). False on first failure, except
        tolerant_last lets a failed final move (scrape retract) just warn."""
        self._preflight_waypoints(markerID, waypoints, caller)
        self._walk_grasp_joints = None
        n = len(waypoints)
        for i, wp in enumerate(waypoints):
            if 'scan' in wp:
                if not self._scan_with_retries(markerID, float(wp['scan']), caller):
                    return False
                continue
            if 'gripper' in wp:
                if wp['gripper'] == 'close':
                    self._walk_grasp_joints = self._current_arm_joints()
                    self.close_gripper()
                    time.sleep(3.0)
                else:
                    self.open_gripper()
                continue
            tilt_ori = self._tilted_offset_ori(wp.get('angle_deg'))
            pos = np.asarray(wp['pos'], dtype=float)
            if not self._move_to_marker_offset(markerID, pos, tilt_ori):
                if tolerant_last and i == n - 1:
                    self.get_logger().error(
                        f"{caller}: final waypoint {n}/{n} failed for marker {markerID}. Continuing."
                    )
                    return True
                self.get_logger().error(
                    f"{caller}: waypoint {i+1}/{n} failed for marker {markerID}."
                )
                return False
        return True

    # ---- BambuPrinter integration ----

    def register_bambu_printer(self, marker_id, printer: BambuPrinter):
        """Attach a connected BambuPrinter to a marker so transferPlate can
        move its head clear before pickup and home it after placing."""
        self._bambu_printers[marker_id] = printer
        self.get_logger().info(
            f"register_bambu_printer: marker {marker_id} → printer {printer.serial} at {printer.ip}"
        )
        printer.homing()

    # ---- Gripper ----

    def open_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping open.")
            return
        self.get_logger().info("Opening gripper...")
        if self.gripper == 'lite6_service':
            return self._call_lite6_gripper('open')
        self.gripper.open()
        return True

    def close_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping close.")
            return
        self.get_logger().info("Closing gripper...")
        if self.gripper == 'lite6_service':
            return self._call_lite6_gripper('close')
        self.gripper.close()
        return True

    def _call_lite6_gripper(self, action, timeout=5.0):
        """Call the Lite 6 controller gripper and require an explicit success."""
        from xarm_msgs.srv import Call
        client = self._lite6_gripper_clients[action]
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(
                f"Lite 6 gripper service unavailable for {action}")
        future = client.call_async(Call.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"Lite 6 gripper {action} timed out")
        response = future.result()
        if response is None or int(response.ret) != 0:
            ret = None if response is None else int(response.ret)
            message = '' if response is None else response.message
            raise RuntimeError(
                f"Lite 6 gripper {action} failed (ret={ret}): {message}")
        return True

    # ---- Marker updates ----

    def freeze_markers(self):
        """Disable marker pose updates. Call before moving the robot."""
        self.stream.marker_updates_enabled = False
        self.get_logger().info("Marker pose updates frozen.")

    def unfreeze_markers(self):
        """Re-enable marker pose updates. Call after the robot has stopped."""
        # let frames captured mid-move drain first (camera pipeline lag)
        time.sleep(0.5)
        self.stream.marker_updates_enabled = True
        self.get_logger().info("Marker pose updates resumed.")

    def lock_marker(self, marker_id):
        """Pin the marker to its current pose; camera detections can't move it.
        Used for fixed references like the scrape marker."""
        self.stream.locked_marker_ids.add(marker_id)
        entry = self._find_marker_entry(marker_id)
        if entry is not None:
            self.get_logger().info(
                f"Marker {marker_id} locked at pos={np.round(entry['positionInBase'], 4)} "
                f"euler_deg={np.round(np.degrees(entry['eulerInBase']), 2)} — camera updates ignored."
            )
        else:
            self.get_logger().warn(f"Marker {marker_id} locked, but no pose entry exists yet.")

    def unlock_marker(self, marker_id):
        """Allow camera detections to update marker_id again."""
        self.stream.locked_marker_ids.discard(marker_id)
        self.get_logger().info(f"Marker {marker_id} unlocked — camera updates allowed.")

    # ---- Marker registration & scanning ----

    def register_estimated_marker(self, marker_id, bad_pos, bad_euler):
        """Seed an estimated marker pose in TF and found_markers; the first
        real detection overwrites it."""
        # locked markers are fixed references, don't move them here either
        if marker_id in self.stream.locked_marker_ids:
            self.get_logger().info(
                f"register_estimated_marker: marker {marker_id} is locked — keeping file pose, skipping."
            )
            return
        bad_pos = np.array(bad_pos, dtype=float)
        bad_euler = np.array(bad_euler, dtype=float)
        if self.randomize_estimated_markers:
            rng = np.random.default_rng()
            random_dir = rng.normal(size=3)
            random_dir /= np.linalg.norm(random_dir)
            bad_pos = bad_pos + random_dir * 0.03
            random_ori_dir = rng.normal(size=3)
            random_ori_dir /= np.linalg.norm(random_ori_dir)
            bad_euler = bad_euler + random_ori_dir * 0.05
        tf2Name = f"{self.markerNamePrefix}{marker_id}"

        self._broadcast_static_tf(bad_pos, bad_euler, tf2Name)

        # Compute good-frame values for display
        R_BF_GF = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
        goodPos = R_BF_GF.apply(bad_pos)
        goodEuler = (R_BF_GF * R.from_euler("XYZ", bad_euler, degrees=False)).as_euler("XYZ", degrees=False)

        entry = {
            'id': marker_id,
            'tf2Name': tf2Name,
            'positionInBase': bad_pos,
            'eulerInBase': bad_euler,
            'positionInWorld': goodPos,
            'orientInWorld': {
                'roll': np.degrees(goodEuler[0]),
                'pitch': np.degrees(goodEuler[1]),
                'yaw': np.degrees(goodEuler[2]),
            },
            'positionFromCamera': np.array([0.0, 0.0, 0.0]),
            'eulerFromCamera': np.array([0.0, 0.0, 0.0]),
            'orientFromCamera': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'distanceFromCamera': 0.0,
            'estimated': True,
        }
        self.stream.found_markers[marker_id] = entry
        self.get_logger().info(
            f"Registered estimated marker {marker_id} at base_link pos={bad_pos}, euler={bad_euler}"
        )

    @_timed
    def scanToMarker(self, marker_id=0, viewing_distance=0.20):
        """Move the camera to face a known/estimated marker; extra passes
        re-aim at the refreshed pose so the measurement is head-on."""
        move_ok = False
        marker_spotted = False
        for scan_pass in range(max(1, 1)):
            entry = self._find_marker_entry(marker_id)
            if entry is None:
                self.get_logger().error(f"Marker {marker_id} not found in found_markers. Register it first.")
                # The public scan API always returns ``(move_ok, spotted)``.
                # Keeping that contract on an early lookup failure lets GUI
                # callers report the actual marker error instead of raising
                # "cannot unpack non-iterable bool object".
                return False, False

            offsetPos = np.array([0.0, 0.0, viewing_distance])
            badPos, badEuler = self._apply_offset_in_marker_frame(
                entry['positionInBase'], entry['eulerInBase'], offsetPos, self.offsetOri,
            )
            # raise in base Z so the camera (below the gripper) faces the marker
            badPos = badPos + np.array([0.0, 0.0, self.camera_z_offset])

            goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
            self.get_logger().info(
                f"Scanning marker {marker_id} (pass {scan_pass + 1}/{max(1, self.scan_passes)}): "
                f"moving to viewing pos={goodPos}"
            )
            self.freeze_markers()
            move_ok = self.move_to_pose(goodPos, goodEuler)
            self.unfreeze_markers()
            if not move_ok:
                break

            # Gazebo setups without a bridged RGBD sensor deliberately use
            # the registered geometric marker estimates. Treat reaching the
            # viewing pose as a successful simulated observation instead of
            # waiting four seconds for an image topic that does not exist.
            if getattr(self, "simulation_mode", False):
                marker_spotted = True
                break

            # raw-measurement logging is active only during this observation window
            self._scan_log_movement_id += 1
            self._scan_log_marker_id = marker_id
            self._scan_log_distance = viewing_distance
            # poll instead of a blind sleep: the first camera-pose commit takes
            # 1-2+ s after arrival (TF + detect latency), and a fixed sleep
            # intermittently loses that race; polling exits as soon as a real
            # detection lands, so fast detections cost nothing extra
            observation_pause = 10.0 if self.collect_orientation_noise_data else 4.0
            deadline = time.time() + observation_pause
            while time.time() < deadline:
                observed_entry = self._find_marker_entry(marker_id)
                if (not self.collect_orientation_noise_data
                        and observed_entry is not None
                        and not observed_entry.get('estimated', False)):
                    break
                time.sleep(0.25)
            self._scan_log_marker_id = None
            self._scan_log_distance = None

            observed_entry = self._find_marker_entry(marker_id)
            marker_spotted = observed_entry is not None and not observed_entry.get('estimated', False)
            if not marker_spotted:
                # nothing fresh to re-aim at
                break

        observed_entry = self._find_marker_entry(marker_id)
        if not move_ok:
            print(f"[SCAN] Marker {marker_id}: movement FAILED (pose unreachable).")
        elif not marker_spotted:
            print(f"[SCAN] Marker {marker_id}: NOT detected after moving to view position.")
        else:
            pos = observed_entry.get('positionInWorld', 'N/A')
            print(f"[SCAN] Marker {marker_id}: detected at {pos}")
        return move_ok, marker_spotted

    @_timed
    def scanLocationForMarkers(self, estimated_pos, estimated_orient=[0,0,0], viewing_distance=0.15, frame_name=None):
        """Move the camera to face an estimated marker location."""
        estimated_pos = np.array(estimated_pos)
        if frame_name is None:
            frame_name = f"{self.estimatedMarkerPrefix}0"

        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = np.array([0.0, 0.0, 0.0])

        markerBadPos, markerBadEuler = self.to_bad_frame(estimated_pos, estimated_orient)

        badPos, badEuler = self._apply_offset_in_marker_frame(
            markerBadPos, markerBadEuler, offsetPos, offsetOri,
        )
        # raise in base Z so the camera (below the gripper) faces the marker
        badPos = badPos + np.array([0.0, 0.0, self.camera_z_offset])

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f'Scanning for markers at estimated position: {estimated_pos}')
        self.freeze_markers()
        try:
            return bool(self.move_to_pose(goodPos, goodEuler))
        finally:
            self.unfreeze_markers()

    def scanMultipleLocations(self, locations, viewing_distance=0.15, pause_duration=2.0):
        """Scan multiple estimated marker locations sequentially."""
        for i, location in enumerate(locations):
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
                time.sleep(pause_duration)
                markers = self.marker_poses
                if markers:
                    self.get_logger().info(f"Detected {len(markers)} markers at location {i+1}")
                else:
                    self.get_logger().info(f"No markers detected at location {i+1}")

    # ---- Plate operations ----

    @_timed
    def moveToMarker(self, markerID=0):
        """Walk the marker's pickup waypoint list (scan/gripper entries included)."""
        waypoints = self._get_waypoints_for_marker(markerID, 'pickup')
        return self._follow_waypoints(markerID, waypoints, "moveToMarker")

    @_timed
    def pickupPlate(self, markerID=0):
        if not self.moveToMarker(markerID):
            self.get_logger().error(f"pickupPlate: moveToMarker failed for marker {markerID}.")
            return False
        # keep grasp + carry joints so placePlate can replay them
        grasp_joints = self._walk_grasp_joints
        lift_joints = self._current_arm_joints()
        if grasp_joints is not None and lift_joints is not None:
            self._pickup_replay = {'marker': markerID, 'grasp': grasp_joints,
                                   'lift': lift_joints}
        else:
            self._pickup_replay = None
        return True

    @_timed
    def placePlate(self, markerID=0):
        """Walk the marker's place list. If the pickup here was recorded, the
        moves before the release are replayed in joint space instead, pinning
        J6 so pose IK can't flip the plate into a collision (plain
        carry+descend only; longer descents are walked pose-based)."""
        place_waypoints = self._get_waypoints_for_marker(markerID, 'place')
        open_idx = next((i for i, wp in enumerate(place_waypoints)
                         if wp.get('gripper') == 'open'), None)
        pre_release = place_waypoints if open_idx is None else place_waypoints[:open_idx]
        replay = getattr(self, '_pickup_replay', None)
        custom_descent = sum(1 for wp in pre_release if 'pos' in wp) > 2
        if custom_descent and replay is not None:
            self.get_logger().warn(
                f"placePlate: custom placement descent configured for marker {markerID}; "
                "using pose-based placement instead of the wrist-continuous joint replay."
            )
            replay = None
        if replay is not None and replay.get('marker') == markerID:
            self.get_logger().info(
                f"placePlate: replaying recorded pickup joints for marker {markerID} (wrist-continuous)."
            )
            if not (self.move_to_configuration(replay['lift'])
                    and self.move_to_configuration(replay['grasp'])):
                # no pose-based fallback: IK could pick a flipped wrist and
                # drop the plate at an angle. Abort still holding it.
                self.get_logger().error(
                    "placePlate: wrist-continuous joint replay failed; aborting WITHOUT placing "
                    "to avoid an angled/flipped drop. Plate is still held — recover and re-run."
                )
                return False
            self.open_gripper()
            # Continue with the entries after the release (e.g. the withdraw).
            if open_idx is not None and open_idx + 1 < len(place_waypoints):
                return self._follow_waypoints(
                    markerID, place_waypoints[open_idx + 1:], "placePlate"
                )
            return True

        if not self._follow_waypoints(markerID, place_waypoints, "placePlate"):
            return False
        if open_idx is None:
            # Config without an explicit release entry — release at the end.
            self.open_gripper()
        return True

    @_timed
    def _scan_with_retries(self, marker_id, scan_distance, caller):
        """scanToMarker, retried at 0.85x and 0.70x distance if the move fails."""
        for factor in (1.0, 0.85, 0.70):
            move_ok, _ = self.scanToMarker(
                marker_id=marker_id, viewing_distance=factor * scan_distance
            )
            if move_ok:
                return True
        self.get_logger().error(
            f"{caller}: could not reach marker {marker_id}. Aborting."
        )
        return False

    def transferPlate(self, source_id, dest_id, rescan_id):
        """Pick up from source, place at dest, pick up from rescan, place back
        at source. All motion (scans included) comes from the offset-config
        waypoint lists; aborts on the first failed step."""
        self.get_logger().info(
            f"transferPlate: source={source_id}, dest={dest_id}, rescan={rescan_id}"
        )

        # Step 1 – pick up from source
        self.get_logger().info(f"Step 1: picking up plate from marker {source_id}")
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False

        # Step 2 – place at destination
        self.get_logger().info(f"Step 2: placing plate at marker {dest_id}")
        if not self.placePlate(markerID=dest_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {dest_id}. Aborting."
            )
            return False
        _p = self._bambu_printers.get(dest_id)
        if _p:
            _p.home()

        # Step 3 – pick up from rescan printer
        self.get_logger().info(f"Step 3: picking up plate from marker {rescan_id}")
        _p = self._bambu_printers.get(rescan_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=rescan_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {rescan_id}. Aborting."
            )
            return False

        # Step 4 – place back at source
        self.get_logger().info(f"Step 4: placing plate at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.home()

        self.get_logger().info("transferPlate: sequence complete.")
        return True

    @_timed
    def scanMarkerApproach(self, marker_id, viewing_distance=0.15):
        """Scan at progressively closer distances (1.75x down to 1.0x).
        Bails out if the marker isn't seen at the first, longest distance."""
        distances = [
            (1.75 * viewing_distance, 2.0),
            (1.50 * viewing_distance, 1.0),
            (1.25 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
        ]

        for i, (dist, pause) in enumerate(distances):
            move_ok, spotted = self.scanToMarker(marker_id=marker_id, viewing_distance=dist)
            time.sleep(pause)
            if i == 0 and not spotted:
                # the farthest pose is fragile on short-reach arms (lite6):
                # estimate noise can make it unreachable or off-frame. Retry
                # once closer before giving up on the whole approach.
                fallback = 1.4 * viewing_distance
                self.get_logger().warn(
                    f"scanMarkerApproach: marker {marker_id} not seen at max distance "
                    f"({dist:.3f} m) — retrying closer at {fallback:.3f} m."
                )
                move_ok, spotted = self.scanToMarker(marker_id=marker_id, viewing_distance=fallback)
                time.sleep(pause)
                if not spotted:
                    self.get_logger().warn(
                        f"scanMarkerApproach: marker {marker_id} not seen at fallback distance "
                        f"({fallback:.3f} m) either — aborting approach."
                    )
                    return False

        return True

    def _scrape_dbg(self, msg):
        """Timestamped line to the scrape debug log. Instrumentation only."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.get_logger().info(f"DBG {msg}")
        try:
            with open(os.path.join(_LOG_DIR, "scrape_debug.log"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _current_arm_joints(self):
        """Return the 6 arm joint angles (rad) in self.moveit2.joint_names order, or None."""
        js = self.moveit2.joint_state
        if js is None:
            return None
        names = list(js.name)
        try:
            return [float(js.position[names.index(j)]) for j in self.moveit2.joint_names]
        except ValueError:
            return None

    @_timed
    def scrapePlate(self, source_id, scrape_id, wait_after_pickup=False, wait_duration=60.0, rotate_after_scrape=False, rotate_degrees=60.0):
        """Pick up from source_id, scrape against the scrape_id surface, put
        it back. wait_after_pickup delays before scraping (cooldown);
        rotate_after_scrape rolls the wrist after the retract."""
        scrape_waypoints = self._get_waypoints_for_marker(scrape_id, 'scrape')
        if not scrape_waypoints:
            self.get_logger().error(
                f"scrapePlate: no 'scrape' waypoints configured for marker {scrape_id}'s "
                "offset config. Aborting."
            )
            return False
        self.get_logger().info(
            f"scrapePlate: source={source_id}, scrape={scrape_id}, "
            f"{len(scrape_waypoints)} scrape waypoint(s)"
        )

        # Step 1 – pick up plate from source (its pickup list scans the marker first)
        self.get_logger().info(f"Step 1: picking up plate from marker {source_id}")
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False
        if wait_after_pickup:
            self.get_logger().info(
                f"scrapePlate: waiting {wait_duration} s after pickup before scraping."
            )
            time.sleep(wait_duration)

        # freeze so a close-range sighting can't corrupt the scrape marker pose
        # (it's normally also locked to its file pose, see runScrapePlate.py)
        self.freeze_markers()

        # log the marker pose the waypoints get applied to
        _e4 = self._find_marker_entry(scrape_id)
        if _e4 is not None:
            self._scrape_dbg(
                f"SCRAPE marker {scrape_id} pos={np.round(_e4['positionInBase'],4)} "
                f"euler_deg={np.round(np.degrees(_e4['eulerInBase']),2)} estimated={_e4.get('estimated')}"
            )

        # Step 2 – scrape (a failed final retract only warns so the plate
        # can still be returned)
        self.get_logger().info(f"Step 2: walking {len(scrape_waypoints)} scrape waypoint(s)")
        if not self._follow_waypoints(scrape_id, scrape_waypoints, "scrapePlate",
                                      tolerant_last=True):
            self.get_logger().error(
                f"scrapePlate: scrape waypoints failed for marker {scrape_id}. Aborting."
            )
            return False
        self._scrape_dbg(
            "SCRAPE walk done joints_deg=" +
            str(np.round(np.degrees(self._current_arm_joints() or []), 1).tolist())
        )

        # Step 2b – optional end-effector rotation to dislodge debris / change plate orientation
        if rotate_after_scrape:
            self.get_logger().info(
                f"scrapePlate: rotating end-effector joint by {rotate_degrees:.1f}° after scrape."
            )
            js = self.moveit2.joint_state
            if js is not None:
                joint_names_list = list(js.name)
                current_joints = [
                    float(js.position[joint_names_list.index(j)])
                    for j in self.moveit2.joint_names
                ]
                rotated_joints = list(current_joints)
                rotated_joints[-1] -= np.radians(rotate_degrees)

                # J6 angle vs its limit (+/-180 mk3, +/-155 mk2)
                j6_now = np.degrees(current_joints[-1])
                j6_tgt = np.degrees(rotated_joints[-1])
                self._scrape_dbg(
                    "ROTATE current_joints_deg=" + str(np.round(np.degrees(current_joints), 1).tolist())
                )
                self._scrape_dbg(
                    f"ROTATE j6 {j6_now:.1f} deg --({-rotate_degrees:.0f})--> target {j6_tgt:.1f} deg; "
                    f"exceeds_155={abs(j6_tgt) > 155.0} exceeds_180={abs(j6_tgt) > 180.0}"
                )

                rot_ok = self.move_to_configuration(rotated_joints)
                self._scrape_dbg(f"ROTATE move_to(rotated) ok={rot_ok}")
                time.sleep(0.5)
                # restore the wrist angle before placing
                res_ok = self.move_to_configuration(current_joints)
                self._scrape_dbg(f"ROTATE move_to(restore) ok={res_ok} "
                                 f"joints_after={np.round(np.degrees(self._current_arm_joints() or []),1).tolist()}")
                time.sleep(0.5)
                if not rot_ok:
                    self.get_logger().warn(
                        "scrapePlate: post-scrape rotation failed. Continuing to place "
                        "(placePlate replay will return the wrist to the grasp config)."
                    )
            else:
                self.get_logger().warn(
                    "scrapePlate: joint state unavailable — skipping rotation."
                )

        # Step 3 – place back at source; unfreeze first so the camera can
        # refresh the source marker pose
        self.unfreeze_markers()
        self.get_logger().info(f"Step 3: placing plate back at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False

        self.get_logger().info("scrapePlate: sequence complete.")
        return True

    @_timed
    def go_home(self, velocity_scaling=0.2):
        """Move all joints to zero. Plans from the actual encoder state, so it
        corrects drift from lost steps; kept slow to avoid further stalls."""
        self.get_logger().warn(
            f"go_home: resyncing to home position (velocity_scaling={velocity_scaling}). "
            "Planning from actual encoder state to correct any step-loss drift."
        )
        # settle before reading the pose
        time.sleep(0.5)

        home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        prev_velocity = self.moveit2.max_velocity
        prev_acceleration = self.moveit2.max_acceleration
        try:
            self.moveit2.max_velocity = velocity_scaling
            self.moveit2.max_acceleration = velocity_scaling
            self.freeze_markers()
            self.move_to_configuration(home_joints)
            time.sleep(self.move_settle_delay)
        finally:
            self.moveit2.max_velocity = prev_velocity
            self.moveit2.max_acceleration = prev_acceleration
            self.unfreeze_markers()

        self.get_logger().info("go_home: reached home position.")
