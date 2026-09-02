import os
# Silence OpenCV's noisy V4L2/GStreamer probe warnings (must be set before cv2 import).
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
from flask import Flask, Response
import threading
import logging
import time
import subprocess
import re
import fcntl
import struct
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def create_aruco_detector_parameters():
    """Create ArUco parameters across OpenCV's pre/post-4.7 APIs."""
    factory = getattr(cv2.aruco, "DetectorParameters_create", None)
    if factory is not None:
        return factory()
    return cv2.aruco.DetectorParameters()


def detect_aruco_markers(image, dictionary, parameters):
    """Detect markers with either the legacy or OpenCV 4.7+ API."""
    legacy = getattr(cv2.aruco, "detectMarkers", None)
    if legacy is not None:
        return legacy(image, dictionary, parameters=parameters)
    return cv2.aruco.ArucoDetector(
        dictionary, parameters).detectMarkers(image)


def estimate_single_marker_pose(corner, marker_size, camera_matrix,
                                distortion):
    """Replacement for estimatePoseSingleMarkers removed in OpenCV 4.11."""
    legacy = getattr(cv2.aruco, "estimatePoseSingleMarkers", None)
    if legacy is not None:
        return legacy(corner, marker_size, camera_matrix, distortion)
    half = float(marker_size) / 2.0
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float32)
    image_points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("OpenCV could not estimate ArUco pose")
    return rvec.reshape(1, 1, 3), tvec.reshape(1, 1, 3), object_points


# ------------------------------------------------------------------
# Camera discovery
# ------------------------------------------------------------------

def _is_capture_device(index):
    """True if /dev/video{index} supports capture. UVC webcams expose extra
    metadata-only nodes that spew warnings when opened; skip them. On any
    error return True so the node still gets probed normally."""
    V4L2_CAP_VIDEO_CAPTURE = 0x00000001
    V4L2_CAP_DEVICE_CAPS   = 0x80000000
    VIDIOC_QUERYCAP = (2 << 30) | (104 << 16) | (ord('V') << 8) | 0  # _IOR('V', 0, v4l2_capability)
    try:
        fd = os.open(f"/dev/video{index}", os.O_RDONLY | os.O_NONBLOCK)
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
            caps, device_caps = struct.unpack_from("<II", buf, 84)  # capabilities, device_caps
            effective = device_caps if (caps & V4L2_CAP_DEVICE_CAPS) else caps
            return bool(effective & V4L2_CAP_VIDEO_CAPTURE)
        finally:
            os.close(fd)
    except Exception:
        return True  # can't tell -> let the normal open decide


def select_camera(preset_keyword=None):
    """List cameras and auto-select by keyword or prompt the user."""
    cameras = []
    result = subprocess.run(['ls', '/sys/class/video4linux/'], capture_output=True, text=True)
    for device in sorted(result.stdout.strip().split()):
        index = int(re.search(r'\d+', device).group())
        if not _is_capture_device(index):
            continue  # skip metadata-only nodes
        with open(f'/sys/class/video4linux/{device}/name', 'r') as f:
            name = f.read().strip()
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append((index, name))
            cap.release()

    if not cameras:
        print("No cameras found!")
        return None

    if preset_keyword is not None:
        for idx, name in cameras:
            if preset_keyword.lower() in name.lower():
                print(f"Auto-selected camera [{idx}] '{name}'")
                return idx

    print("\nAvailable cameras:")
    for idx, name in cameras:
        print(f"  [{idx}] {name}")
    while True:
        choice = int(input("Select camera index: "))
        if choice in [c[0] for c in cameras]:
            return choice


# ------------------------------------------------------------------
# Depth visualisation
# ------------------------------------------------------------------

def visualize_depth(depth: np.ndarray, colormap_name: str = "turbo") -> np.ndarray:
    cmap_map = {'turbo': cv2.COLORMAP_TURBO, 'jet': cv2.COLORMAP_JET,
                'inferno': cv2.COLORMAP_INFERNO}
    cmap = cmap_map.get(colormap_name.lower(), cv2.COLORMAP_TURBO)

    d = depth.astype(np.float32) if depth.dtype == np.uint16 else depth.copy().astype(np.float32)
    mask = np.isfinite(d) & (d > 0)
    if not np.any(mask):
        return np.zeros((*d.shape[:2], 3), dtype=np.uint8)
    lo = float(np.percentile(d[mask], 1.0))
    hi = float(np.percentile(d[mask], 99.0))
    if hi <= lo:
        hi = lo + 1e-3
    d_norm = np.clip((d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cmap)


# ------------------------------------------------------------------
# ArUco dict lookup
# ------------------------------------------------------------------

ARUCO_DICT_MAP = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250, 'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50, 'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250, 'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50, 'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250, 'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50, 'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250, 'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}


# ------------------------------------------------------------------
# Single unified class
# ------------------------------------------------------------------

class WebVideoStream:
    """MJPEG web server + ArUco detection over a webcam or ROS source.
    Markers persist in found_markers (by ID); see the marker_poses property."""

    def __init__(self,
                 source="webcam",
                 port=5000,
                 fps=30.0,
                 display_scale=1.0,
                 depth_colormap="turbo",
                 enable_aruco=True,
                 marker_sizes=None,
                 dict_names=None,
                 enrich_fn=None,
                 log_fn=None,
                 camera_index=None,
                 camera_keyword="GENERAL WEBCAM",
                 calibration_file=None,
                 color_topic="/rgbd_camera/image",
                 depth_topic="/rgbd_camera/depth_image",
                 camera_info_topic="/rgbd_camera/camera_info",
                 feed_rotation_deg=0.0):

        self.source = source.lower()
        # feed rotation (deg), webcam source only
        self.feed_rotation_deg = float(feed_rotation_deg) if self.source == "webcam" else 0.0
        self.fps = fps
        self.display_scale = display_scale
        self.depth_colormap = depth_colormap
        self.enrich_fn = enrich_fn
        self.log_fn = log_fn
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.started_monotonic = time.monotonic()
        self.last_color_monotonic = None
        self.last_depth_monotonic = None
        self.last_camera_info_monotonic = None
        self.last_detection_monotonic = None
        self.camera_frame_id = None
        self.camera_index = camera_index
        self.last_error = None
        self.calibration_file = calibration_file
        self.cap = None
        self._webcam_thread = None

        # Permanent dict of all markers ever found: {marker_id: entry_dict}
        # This is NEVER cleared. Once a marker is seen, it stays here forever.
        self.found_markers = {}
        # When False, found_markers will not be updated with new detections.
        self.marker_updates_enabled = True
        # marker IDs camera detections may never overwrite (fixed references
        # from the save file); takes precedence over marker_updates_enabled
        self.locked_marker_ids = set()
        # Per-marker EMA state used exclusively for the web panel display.
        # Does NOT affect found_markers or anything returned by detect().
        self._display_filter_states = {}

        # ---- ArUco detector ----
        self.enable_aruco = enable_aruco
        if enable_aruco:
            self.dict_names = dict_names or ['DICT_4X4_50', 'DICT_6X6_50']
            self.marker_sizes = marker_sizes or [0.03, 0.025]
            self.aruco_dicts = [cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[n])
                                for n in self.dict_names]
            self.aruco_params = create_aruco_detector_parameters()
            self.camera_matrix = None
            self.dist_coeffs = None
            self.last_marker_count = 0

        # ---- Web server ----
        self.frame = None
        # Unannotated camera image used by browser-driven calibration.  Keep
        # this separate from ``frame``: the latter contains ArUco drawings and
        # the diagnostics panel and must never be fed back into ChArUco pose
        # estimation.
        self.raw_frame = None
        self.frame_id = 0
        # Condition (not Event): every MJPEG client waits on the same lock but
        # notify_all wakes all of them. A shared Event let whichever client
        # cleared it first swallow the wakeup for the others, dropping them to
        # the 1 s poll timeout — a ~1 fps feed that looks frozen.
        self.lock = threading.Condition()

        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.app = Flask(__name__)

        @self.app.route('/')
        def index():
            return ('<html><body style="margin:0;background:#000;">'
                    '<img src="/video_feed" style="width:100%;height:auto;">'
                    '</body></html>')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        threading.Thread(
            target=self.app.run,
            kwargs={'host': '0.0.0.0', 'port': port, 'threaded': True},
            daemon=True,
        ).start()
        print(f"Video stream at http://localhost:{port}")

        # ---- Source-specific setup ----
        if self.source == "webcam":
            if camera_index is None:
                camera_index = select_camera(preset_keyword=camera_keyword)
            assert camera_index is not None, "No camera selected"
            self.cap = cv2.VideoCapture(camera_index)
            self.camera_index = camera_index
            assert self.cap.isOpened(), f"Could not open camera {camera_index}"
            # Do not synchronously read here. Some UVC nodes open correctly
            # but block their first read indefinitely; that used to hold the
            # whole robot backend in state=starting. The capture thread owns
            # all reads and reports failures through diagnostics.
            print(f"Camera opened (index {camera_index})")
            if enable_aruco:
                import pathlib
                _default = pathlib.Path(__file__).resolve().parents[1] / "calibration" / "camera_matrix.npz"
                _cal_path = pathlib.Path(calibration_file) if calibration_file is not None else _default
                if not _cal_path.exists():
                    raise RuntimeError(
                        f"Camera calibration file not found at '{_cal_path}'. "
                        "Run CameraCalibration.py first to generate 'camera_matrix.npz'.")
                _data = np.load(_cal_path)
                self.camera_matrix = _data["camera_matrix"]
                self.dist_coeffs = _data["dist_coeffs"]
                print(f"Loaded calibration from '{_cal_path}'")

        else:  # ros
            self._rclpy = rclpy

            outer = self

            class _Node(Node):
                def __init__(self):
                    super().__init__('web_video_server')
                    self.bridge = CvBridge()
                    self.latest_color = None
                    self.latest_depth = None
                    # Image intake and the heavy _tick render MUST live in
                    # separate callback groups. In one (default) mutually
                    # exclusive group the executor runs a single callback at a
                    # time, so a _tick doing full ArUco detection on 1280x720
                    # blocks _color_cb for longer than the frame period —
                    # latest_color goes stale and the web feed re-renders the
                    # same image. Separate groups let intake keep running while
                    # a render is in flight (depth 1 then does its intended job
                    # of dropping stale frames rather than masking the stall).
                    self._intake_group = MutuallyExclusiveCallbackGroup()
                    self._render_group = MutuallyExclusiveCallbackGroup()
                    self.create_subscription(Image, color_topic, self._color_cb, 1,
                                             callback_group=self._intake_group)
                    self.create_subscription(Image, depth_topic, self._depth_cb, 1,
                                             callback_group=self._intake_group)
                    self.create_subscription(CameraInfo, camera_info_topic, self._info_cb, 10,
                                             callback_group=self._intake_group)
                    # guards against _tick piling up when a render overruns the
                    # timer period (the render group serialises, but the
                    # executor would still queue every missed tick)
                    self._rendering = False
                    self.create_timer(1.0 / outer.fps, self._tick,
                                      callback_group=self._render_group)

                def _color_cb(self, msg):
                    try:
                        self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                        outer.last_color_monotonic = time.monotonic()
                        if msg.header.frame_id:
                            outer.camera_frame_id = msg.header.frame_id.lstrip('/')
                        outer.last_error = None
                    except Exception as exc:
                        outer.last_error = f"color conversion failed: {exc}"

                def _depth_cb(self, msg):
                    try:
                        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                        outer.last_depth_monotonic = time.monotonic()
                        outer.last_error = None
                    except Exception as exc:
                        outer.last_error = f"depth conversion failed: {exc}"

                def _info_cb(self, msg):
                    outer.last_camera_info_monotonic = time.monotonic()
                    if msg.header.frame_id:
                        outer.camera_frame_id = msg.header.frame_id.lstrip('/')
                    if outer.enable_aruco and outer.camera_matrix is None:
                        outer.camera_matrix = np.array(msg.k).reshape(3, 3)
                        outer.dist_coeffs = np.array(msg.d)

                def _tick(self):
                    if self.latest_color is None or self._rendering:
                        return
                    self._rendering = True
                    try:
                        outer._build_frame(self.latest_color, self.latest_depth)
                    finally:
                        self._rendering = False

            self._ros_node = _Node()

    # ---- Calibration ----

    def set_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def configure_webcam(self, camera_index: int):
        """Switch the live capture device without rebuilding the ROS node."""
        old = self.cap
        if old is not None:
            old.release()
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            self.cap = None
            self.last_error = f"could not open camera index {camera_index}"
            raise RuntimeError(self.last_error)
        self.cap = cap
        self.camera_index = int(camera_index)
        self.source = 'webcam'
        if self.enable_aruco:
            import pathlib
            default = (pathlib.Path(__file__).resolve().parents[1] /
                       'calibration' / 'camera_matrix.npz')
            path = pathlib.Path(self.calibration_file) if self.calibration_file else default
            if not path.is_file():
                cap.release()
                self.cap = None
                raise RuntimeError(f"camera calibration not found: {path}")
            data = np.load(path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
        self.last_error = None
        if self._webcam_thread is None or not self._webcam_thread.is_alive():
            self._webcam_thread = threading.Thread(target=self.run, daemon=True)
            self._webcam_thread.start()

    def disable_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.source = 'disabled'
        self.camera_index = None
        self.last_color_monotonic = None
        self.camera_frame_id = None

    def _set_default_calibration(self, width: int, height: int):
        f = max(width, height)
        self.camera_matrix = np.array([
            [f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

    @property
    def is_calibrated(self) -> bool:
        return self.enable_aruco and self.camera_matrix is not None

    # ---- Public marker access ----

    @property
    def marker_poses(self):
        """All markers ever found, as a list. Never empty once a marker has been seen."""
        return list(self.found_markers.values())

    def diagnostics(self) -> dict:
        """Small JSON-safe health report for the GUI/status endpoint."""
        now = time.monotonic()

        def age(value):
            return None if value is None else max(0.0, now - value)

        return {
            'source': self.source,
            'color_topic': self.color_topic if self.source == 'ros' else None,
            'depth_topic': self.depth_topic if self.source == 'ros' else None,
            'camera_info_topic': (
                self.camera_info_topic if self.source == 'ros' else None),
            'camera_frame': self.camera_frame_id,
            'camera_index': self.camera_index if self.source == 'webcam' else None,
            'calibrated': bool(self.is_calibrated),
            'color_age_s': age(self.last_color_monotonic),
            'depth_age_s': age(self.last_depth_monotonic),
            'camera_info_age_s': age(self.last_camera_info_monotonic),
            'detection_age_s': age(self.last_detection_monotonic),
            'visible_marker_count': int(self.last_marker_count),
            'known_marker_count': len(self.found_markers),
            'marker_dictionaries': list(getattr(self, 'dict_names', [])),
            'marker_sizes_m': [float(v) for v in getattr(self, 'marker_sizes', [])],
            'last_error': self.last_error,
        }

    # ---- MJPEG stream ----

    def _update_frame(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()
            self.frame_id += 1
            self.lock.notify_all()

    def _generate(self):
        last_id = -1
        while True:
            with self.lock:
                # wait_for re-checks under the lock, so a frame published
                # between the predicate check and the wait cannot be missed
                got = self.lock.wait_for(
                    lambda: self.frame is not None and self.frame_id != last_id,
                    timeout=1.0)
                if not got:
                    continue
                frame = self.frame.copy()
                last_id = self.frame_id
            # encode outside the lock: JPEG on the full composite is slow and
            # would otherwise stall _update_frame (and thus the render tick)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # ---- ArUco detection ----

    def detect(self, image: np.ndarray) -> tuple:
        _log = self.log_fn or print
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        all_corners, all_ids, dict_indices = [], [], []
        for idx, aruco_dict in enumerate(self.aruco_dicts):
            corners, ids, _ = detect_aruco_markers(
                gray, aruco_dict, self.aruco_params)
            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids)
                dict_indices.extend([idx] * len(ids))

        output = image.copy()
        live_marker_poses = []

        if all_ids:
            all_ids_np = np.array(all_ids).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(output, all_corners, all_ids_np)

            if not self.is_calibrated:
                return output, live_marker_poses

            h_img, w_img = image.shape[:2]
            f_naive = float(max(w_img, h_img))
            naive_K = np.array([[f_naive, 0, w_img / 2.0],
                                 [0, f_naive, h_img / 2.0],
                                 [0, 0, 1.0]], dtype=np.float64)
            naive_d = np.zeros(5, dtype=np.float64)

            seen = set()
            for i, corner in enumerate(all_corners):
                marker_id = all_ids[i][0]
                if marker_id in seen:
                    continue
                seen.add(marker_id)

                marker_size = self.marker_sizes[dict_indices[i]]
                rvec, tvec, _ = estimate_single_marker_pose(
                    corner, marker_size, self.camera_matrix, self.dist_coeffs)

                position_cam = tvec[0][0].copy()
                distance = float(np.linalg.norm(position_cam))
                rot_mat, _ = cv2.Rodrigues(rvec[0])
                roll, pitch, yaw = R.from_matrix(rot_mat).as_euler('XYZ', degrees=False)

                rvec_n, tvec_n, _ = estimate_single_marker_pose(
                    corner, marker_size, naive_K, naive_d)
                pos_naive = tvec_n[0][0].copy()
                dist_naive = float(np.linalg.norm(pos_naive))
                rot_naive, _ = cv2.Rodrigues(rvec_n[0])
                roll_n, pitch_n, yaw_n = R.from_matrix(rot_naive).as_euler('XYZ', degrees=False)

                entry = {
                    'id': marker_id,
                    'dict_name': self.dict_names[dict_indices[i]],
                    'camera_frame': self.camera_frame_id,
                    'positionFromCamera': position_cam,
                    'eulerFromCamera': np.array([roll, pitch, yaw]),
                    'orientFromCamera': {
                        'roll': np.degrees(roll),
                        'pitch': np.degrees(pitch),
                        'yaw': np.degrees(yaw),
                    },
                    'distanceFromCamera': distance,
                    'positionFromCamera_naive': pos_naive,
                    'distanceFromCamera_naive': dist_naive,
                    'orientFromCamera_naive': {
                        'roll': np.degrees(roll_n),
                        'pitch': np.degrees(pitch_n),
                        'yaw': np.degrees(yaw_n),
                    },
                }

                if self.marker_updates_enabled and marker_id not in self.locked_marker_ids:
                    if self.enrich_fn is not None:
                        enriched = self.enrich_fn(entry)
                        if enriched is not None:
                            # Only commit a fully enriched entry (has positionInBase /
                            # eulerInBase). This keeps found_markers usable downstream.
                            entry = enriched
                            self.found_markers[marker_id] = entry
                        else:
                            # enrichment failed (TF not ready); keep the old
                            # pose, a camera-only entry would break scanToMarker
                            _log = self.log_fn or print
                            _log(f"[detect] ID={marker_id} enrich_fn returned None — keeping previous pose")
                    else:
                        # No enrichment configured: store the raw camera entry as-is.
                        self.found_markers[marker_id] = entry.copy()
                live_marker_poses.append(entry)
                self.last_detection_monotonic = time.monotonic()

                cv2.drawFrameAxes(output, self.camera_matrix, self.dist_coeffs,
                                  rvec[0], tvec[0], marker_size * 0.5)
                centre = corner[0].mean(axis=0).astype(int)
                cv2.putText(output, f"ID:{marker_id} D:{distance:.3f}m",
                            tuple(centre + np.array([0, -10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            count = len(all_ids)
            if count != self.last_marker_count:
                self.last_marker_count = count
                _log(f'Detected {count} ArUco marker(s): {[id_[0] for id_ in all_ids]}')

            # found_markers is already updated above per-marker

        else:
            if self.last_marker_count > 0:
                _log('No ArUco markers detected')
                self.last_marker_count = 0

        return output, live_marker_poses

    # ---- Display-only low-pass filter ----

    def _smooth_for_display(self, marker_list: list) -> list:
        """Return a display copy of marker_list with poses smoothed by a 4-frame
        time-constant first-order IIR filter (same approach as ArucoDetector.py).
        Never modifies found_markers or any value returned by detect()."""
        dt = 1.0 / self.fps
        RC = 4.0 / self.fps        # 4-frame time constant: RC = 4 * dt
        alpha = dt / (RC + dt)     # = 0.2

        result = []
        for entry in marker_list:
            mid = entry['id']
            p = dict(entry)        # shallow copy, arrays replaced below

            if mid not in self._display_filter_states:
                self._display_filter_states[mid] = {}
            state = self._display_filter_states[mid]

            def _smooth(pos_key, orient_dict_key, dist_key, prefix):
                if pos_key not in p:
                    return
                pos_new = np.asarray(p[pos_key], dtype=float)
                od = p.get(orient_dict_key) or {}
                euler_new = np.radians([od.get('roll', 0.0),
                                        od.get('pitch', 0.0),
                                        od.get('yaw', 0.0)])
                sp, sq = prefix + '_pos', prefix + '_quat'
                if sp not in state:
                    state[sp] = pos_new.copy()
                    state[sq] = R.from_euler('XYZ', euler_new).as_quat()
                else:
                    state[sp] = alpha * pos_new + (1.0 - alpha) * state[sp]
                    q_new = R.from_euler('XYZ', euler_new).as_quat()
                    q_prev = state[sq]
                    if np.dot(q_prev, q_new) < 0.0:
                        q_new = -q_new
                    q_smooth = R.from_quat(q_prev) * R.from_rotvec(
                        alpha * (R.from_quat(q_prev).inv() * R.from_quat(q_new)).as_rotvec())
                    state[sq] = q_smooth.as_quat()
                p[pos_key] = state[sp].copy()
                if dist_key and dist_key in p:
                    p[dist_key] = float(np.linalg.norm(state[sp]))
                smooth_euler = R.from_quat(state[sq]).as_euler('XYZ', degrees=False)
                p[orient_dict_key] = {
                    'roll':  float(np.degrees(smooth_euler[0])),
                    'pitch': float(np.degrees(smooth_euler[1])),
                    'yaw':   float(np.degrees(smooth_euler[2])),
                }

            _smooth('positionFromCamera',       'orientFromCamera',       'distanceFromCamera',       'cam')
            _smooth('positionFromCamera_naive',  'orientFromCamera_naive', 'distanceFromCamera_naive', 'naive')
            _smooth('positionInWorld',           'orientInWorld',          None,                       'world')
            result.append(p)
        return result

    # ---- Pose panel ----

    def _draw_pose_panel(self, width: int, marker_list: list) -> np.ndarray:
        line_h = 14
        panel = np.zeros((max(160, 40 + len(marker_list) * 210), width, 3), dtype=np.uint8)
        y = 18

        if not marker_list:
            cv2.putText(panel, "No markers detected", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return panel

        for p in marker_list:
            pos, ori, dist = p['positionFromCamera'], p['orientFromCamera'], p['distanceFromCamera']
            cv2.putText(panel, f"Marker {p['id']} (Camera->Marker):", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2); y += line_h
            cv2.putText(panel, f"  Pos: X={pos[0]:+.3f} Y={pos[1]:+.3f} Z={pos[2]:+.3f}  Dist={dist:.3f}m",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); y += line_h
            cv2.putText(panel, f"  RPY: R={ori['roll']:+.1f} P={ori['pitch']:+.1f} Y={ori['yaw']:+.1f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); y += line_h

            if 'positionInWorld' in p:
                pb, ob = p['positionInWorld'], p['orientInWorld']
                cv2.putText(panel, f"  (Base->Marker):", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2); y += line_h
                cv2.putText(panel, f"    Pos: X={pb[0]:+.3f} Y={pb[1]:+.3f} Z={pb[2]:+.3f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); y += line_h
                cv2.putText(panel, f"    RPY: R={ob['roll']:+.1f} P={ob['pitch']:+.1f} Y={ob['yaw']:+.1f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); y += line_h
            if 'positionFromCamera_naive' in p:
                pn = p['positionFromCamera_naive']
                on_n = p['orientFromCamera_naive']
                dn = p['distanceFromCamera_naive']
                cv2.putText(panel, f"  (Naive est.):", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 130, 255), 2); y += line_h
                cv2.putText(panel, f"    Pos: X={pn[0]:+.3f} Y={pn[1]:+.3f} Z={pn[2]:+.3f}  Dist={dn:.3f}m",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1); y += line_h
                cv2.putText(panel, f"    RPY: R={on_n['roll']:+.1f} P={on_n['pitch']:+.1f} Y={on_n['yaw']:+.1f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1); y += line_h
            y += 2

        return panel

    # ---- Composite frame ----

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame by self.feed_rotation_deg around its centre."""
        if self.feed_rotation_deg == 0.0:
            return frame
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), self.feed_rotation_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

    def _build_frame(self, color, depth=None):
        with self.lock:
            self.raw_frame = color.copy()
        frame = color.copy()
        live_marker_poses = []
        if self.enable_aruco:
            frame, live_marker_poses = self.detect(frame)
        # Apply feed rotation AFTER detection so pose estimation uses the
        # original camera frame (rotation only affects the web stream display)
        frame = self._rotate_frame(frame)

        if depth is not None:
            depth_raw = depth
            if depth_raw.shape[:2] != frame.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (frame.shape[1], frame.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            depth_vis = visualize_depth(depth_raw, self.depth_colormap)
            if len(depth_vis.shape) == 2:
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            frame = np.hstack([frame, depth_vis])

        if self.enable_aruco:
            # Show ALL known markers in the panel (persistent), smoothed for display only
            frame = np.vstack([frame, self._draw_pose_panel(frame.shape[1], self._smooth_for_display(self.marker_poses))])

        if self.display_scale and self.display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.display_scale, fy=self.display_scale,
                               interpolation=cv2.INTER_LINEAR)

        self._update_frame(frame)
        return live_marker_poses

    # ---- Blocking run ----

    def run(self):
        if self.source in {"webcam", "disabled"}:
            if self._webcam_thread is None:
                self._webcam_thread = threading.current_thread()
            dt = 1.0 / self.fps
            while True:
                try:
                    cap = self.cap
                    if cap is None:
                        time.sleep(0.1)
                        continue
                    ret, frame = cap.read()
                    if not ret:
                        self.last_error = (
                            f"camera index {self.camera_index} returned no frame")
                        time.sleep(0.1)
                        continue
                    self.last_color_monotonic = time.monotonic()
                    self.camera_frame_id = 'usb_camera_optical_frame'
                    self.last_error = None
                    self._build_frame(frame)
                except Exception as e:
                    self.last_error = f'frame processing failed: {e}'
                    # Camera preview is a diagnostic/safety aid in its own
                    # right. Preserve the unprocessed image if ArUco drawing
                    # or panel composition fails, while reporting the error so
                    # vision-dependent commands remain fail-closed.
                    if 'frame' in locals() and frame is not None:
                        self._update_frame(frame)
                    _log = self.log_fn or print
                    _log(f'Webcam frame error (recovering): {e}')
                time.sleep(dt)
            
        else:
            self._rclpy.spin(self._ros_node)
            


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    stream = WebVideoStream(
        source="webcam",
        port=5000,
        fps=30.0,
        display_scale=1.0/0.702,
        depth_colormap="turbo",
        marker_sizes=[0.03, 0.025],
        dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
        camera_keyword="GENERAL WEBCAM",
    )
    stream.run()
