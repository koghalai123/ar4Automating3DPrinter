import cv2
import numpy as np
from flask import Flask, Response
import threading
import logging
import time
import subprocess
import re
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ------------------------------------------------------------------
# Camera discovery
# ------------------------------------------------------------------

def select_camera(preset_keyword=None):
    """List cameras and auto-select by keyword or prompt the user."""
    cameras = []
    result = subprocess.run(['ls', '/sys/class/video4linux/'], capture_output=True, text=True)
    for device in sorted(result.stdout.strip().split()):
        index = int(re.search(r'\d+', device).group())
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
    """
    All-in-one: MJPEG web server + ArUco detection + webcam or ROS source.

    Usage:
        stream = WebVideoStream(source="ros", ...)
        stream.run()   # blocking
    """

    def __init__(self,
                 # Source
                 source="ros",                          # "webcam" or "ros"
                 # Web server
                 port=5000,
                 # Display
                 fps=30.0,
                 display_scale=1.0,
                 depth_colormap="turbo",
                 # ArUco
                 enable_aruco=True,
                 marker_sizes=None,
                 dict_names=None,
                 enrich_fn=None,
                 log_fn=None,
                 # Webcam
                 camera_index=None,
                 camera_keyword="GENERAL WEBCAM",
                 # ROS topics
                 color_topic="/rgbd_camera/image",
                 depth_topic="/rgbd_camera/depth_image",
                 camera_info_topic="/rgbd_camera/camera_info"):

        self.source = source.lower()
        self.fps = fps
        self.display_scale = display_scale
        self.depth_colormap = depth_colormap
        self.enrich_fn = enrich_fn
        self.log_fn = log_fn
        self.marker_poses = []

        # ---- ArUco detector ----
        self.enable_aruco = enable_aruco
        if enable_aruco:
            self.dict_names = dict_names or ['DICT_4X4_50', 'DICT_6X6_50']
            self.marker_sizes = marker_sizes or [0.03, 0.05]
            self.aruco_dicts = [cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[n])
                                for n in self.dict_names]
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.camera_matrix = None
            self.dist_coeffs = None
            self.last_marker_count = 0

        # ---- Web server ----
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

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
            assert self.cap.isOpened(), f"Could not open camera {camera_index}"
            ret, frame = self.cap.read()
            assert ret, "Could not read initial frame"
            h, w = frame.shape[:2]
            print(f"Camera: {w}x{h} (index {camera_index})")
            if enable_aruco and self.camera_matrix is None:
                self._set_default_calibration(w, h)

        else:  # ros
            

            self._rclpy = rclpy
            

            outer = self

            class _Node(Node):
                def __init__(self):
                    super().__init__('web_video_server')
                    self.bridge = CvBridge()
                    self.latest_color = None
                    self.latest_depth = None
                    self.create_subscription(Image, color_topic, self._color_cb, 10)
                    self.create_subscription(Image, depth_topic, self._depth_cb, 10)
                    self.create_subscription(CameraInfo, camera_info_topic, self._info_cb, 10)
                    self.create_timer(1.0 / outer.fps, self._tick)

                def _color_cb(self, msg):
                    self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

                def _depth_cb(self, msg):
                    self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

                def _info_cb(self, msg):
                    if outer.enable_aruco and outer.camera_matrix is None:
                        outer.camera_matrix = np.array(msg.k).reshape(3, 3)
                        outer.dist_coeffs = np.array(msg.d)

                def _tick(self):
                    if self.latest_color is None:
                        return
                    outer.marker_poses = outer._build_frame(
                        self.latest_color, self.latest_depth)

            self._ros_node = _Node()

    # ---- Calibration ----

    def set_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def _set_default_calibration(self, width: int, height: int):
        f = max(width, height)
        self.camera_matrix = np.array([
            [f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

    @property
    def is_calibrated(self) -> bool:
        return self.enable_aruco and self.camera_matrix is not None

    # ---- MJPEG stream ----

    def _update_frame(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()
            self.frame_id += 1
        self.new_frame_event.set()

    def _generate(self):
        last_id = -1
        while True:
            self.new_frame_event.wait(timeout=1.0)
            self.new_frame_event.clear()
            with self.lock:
                if self.frame is None or self.frame_id == last_id:
                    continue
                frame = self.frame.copy()
                last_id = self.frame_id
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # ---- ArUco detection ----

    def detect(self, image: np.ndarray) -> tuple:
        _log = self.log_fn or print
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        all_corners, all_ids, dict_indices = [], [], []
        for idx, aruco_dict in enumerate(self.aruco_dicts):
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids)
                dict_indices.extend([idx] * len(ids))

        output = image.copy()
        marker_poses = []

        if all_ids:
            all_ids_np = np.array(all_ids).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(output, all_corners, all_ids_np)

            if not self.is_calibrated:
                return output, marker_poses

            seen = set()
            for i, corner in enumerate(all_corners):
                marker_id = all_ids[i][0]
                if marker_id in seen:
                    continue
                seen.add(marker_id)

                marker_size = self.marker_sizes[dict_indices[i]]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, marker_size, self.camera_matrix, self.dist_coeffs)

                position_cam = tvec[0][0]
                distance = float(np.linalg.norm(position_cam))
                rot_mat, _ = cv2.Rodrigues(rvec[0])
                roll, pitch, yaw = R.from_matrix(rot_mat).as_euler('XYZ', degrees=False)

                entry = {
                    'id': marker_id,
                    'positionFromCamera': position_cam,
                    'eulerFromCamera': np.array([roll, pitch, yaw]),
                    'orientFromCamera': {
                        'roll': np.degrees(roll),
                        'pitch': np.degrees(pitch),
                        'yaw': np.degrees(yaw),
                    },
                    'distanceFromCamera': distance,
                }

                if self.enrich_fn is not None:
                    entry = self.enrich_fn(entry)
                if entry is None:
                    continue
                marker_poses.append(entry)

                cv2.drawFrameAxes(output, self.camera_matrix, self.dist_coeffs,
                                  rvec[0], tvec[0], marker_size * 0.5)
                centre = corner[0].mean(axis=0).astype(int)
                cv2.putText(output, f"ID:{marker_id} D:{distance:.2f}m",
                            tuple(centre + np.array([0, -10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            count = len(all_ids)
            if count != self.last_marker_count:
                self.last_marker_count = count
                _log(f'Detected {count} ArUco marker(s): {[id_[0] for id_ in all_ids]}')
        else:
            if self.last_marker_count > 0:
                _log('No ArUco markers detected')
                self.last_marker_count = 0

        return output, marker_poses

    # ---- Pose panel ----

    def _draw_pose_panel(self, width: int, marker_poses: list) -> np.ndarray:
        line_h = 14
        panel = np.zeros((max(160, 40 + len(marker_poses) * 160), width, 3), dtype=np.uint8)
        y = 18

        if not marker_poses:
            cv2.putText(panel, "No markers detected", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return panel

        for p in marker_poses:
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
            y += 2

        return panel

    # ---- Composite frame ----

    def _build_frame(self, color, depth=None):
        frame = color.copy()
        marker_poses = []

        if self.enable_aruco:
            frame, marker_poses = self.detect(frame)

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
            frame = np.vstack([frame, self._draw_pose_panel(frame.shape[1], marker_poses)])

        if self.display_scale and self.display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.display_scale, fy=self.display_scale,
                               interpolation=cv2.INTER_LINEAR)

        self._update_frame(frame)
        return marker_poses

    # ---- Blocking run ----

    def run(self):
        if self.source == "webcam":
            dt = 1.0 / self.fps
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    self.marker_poses = self._build_frame(frame)
                    time.sleep(dt)
            except KeyboardInterrupt:
                pass
            finally:
                self.cap.release()
        else:
            try:
                self._rclpy.spin(self._ros_node)
            except KeyboardInterrupt:
                pass
            finally:
                self._ros_node.destroy_node()
                if self._rclpy.ok():
                    self._rclpy.shutdown()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    rclpy.init()
    stream = WebVideoStream(
        source="ros",  # webcam or "ros"
        port=5000,
        fps=30.0,
        display_scale=2.0,
        depth_colormap="turbo",
        marker_sizes=[0.03, 0.05],
        dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
        color_topic="/rgbd_camera/image",
        depth_topic="/rgbd_camera/depth_image",
        camera_info_topic="/rgbd_camera/camera_info",
    )
    stream.run()