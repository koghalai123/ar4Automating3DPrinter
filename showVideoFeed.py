#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraViewer(Node):
    def __init__(self, node_name: str = None):
        super().__init__(node_name or 'camera_viewer')

        self.declare_parameter('color_topic', '/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/rgbd_camera/depth_image')
        self.declare_parameter('colormap', 'turbo')

        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.colormap_name = self.get_parameter('colormap').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None

        self.color_sub = self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.timer = self.create_timer(1.0 / 30.0, self.render)

        self.get_logger().info(f'Camera viewer subscribed to color: {color_topic}, depth: {depth_topic}')
        self.get_logger().info('Press "q" in the window to quit')

    def color_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Color image error: {str(e)}')

    def depth_callback(self, msg: Image):
        try:
            # Keep original floats/uint16 then visualize later
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth image error: {str(e)}')

    def render(self):
        if self.latest_color is None or self.latest_depth is None:
            return

        color_img = self.latest_color
        depth_raw = self.latest_depth

        # Resize depth to match color height
        if depth_raw.shape[:2] != color_img.shape[:2]:
            depth_raw = cv2.resize(depth_raw, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth_vis = self._visualize_depth(depth_raw)

        # If color and depth have different channels, ensure both are 3-channel BGR for display
        if len(color_img.shape) == 2:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
        if len(depth_vis.shape) == 2:
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([color_img, depth_vis])
        cv2.imshow('RGB (left) + Depth (right)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit key pressed, shutting down...')
            rclpy.shutdown()

    def _visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        # Handle float32 (meters) or uint16 depths
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            d = depth.copy()
            # Mask invalid values
            mask = np.isfinite(d) & (d > 0)
            if not np.any(mask):
                return np.zeros_like(d, dtype=np.uint8)
            # Robust scaling using percentiles to avoid outliers
            low = float(np.percentile(d[mask], 1.0))
            high = float(np.percentile(d[mask], 99.0))
            if high <= low:
                high = low + 1e-3
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)
        elif depth.dtype == np.uint16:
            d = depth.astype(np.float32)
            # Heuristic: many sources publish mm; normalize without unit assumption
            low = float(np.percentile(d[d > 0], 1.0)) if np.any(d > 0) else 0.0
            high = float(np.percentile(d[d > 0], 99.0)) if np.any(d > 0) else 1.0
            if high <= low:
                high = low + 1.0
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)
        else:
            # Unknown format; try to normalize generically
            d = depth.astype(np.float32)
            low = float(np.percentile(d, 1.0))
            high = float(np.percentile(d, 99.0))
            if high <= low:
                high = low + 1.0
            d = np.clip(d, low, high)
            d_norm = ((d - low) / (high - low) * 255.0).astype(np.uint8)

        cmap = self._cv_colormap(self.colormap_name)
        return cv2.applyColorMap(d_norm, cmap)

    def _cv_colormap(self, name: str) -> int:
        if name.lower() == 'turbo':
            return cv2.COLORMAP_TURBO
        if name.lower() == 'jet':
            return cv2.COLORMAP_JET
        if name.lower() == 'inferno':
            return cv2.COLORMAP_INFERNO
        return cv2.COLORMAP_TURBO


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
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
