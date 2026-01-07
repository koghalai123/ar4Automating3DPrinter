#!/usr/bin/env python3
"""
Simple ROS 2 image viewer for Gazebo/ROS camera topics.

Usage (example):
  source ~/ar4_ws/install/setup.bash
  python3 src/ar4Automating3DPrinter/showVideoFeed.py --topic /overhead_camera

Options:
  --topic: Image topic to subscribe to (default: /overhead_camera)
"""

import argparse
import signal
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
	from cv_bridge import CvBridge
except Exception as exc:  # pragma: no cover - import-time error path
	raise SystemExit(
		"cv_bridge is required. Install: sudo apt install ros-jazzy-cv-bridge"
	) from exc


class ImageViewer(Node):
	def __init__(self, topic: str, window: str = "Camera") -> None:
		super().__init__("image_viewer")
		self._bridge = CvBridge()
		self._window = window
		self._last_frame = None  # type: Optional[cv2.Mat]
		self._subscription = self.create_subscription(
			Image,
			topic,
			self._image_cb,
			qos_profile_sensor_data,
		)
		self.get_logger().info(f"Subscribed to: {topic}")

		# Use a timer to drive imshow without blocking callbacks
		self._timer = self.create_timer(1.0 / 60.0, self._show_frame)

	def _image_cb(self, msg: Image) -> None:
		try:
			frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			self._last_frame = frame
		except Exception as e:
			self.get_logger().warn(f"cv_bridge conversion failed: {e}")

	def _show_frame(self) -> None:
		if self._last_frame is None:
			return
		cv2.imshow(self._window, self._last_frame)
		# waitKey with small delay keeps window responsive
		if cv2.waitKey(1) & 0xFF == ord("q"):
			self.get_logger().info("'q' pressed, shutting down viewer…")
			rclpy.shutdown()


def main() -> None:
	parser = argparse.ArgumentParser(description="ROS 2 Image Viewer")
	parser.add_argument(
		"--topic",
		default="/overhead_camera",
		help="Image topic to subscribe to",
	)
	args = parser.parse_args()

	rclpy.init()
	node = ImageViewer(topic=args.topic)

	# Graceful Ctrl+C handling for OpenCV window cleanup
	def _sigint_handler(signum, frame):  # noqa: ARG001
		node.get_logger().info("SIGINT received, shutting down…")
		rclpy.shutdown()

	signal.signal(signal.SIGINT, _sigint_handler)

	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

