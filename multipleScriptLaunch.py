import rclpy
from rclpy.executors import MultiThreadedExecutor

from webcamNode import WebcamPublisher
from ArucoDetector import ArucoDetectionViewer


def main():
    rclpy.init()

    webcam_node = WebcamPublisher(
        camera_keyword="GENERAL WEBCAM",
        show_preview=False,
    )
    aruco_node = ArucoDetectionViewer()

    executor = MultiThreadedExecutor()
    executor.add_node(webcam_node)
    executor.add_node(aruco_node)

    print("\n" + "=" * 50)
    print("Webcam publisher + Aruco detector running.")
    print("View stream at http://localhost:5000")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    executor.spin()

    webcam_node.destroy_node()
    aruco_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()