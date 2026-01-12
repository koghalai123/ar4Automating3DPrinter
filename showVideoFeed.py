#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        # Create CV Bridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/rgb_camera_moving/image',
            self.image_callback,
            10
        )
        
        self.get_logger().info('Camera viewer node started. Subscribing to /rgb_camera_moving/image')
        self.get_logger().info('Press "q" in the window to quit')
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Display the image in a window
            cv2.imshow('Robot Arm Camera Feed', cv_image)
            
            # Wait for 1ms and check if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Quit key pressed, shutting down...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    camera_viewer = CameraViewer()
    
    try:
        rclpy.spin(camera_viewer)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        camera_viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
