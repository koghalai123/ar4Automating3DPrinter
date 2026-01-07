#!/usr/bin/env bash

# Launch Gazebo with empty world and sensors plugin
echo "Launching Gazebo with sensor support..."
gz sim -r -v 4 /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/RGB_camera.sdf &
GAZEBO_PID=$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to start..."
sleep 5

# Check if camera topics exist
echo "Checking for camera topics..."
gz topic -l | grep rgb_camera

# Bridge to ROS 2
echo "Bridging camera to ROS 2..."
ros2 run ros_gz_bridge parameter_bridge \
  /rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo &
BRIDGE_PID=$!

echo ""
echo "Test setup complete!"
echo "Camera should be publishing on:"
echo "  Gazebo: /rgb_camera/image"
echo "  ROS 2:  /rgb_camera/image"
echo ""
echo "Test with:"
echo "  gz topic -hz /rgb_camera/image"
echo "  ros2 topic hz /rgb_camera/image"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for user interrupt
wait
