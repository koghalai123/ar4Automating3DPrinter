#!/usr/bin/env bash

echo "============================================"
echo "Testing Camera in World (No Spawn)"
echo "============================================"
echo ""

# Launch Gazebo with world that includes the camera
echo "Step 1: Launching Gazebo with camera included in world..."
gz sim -r /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/empty_world_with_sensors.sdf &
GAZEBO_PID=$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to start..."
sleep 8

# Check if camera topics exist
echo ""
echo "Step 2: Checking for camera topics in Gazebo..."
gz topic -l | grep rgb_camera

# Bridge to ROS 2
echo ""
echo "Step 3: Bridging camera topics to ROS 2..."
ros2 run ros_gz_bridge parameter_bridge \
  /rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo &
BRIDGE_PID=$!

sleep 2

echo ""
echo "============================================"
echo "✓ Test Setup Complete!"
echo "============================================"
echo ""
echo "Camera should be publishing on:"
echo "  Gazebo: /rgb_camera/image"
echo "  ROS 2:  /rgb_camera/image"
echo ""
echo "Test with:"
echo "  gz topic -hz /rgb_camera/image"
echo "  ros2 topic hz /rgb_camera/image"
echo "  ros2 topic echo /rgb_camera/camera_info --once"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Wait for user interrupt
wait
