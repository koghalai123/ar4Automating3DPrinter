#!/usr/bin/env bash

# Wait for Gazebo to be ready
echo "Waiting for Gazebo world to be ready..."
sleep 2

# Spawn the RGB camera into the running world
echo "Spawning RGB camera..."

# Use ros_gz_sim create node instead of gz service
# This properly handles SDF files
ros2 run ros_gz_sim create \
  -file /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/RGB_camera_model.sdf \
  -world default

# Check if spawn was successful
if [ $? -eq 0 ]; then
  echo "Camera spawned successfully!"
  
  # Bridge camera topics to ROS 2
  echo "Bridging camera topics to ROS 2..."
  ros2 run ros_gz_bridge parameter_bridge \
    /rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
    /rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo
else
  echo "Failed to spawn camera!"
  exit 1
fi
