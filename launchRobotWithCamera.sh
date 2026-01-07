#!/usr/bin/env bash

ros2 launch annin_ar4_gazebo gazebo.launch.py &
GAZEBO_PID=$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to initialize..."
sleep 6

# Launch MoveIt
ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True &
MOVEIT_PID=$!

# Wait for system to stabilize
sleep 4

# Spawn the RGB camera
echo "Spawning RGB camera..."
gz service -s /world/default/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req 'sdf_filename: "/home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/RGB_camera_model.sdf"'

if [ $? -eq 0 ]; then
  echo "Camera spawned successfully!"
  
  # Wait a moment for camera to initialize
  sleep 2
  
  # Bridge camera topics to ROS 2
  echo "Bridging camera topics to ROS 2..."
  ros2 run ros_gz_bridge parameter_bridge \
    /rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
    /rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo
else
  echo "Failed to spawn camera!"
fi

# Wait for all processes
wait
