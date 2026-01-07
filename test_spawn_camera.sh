#!/usr/bin/env bash

echo "============================================"
echo "Testing Camera Spawn in Separate World"
echo "============================================"
echo ""

# Step 1: Launch empty world with sensors plugin
echo "Step 1: Launching empty Gazebo world with sensor support..."
gz sim -r -v 4 /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/empty_world_with_sensors.sdf &
GAZEBO_PID=$!

# Wait for Gazebo to fully initialize
echo "Waiting for Gazebo to initialize..."
sleep 6

# Step 2: Verify world is running
echo ""
echo "Step 2: Checking if world is ready..."
gz topic -l | grep "/world/default" | head -3

# Step 3: Spawn the camera using the service
echo ""
echo "Step 3: Spawning RGB camera into the world..."

# Use ros_gz_sim create node instead of gz service
# This properly handles SDF files
ros2 run ros_gz_sim create \
  -file /home/koghalai/ar4_ws/src/ar4Automating3DPrinter/models/RGB_camera_model.sdf \
  -world default

# Check spawn result
if [ $? -eq 0 ]; then
  echo "✓ Camera spawned successfully!"
else
  echo "✗ Failed to spawn camera!"
  exit 1
fi

# Step 4: Wait for camera to initialize
echo ""
echo "Step 4: Waiting for camera to initialize..."
sleep 2

# Step 5: Make sure simulation is running (not paused)
echo ""
echo "Step 5: Ensuring simulation is running..."
gz topic -t /world/default/control --msgtype gz.msgs.WorldControl -m 'pause: false'

sleep 1

# Step 6: Check if camera topics exist
echo ""
echo "Step 6: Checking for camera topics in Gazebo..."
gz topic -l | grep rgb_camera

# Step 7: Bridge to ROS 2
echo ""
echo "Step 7: Bridging camera topics to ROS 2..."
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
echo "  ros2 topic echo /rgb_camera/image --once"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Wait for user interrupt
wait
