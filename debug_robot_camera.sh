#!/usr/bin/env bash

echo "Testing AR4 Robot World with Camera"
echo "===================================="

# Launch the AR4 Gazebo world
ros2 launch annin_ar4_gazebo gazebo.launch.py &
LAUNCH_PID=$!

echo "Waiting for Gazebo to load..."
sleep 10

echo ""
echo "Checking for camera topics in Gazebo..."
gz topic -l | grep rgb_camera

echo ""
echo "Checking if camera sensor exists..."
gz model -m rgb_camera -l link -s 2>&1 | grep -E "(Name:|Sensor)"

echo ""
echo "Checking all models in world..."
gz model --list

echo ""
echo "Press Ctrl+C to stop..."
wait
