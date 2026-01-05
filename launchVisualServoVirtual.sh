#!/usr/bin/env bash
set -euo pipefail

# Clean up any existing ROS 2 / Gazebo / MoveIt processes
echo "Stopping existing ROS 2 processes (rviz2, move_group, gzserver/gzclient, TF publishers, visualServo)..."
ROS_PATTERNS=(
  "rviz2"
  "move_group"
  "gzserver"
  "gzclient"
  "static_transform_publisher"
  "robot_state_publisher"
  "visualServo.py"
  "ros2 launch annin_ar4_moveit_config"
  "ros2 launch annin_ar4_gazebo"
)

# Try graceful termination first
for pat in "${ROS_PATTERNS[@]}"; do
  pkill -f -TERM "$pat" 2>/dev/null || true
done
sleep 1

# Force kill any stubborn processes
for pat in "${ROS_PATTERNS[@]}"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    pkill -f -KILL "$pat" 2>/dev/null || true
  fi
done

# Stop ROS 2 daemon to avoid stale graph
ros2 daemon stop 2>/dev/null || true

# Source ROS 2 and workspace
set +u
source /opt/ros/jazzy/setup.bash
source ~/ar4_ws/install/setup.bash
set -u

# Start Gazebo + MoveIt
~/ar4_ws/src/ar4Automating3DPrinter/launchVirtualRobot.sh &
LAUNCH_PID=$!

# Wait for Gazebo to publish /clock
echo "Waiting for /clock..."
for i in {1..60}; do
  if ros2 topic list | grep -qx '/clock'; then break; fi
  sleep 0.5
done

# Wait for move_group and rviz2 nodes
echo "Waiting for move_group and rviz2..."
for i in {1..60}; do
  NODES="$(ros2 node list || true)"
  [[ "$NODES" == *"/move_group"* ]] && [[ "$NODES" == *"/rviz2"* ]] && break
  sleep 0.5
done

# Ensure they use sim time
ros2 param set /move_group use_sim_time true || true
ros2 param set /rviz2 use_sim_time true || true
sleep 1

# Publish static TFs under sim time (positional args first, then ROS args)
ros2 run tf2_ros static_transform_publisher 0 0 0.08 0 0 0 link_6 ee_camera_link --ros-args -p use_sim_time:=true & CAM1=$!
ros2 run tf2_ros static_transform_publisher 0 0 0 -1.5707963 0 -1.5707963 ee_camera_link ee_camera_optical_frame --ros-args -p use_sim_time:=true & CAM2=$!
ros2 run tf2_ros static_transform_publisher 0.2 0 0.3 0 0 0 base_link target --ros-args -p use_sim_time:=true & TGT=$!

trap "kill $LAUNCH_PID $CAM1 $CAM2 $TGT || true" EXIT
wait $LAUNCH_PID