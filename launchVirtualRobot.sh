#!/usr/bin/env bash

ros2 launch annin_ar4_gazebo gazebo.launch.py &
sleep 6

# Camera is now included directly in the world file (empty.world)
# No need to spawn it separately

sleep 2

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True


