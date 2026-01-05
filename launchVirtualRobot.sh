#!/usr/bin/env bash

ros2 launch annin_ar4_gazebo gazebo.launch.py &
sleep 8
ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True