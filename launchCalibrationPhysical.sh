ros2 launch annin_ar4_driver driver.launch.py calibrate:=True include_gripper:=False
sleep 2
ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=False