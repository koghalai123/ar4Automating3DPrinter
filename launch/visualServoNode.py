from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    cam_xyz = ["0.0", "0.0", "0.08"]
    cam_rpy = ["0.0", "0.0", "0.0"]
    opt_rpy = ["-1.5707963", "0.0", "-1.5707963"]

    return LaunchDescription([
        ExecuteProcess(
            cmd=["/bin/bash", "-c", "source ~/ar4_ws/install/setup.bash && " +
                              "~/ar4_ws/src/ar4Automating3DPrinter/launchVirtualRobot.sh"],
            output="screen"
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="stf_link6_to_cam",
            arguments=cam_xyz + cam_rpy + ["link_6", "ee_camera_link"],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="stf_cam_to_optical",
            arguments=["0", "0", "0"] + opt_rpy + ["ee_camera_link", "ee_camera_optical_frame"],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="stf_base_to_target",
            arguments=["0.2", "0.0", "0.3", "0", "0", "0", "base_link", "target"],
            output="screen",
        ),

        # Run visualServo.py with ROS params
        ExecuteProcess(
            cmd=["/bin/bash", "-lc",
                 "source ~/ar4_ws/install/setup.bash && "
                 "python3 ~/ar4_ws/src/ar4Automating3DPrinter/visualServo.py "
                 "--ros-args "
                 "-p base_frame:=base_link "
                 "-p camera_frame:=ee_camera_optical_frame "
                 "-p target_frame:=target "
                 "-p pos_gain:=0.6 "
                 "-p rot_gain:=0.6 "
                 "-p pos_step_max:=0.01 "
                 "-p rot_step_max_deg:=5.0 "
                 "-p rate_hz:=2.0"],
            output="screen",
        ),
    ])