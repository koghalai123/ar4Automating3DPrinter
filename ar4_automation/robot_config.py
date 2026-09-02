"""Per-robot names and conventions for the automation stack.

Select with the robot= kwarg on printerAutomation (threads down through
ArucoDetectionViewer and PoseReader), e.g.:
    node = start_node(sim=1, robot='lite6')

'ar4' is the Annin AR4; 'lite6' is the UFACTORY Lite 6; and 'xarm6' is the
UFACTORY xArm 6.  Lite 6 and xArm 6 are distinct robot models even though
both expose six joints through xarm_ros2.
"""

import numpy as np

ROBOT_CONFIGS = {
    'ar4': {
        'joint_names': ["joint_1", "joint_2", "joint_3",
                        "joint_4", "joint_5", "joint_6"],
        'base_link': "base_link",
        'end_effector_link': "link_6",
        'move_group': "ar_manipulator",
        'camera_frame': "ee_camera_link",
        'color_topic': "/rgbd_camera/image",
        'depth_topic': "/rgbd_camera/depth_image",
        'camera_info_topic': "/rgbd_camera/camera_info",
        # camera sits below the gripper; raise the EE this much (m) when scanning
        'camera_z_offset': 0.06,
        'gripper': {
            'gripper_joint_names': ["gripper_jaw1_joint"],
            'open_gripper_joint_positions': [0.00],
            'closed_gripper_joint_positions': [0.0145],
            'gripper_group_name': "ar_gripper",
            'gripper_command_action_name': "gripper_controller/gripper_cmd",
        },
        # bad frame (base_link) -> good frame rotation, and the euler offset of
        # the neutral tool orientation (calibrated for the AR4)
        'frame_rotation_angles': np.array([0.0, 0.0, np.pi / 2]),
        'frame_offset_angles': np.array([-0.6162, -1.5706, -2.1870]),
        # tool orientation used to face a marker (bad-frame euler XYZ)
        'offset_ori': np.array([0.0, np.pi, np.pi / 2]),
    },
    'lite6': {
        'joint_names': ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6"],
        'base_link': "link_base",
        'end_effector_link': "link_eef",
        'move_group': "lite6",
        'trajectory_controller': "lite6_traj_controller",
        # simulated RealSense D435i (add_realsense_d435i:=true on the launch)
        'camera_frame': "camera_color_optical_frame",
        'color_topic': "/camera/color/image_raw",
        'depth_topic': "/camera/depth/image",
        'camera_info_topic': "/camera/color/camera_info",
        # with the flipped offset_ori below the D435i hangs ~4 cm below
        # link_eef, so raise the EEF to keep the marker centered (mirror of the
        # -0.04 that the camera-above roll needed; re-check on the first scans)
        'camera_z_offset': 0.04,
        # Lite 6 built-in gripper is controlled by xarm_api services rather
        # than a FollowJointTrajectory/GripperCommand action.
        'gripper': {
            'type': 'lite6_service',
            'namespace': '/ufactory',
        },
        # good frame == base frame for the lite6 (robot spawns at the world
        # origin with zero yaw, so no AR4-style 90 deg convention)
        'frame_rotation_angles': np.array([0.0, 0.0, 0.0]),
        'frame_offset_angles': np.array([0.0, 0.0, 0.0]),
        # AR4 value rolled 180 deg about the tool approach axis: the D435i is
        # mounted on the opposite side of the eef here, so the unrolled AR4
        # orientation put the camera above the gripper instead of below it.
        # Same tool Z (still faces the marker), only the roll differs.
        'offset_ori': np.array([np.pi, 0.0, np.pi / 2]),
        # Software interlocks used before every GUI/automation motion.  These
        # complement (and never replace) MoveIt's self/environment collision
        # checking and the controller's collision detection.
        #
        # Joint limits mirror xarm_description's lite6 URDF, with a small
        # margin so a command is not planned directly against a hard stop.
        'joint_limits': [
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
            (-2.61799 + 0.035, 2.61799 - 0.035),
            (-0.061087 + 0.035, np.pi * 0.99 - 0.035),
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
            (-2.1642 + 0.035, 2.1642 - 0.035),
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
        ],
        # Conservative automation envelope in link_base coordinates.  It is
        # intentionally configurable: commissioning must narrow it around the
        # actual printer cell before unattended use.
        'workspace': {
            'x': (-0.48, 0.48),
            'y': (-0.48, 0.48),
            'z': (0.025, 0.65),
            'radius': (0.08, 0.50),
        },
        'physical_motion': {
            'max_velocity': 0.10,
            'max_acceleration': 0.10,
            'max_manual_joint_delta': np.radians(20.0),
            'max_jog_translation': 0.02,
            'max_jog_rotation': np.radians(10.0),
            'joint_state_max_age': 1.0,
        },
        'simulation_motion': {
            'max_velocity': 0.35,
            'max_acceleration': 0.30,
            'max_manual_joint_delta': np.radians(60.0),
            'max_jog_translation': 0.05,
            'max_jog_rotation': np.radians(15.0),
            'joint_state_max_age': 2.0,
        },
        'xarm_safety': {
            'namespace': '/ufactory',
            # UFACTORY accepts 0..5.  Level 3 is a conservative commissioning
            # default without selecting the most nuisance-prone setting.
            'collision_sensitivity': 3,
            'self_collision_detection': True,
            'reduced_mode': True,
            'reduced_max_tcp_speed_mm_s': 100.0,
            'reduced_max_joint_speed_rad_s': 0.35,
        },
    },
    'xarm6': {
        'joint_names': ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6"],
        'base_link': "link_base",
        'end_effector_link': "link_eef",
        'move_group': "xarm6",
        'trajectory_controller': "xarm6_traj_controller",
        'camera_frame': "camera_color_optical_frame",
        'color_topic': "/camera/color/image_raw",
        'depth_topic': "/camera/depth/image",
        'camera_info_topic': "/camera/color/camera_info",
        # Provisional until the physical wrist-camera transform is measured.
        'camera_z_offset': 0.04,
        # The actual end tool has not yet been identified/commissioned.  A
        # physical manipulation routine must fail instead of simulating a
        # successful grasp with a no-op gripper.
        'gripper': None,
        'frame_rotation_angles': np.array([0.0, 0.0, 0.0]),
        'frame_offset_angles': np.array([0.0, 0.0, 0.0]),
        'offset_ori': np.array([np.pi, 0.0, np.pi / 2]),
        # xarm_description/urdf/xarm6/xarm6_robot_macro.xacro, limited=true,
        # with a 0.035 rad margin from each hard limit.
        'joint_limits': [
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
            (-2.059 + 0.035, 2.0944 - 0.035),
            (-np.pi * 0.99 + 0.035, 0.19198 - 0.035),
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
            (-1.69297 + 0.035, np.pi * 0.99 - 0.035),
            (-np.pi * 0.99 + 0.035, np.pi * 0.99 - 0.035),
        ],
        # Initial commissioning envelope. Narrow it to the measured printer
        # cell before unattended operation.
        'workspace': {
            'x': (-0.75, 0.75),
            'y': (-0.75, 0.75),
            'z': (0.025, 0.90),
            'radius': (0.10, 0.90),
        },
        'physical_motion': {
            'max_velocity': 0.08,
            'max_acceleration': 0.08,
            'max_manual_joint_delta': np.radians(10.0),
            'max_jog_translation': 0.01,
            'max_jog_rotation': np.radians(5.0),
            'joint_state_max_age': 1.0,
        },
        'simulation_motion': {
            'max_velocity': 0.25,
            'max_acceleration': 0.20,
            'max_manual_joint_delta': np.radians(45.0),
            'max_jog_translation': 0.05,
            'max_jog_rotation': np.radians(15.0),
            'joint_state_max_age': 2.0,
        },
        'xarm_safety': {
            'namespace': '/xarm',
            'collision_sensitivity': 3,
            'self_collision_detection': True,
            'reduced_mode': True,
            'reduced_max_tcp_speed_mm_s': 80.0,
            'reduced_max_joint_speed_rad_s': 0.25,
        },
    },
}


def get_robot_config(robot):
    try:
        return ROBOT_CONFIGS[robot]
    except KeyError:
        raise ValueError(
            f"Unknown robot '{robot}'. Available: {list(ROBOT_CONFIGS)}")
