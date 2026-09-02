#!/usr/bin/env python3
"""Shared setup/menu boilerplate for the run*/scanFor* entry scripts."""

import math
import os
import time
import threading

import rclpy

from .printer_automation import printerAutomation
from .simulated3DPrinter import Simulated3DPrinter

# standard hardware config for all runner scripts
WEBCAM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="webcam",
    feed_rotation_deg=90.0,
    marker_sizes=[0.03, 0.025],
)
WEBCAM_DISTANCE_SCALE = 1.0 / 0.702

# Gazebo: images come from the bridged RGBD camera on /rgbd_camera/*, so no
# webcam calibration file or distance-scale correction here.
SIM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="ros",
)

# sim printer layouts per robot (positions in the robot's good frame);
# marker IDs match the door textures
SIM_PRINTER_SPECS = {
    'ar4': [
        {"marker_id": 0, "pos": [0.22, -0.2, 0.21], "orient": [0.0, 0.0, math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_0.png'},
        # y must stay -0.3: at -0.2 the 0.38m scrape standoff has no IK solution
        # and scrapePlate aborts at the scrape waypoints
        {"marker_id": 1, "pos": [0.44, -0.3, 0.21], "orient": [0.0, 0.0, math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_1.png'},
        {"marker_id": 2, "pos": [0.60, 0.1, 0.21], "orient": [0.0, 0.0, 3/2*math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_2.png'},
    ],
    # tuned in sim for the lite6's 0.44 m reach: the far (1.75x) viewing pose
    # of each marker must stay reachable even with the 0.03 m estimate
    # randomization. Reach-probed grid: door-facing-+y poses only plan
    # reliably for marker x <= ~0.30 with y around -0.24 (at 0.263 m standoff).
    'lite6': [
        {"marker_id": 0, "pos": [0.14, -0.16, 0.15], "orient": [0.0, 0.0, math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_0.png'},
        {"marker_id": 1, "pos": [0.3, -0.3, 0.18], "orient": [0.0, 0.0, math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_1.png'},
        {"marker_id": 2, "pos": [0.58, 0.08, 0.20], "orient": [0.0, 0.0, 3/2*math.pi],
         "door_marker_texture": 'materials/textures/marker6x6_2.png'},
    ],
}


def sim_printer_specs(robot='ar4', count=3):
    """The robot's sim printer layout; count=2 drops marker 0 (ids 1+2)."""
    return SIM_PRINTER_SPECS[robot][-count:]


# legacy names (AR4 layout)
SIM_PRINTER_SPECS_3 = SIM_PRINTER_SPECS['ar4']
SIM_PRINTER_SPECS_2 = SIM_PRINTER_SPECS_3[1:]


def make_webcam_node(robot='ar4', **overrides):
    """Build a printerAutomation node with the standard webcam config."""
    kwargs = dict(WEBCAM_NODE_KWARGS)
    # Lite 6 uses the ROS RealSense/Gazebo source. The commissioned physical
    # xArm 6 has a conventional USB webcam, opened directly with OpenCV just
    # like the AR4. Its hand-eye file supplies the wrist-camera extrinsics.
    if robot == 'lite6':
        kwargs.update(SIM_NODE_KWARGS)
        kwargs.pop('feed_rotation_deg', None)
    elif robot == 'xarm6':
        camera_mode = os.environ.get('XARM_CAMERA_MODE', 'webcam').strip().lower()
        if camera_mode in {'disabled', 'none', 'off'}:
            # Movement-only commissioning: use non-blocking ROS subscriptions
            # with no expected publisher. Vision goals remain fail-closed via
            # the GUI backend camera preflight, but a missing/broken USB camera
            # cannot hold the entire robot backend in state=starting.
            kwargs.update(SIM_NODE_KWARGS)
            kwargs.pop('feed_rotation_deg', None)
        elif camera_mode == 'webcam':
            kwargs['feed_rotation_deg'] = 0.0
            camera_index = os.environ.get('XARM_CAMERA_INDEX', '').strip()
            if camera_index:
                kwargs['camera_index'] = int(camera_index)
        else:
            raise ValueError(
                "XARM_CAMERA_MODE must be 'webcam' or 'disabled'")
    kwargs.update(overrides)
    node = printerAutomation(robot=robot, **kwargs)
    # The 0.702 correction was calibrated for the AR4 webcam.  The Lite 6
    # RealSense depth stream is metric and must not inherit that scale.
    if robot == 'ar4':
        node.stream.distance_scale = WEBCAM_DISTANCE_SCALE
    profile = node.robot_config.get('physical_motion', {})
    node.moveit2.max_velocity = profile.get('max_velocity', 0.15)
    node.moveit2.max_acceleration = profile.get('max_acceleration', 0.15)
    node.move_settle_delay = 0.5
    return node


def make_sim_node(robot='ar4', **overrides):
    """Build a printerAutomation node fed by the Gazebo camera."""
    kwargs = dict(SIM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(robot=robot, **kwargs)
    # gripper action is flaky in sim, see printerAutomation.gripper_disabled
    node.gripper_disabled = True
    node.simulation_mode = True
    # Smooth enough for interactive GUI testing without making every
    # waypoint sequence unnecessarily slow.
    profile = node.robot_config.get('simulation_motion', {})
    node.moveit2.max_velocity = profile.get('max_velocity', 0.35)
    node.moveit2.max_acceleration = profile.get('max_acceleration', 0.30)
    node.move_settle_delay = 0.25
    return node


def start_node(sim=False, robot='ar4', joint_state_timeout=10.0, **overrides):
    """(make_webcam_node or make_sim_node) + spin_in_background + wait_for_joint_states.

    robot: 'ar4', 'lite6', or 'xarm6' (see robot_config.py). Sim launch scripts:
    launchVirtualRobot.sh for the AR4, launchVirtualXArmLite6.sh for the Lite 6.
    """
    node = make_sim_node(robot=robot, **overrides) if sim else make_webcam_node(robot=robot, **overrides)
    spin_in_background(node)
    wait_for_joint_states(node, timeout=joint_state_timeout)
    if not sim and 'xarm_safety' in node.robot_config:
        if not node.configure_xarm_safety():
            node.get_logger().error(
                "UFACTORY safety profile was not applied; motion will remain blocked: "
                f"{node._xarm_safety_error}")
        else:
            try:
                # Changing reduced/self-collision settings can leave the
                # ros2_control trajectory controller inactive even after the
                # physical arm is restored to START. The backend is not ready
                # until both ownership layers are active.
                node.set_trajectory_controller_active(True)
            except Exception as exc:
                node._xarm_safety_configured = False
                node._xarm_safety_error = (
                    f"trajectory controller activation failed: {exc}")
                node.get_logger().error(node._xarm_safety_error)
    return node


def spawn_sim_printers(node, specs):
    """Spawn a sim printer per spec and register its door marker estimate so scans know where to look."""
    printers = []
    for p in specs:
        printer = Simulated3DPrinter(
            node=node,
            pos=p["pos"],
            orient=p["orient"],
            door_marker_texture=p["door_marker_texture"],
        )
        printer.spawn_fast()
        bad_pos, bad_euler = printer.get_door_marker_pose_in_base()
        node.register_estimated_marker(
            marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
        )
        printers.append(printer)
    return printers


def spin_in_background(node):
    """Start a MultiThreadedExecutor for the node (and its stream) in a daemon thread."""
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    if hasattr(node.stream, "_ros_node"):
        executor.add_node(node.stream._ros_node)
    else:
        threading.Thread(target=node.stream.run, daemon=True).start()

    def _resilient_spin():
        # must be spin(), not a spin_once loop: the always-ready 30 Hz sim camera
        # callback would starve TF/joint_states/MoveIt results and every sim move
        # "times out" despite executing
        while rclpy.ok():
            try:
                executor.spin()
            except Exception as e:
                node.get_logger().warn(f"Executor spin error (recovering): {e}")
                time.sleep(0.01)

    threading.Thread(target=_resilient_spin, daemon=True).start()
    return executor


def wait_for_joint_states(node, timeout=10.0):
    """Block until the first joint_states message arrives (or warn on timeout)."""
    node.get_logger().info("Waiting for joint_states...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node._last_joint_msg is not None:
            return True
        time.sleep(0.1)
    node.get_logger().warn("Timed out waiting for joint_states — proceeding anyway")
    return False


def start_webcam_node(robot='ar4', **overrides):
    """make_webcam_node + spin_in_background + wait_for_joint_states."""
    node = make_webcam_node(robot=robot, **overrides)
    spin_in_background(node)
    wait_for_joint_states(node)
    return node


def restore_saved_printers(node):
    """Rebuild sim printers from load_state() configs; their door-marker estimates back-fill markers with no real saved pose."""
    for p in getattr(node, "_saved_printer_configs", []):
        printer = Simulated3DPrinter(
            node=node,
            pos=p["pos"],
            orient=p["orient"],
            door_marker_texture=p["door_marker_texture"],
        )
        bad_pos, bad_euler = printer.get_door_marker_pose_in_base()
        existing = node._find_marker_entry(p["marker_id"])
        if existing is None or existing.get("estimated"):
            node.register_estimated_marker(
                marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
            )


# ---------------------------------------------------------------------------
# Interactive command menu (used by scanFor2Markers.py / scanFor3Markers.py)
# ---------------------------------------------------------------------------

def _print_menu():
    print("\n" + "=" * 50)
    print("  3D Printer Automation - Command Menu")
    print("=" * 50)
    print("  1) Scan location for markers (manual pos/orient)")
    print("  2) Walk pickup waypoints (scan/gripper entries included)")
    print("  3) Pickup plate (walk pickup list, record grasp for replay)")
    print("  4) Place plate at marker")
    print("  5) List detected markers")
    print("  6) Scan to marker (by ID, uses TF)")
    print("  7) Go home & resync (correct step-loss drift)")
    print("  8) Transfer plate (source → dest, rescan → place)")
    print("  9) Scrape plate (pickup → scrape surface → return)")
    print("=" * 50)


def _parse_floats(prompt, count=None):
    """Prompt for space-separated floats, None on bad input."""
    try:
        raw = input(prompt).strip()
        values = [float(v) for v in raw.split()]
        if count is not None and len(values) != count:
            print(f"  Expected {count} values, got {len(values)}")
            return None
        return values
    except ValueError:
        print("  Invalid input. Enter space-separated numbers.")
        return None


def run_command_menu(node):
    """Stdin command loop; node must already be spinning in the background. Blocks until EOF."""
    time.sleep(5.0)
    print("\n[INFO] System ready. Type a command number.")
    node.record_startup_time()

    while rclpy.ok():
        _print_menu()
        try:
            choice = input(">> ").strip()
        except EOFError:
            break

        # ROS log output can interleave with terminal input and leave stray
        # leading chars before the digit the user typed
        _valid_choices = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        if choice not in _valid_choices and len(choice) >= 2 and choice[1:] in _valid_choices:
            choice = choice[1:]

        if choice == "1":
            pos = _parse_floats("  Enter estimated pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter estimated orient (roll pitch yaw) [0 0 0]: ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanLocationForMarkers at {pos}")
            node.scanLocationForMarkers(estimated_pos=pos, estimated_orient=orient, viewing_distance=dist)

        elif choice == "2":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested moveToMarker({mid})")
            node.moveToMarker(markerID=mid)

        elif choice == "3":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested pickupPlate({mid})")
            node.pickupPlate(markerID=mid)

        elif choice == "4":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested placePlate({mid})")
            node.placePlate(markerID=mid)

        elif choice == "5":
            markers = node.marker_poses
            if markers:
                print(f"\n  Found {len(markers)} marker(s):")
                for entry in markers:
                    pos = entry.get('positionInWorld', 'N/A')
                    ori = entry.get('orientInWorld', 'N/A')
                    print(f"    ID {entry['id']}: pos={pos}, orient={ori}")
            else:
                print("  No markers found yet.")

        elif choice == "6":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanToMarker({mid}, dist={dist})")
            node.scanToMarker(marker_id=mid, viewing_distance=dist)

        elif choice == "7":
            scale = _parse_floats("  Velocity scaling [0.2]: ", 1)
            scale = scale[0] if scale else 0.2
            node.get_logger().info(f"User requested go_home(velocity_scaling={scale})")
            node.go_home(velocity_scaling=scale)

        elif choice == "8":
            ids = _parse_floats("  Source, dest, rescan marker IDs (e.g. 1 2 1): ", 3)
            if ids is None:
                continue
            source_id, dest_id, rescan_id = int(ids[0]), int(ids[1]), int(ids[2])
            node.get_logger().info(
                f"User requested transferPlate({source_id}, {dest_id}, {rescan_id})"
            )
            node.transferPlate(source_id=source_id, dest_id=dest_id, rescan_id=rescan_id)

        elif choice == "9":
            ids = _parse_floats("  Source, scrape marker IDs (e.g. 1 2): ", 2)
            if ids is None:
                continue
            source_id, scrape_id = int(ids[0]), int(ids[1])
            # all motion (scans included) comes from the offset-config waypoint lists
            node.get_logger().info(
                f"User requested scrapePlate({source_id}, {scrape_id})"
            )
            node.scrapePlate(source_id=source_id, scrape_id=scrape_id)

        else:
            print("  Unknown option. Try again.")
