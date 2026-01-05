#!/usr/bin/env python3

import time
import argparse
import numpy as np
import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from pymoveit2 import MoveIt2


class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, i_limit=1.0, out_limit=0.05):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = float(i_limit)
        self.out_limit = float(out_limit)
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev_err = None

    def step(self, err, dt):
        if dt <= 0.0:
            return 0.0
        # Integral
        self._i += err * dt
        self._i = np.clip(self._i, -self.i_limit, self.i_limit)
        # Derivative
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        # Output and clamp
        out = self.kp * err + self.ki * self._i + self.kd * d
        return float(np.clip(out, -self.out_limit, self.out_limit))


class AR4ServoPID(Node):
    def __init__(self):
        super().__init__("ar4_servo_pid")

        # MoveIt2 interface (aligns with old_ar4Robot.py)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        self.moveit2.max_velocity = 1.5
        self.moveit2.max_acceleration = 3.0

        # Position and orientation PID (per-axis, same gains each axis)
        self.pid_pos = [PID(kp=1.0, ki=0.0, kd=0.1, i_limit=0.1, out_limit=0.02) for _ in range(3)]
        self.pid_rot = [PID(kp=1.0, ki=0.0, kd=0.05, i_limit=0.2, out_limit=0.05) for _ in range(3)]

    def _get_current_pose(self):
        fk = self.moveit2.compute_fk()
        if fk is None:
            self.get_logger().warn("compute_fk returned None")
            return None, None
        pose_msg = fk[0] if isinstance(fk, list) else fk
        p = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ],
            dtype=float,
        )
        quat = [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ]
        r, pch, y = euler_from_quaternion(quat)  # radians
        rpy = np.array([r, pch, y], dtype=float)
        return p, rpy

    def _move_to_pose(self, pos_xyz, rpy):
        q = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        q_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.moveit2.move_to_pose(position=Point(x=pos_xyz[0], y=pos_xyz[1], z=pos_xyz[2]), quat_xyzw=q_msg)
        self.moveit2.wait_until_executed()

    @staticmethod
    def _angle_wrap(err):
        # Wrap angle error to [-pi, pi]
        return (err + np.pi) % (2.0 * np.pi) - np.pi

    def servo_to_delta(
        self,
        delta_pos=np.zeros(3),
        delta_rpy=np.zeros(3),
        pos_tol=1e-3,
        rot_tol=5e-3,
        rate_hz=20.0,
        max_duration=8.0,
    ):
        """
        PID servo to achieve a delta in position (m) and orientation (rad) from current pose.
        - delta_pos: [dx, dy, dz] in meters
        - delta_rpy: [droll, dpitch, dyaw] in radians
        """

        # Reset controllers
        for pid in self.pid_pos + self.pid_rot:
            pid.reset()

        start = time.time()
        dt_target = 1.0 / float(rate_hz)

        # Snapshot current as start, compute goal
        curr_pos, curr_rpy = self._get_current_pose()
        if curr_pos is None:
            self.get_logger().error("Could not read current pose")
            return False

        goal_pos = curr_pos + np.asarray(delta_pos, dtype=float)
        goal_rpy = curr_rpy + np.asarray(delta_rpy, dtype=float)

        # Control loop
        while True:
            now = time.time()
            if now - start > max_duration:
                self.get_logger().warn("Servo timeout reached")
                break

            pos, rpy = self._get_current_pose()
            if pos is None:
                self.get_logger().warn("Failed to get pose during servo; aborting")
                break

            # Errors
            e_pos = goal_pos - pos  # meters
            e_rot = np.array(
                [
                    self._angle_wrap(goal_rpy[0] - rpy[0]),
                    self._angle_wrap(goal_rpy[1] - rpy[1]),
                    self._angle_wrap(goal_rpy[2] - rpy[2]),
                ]
            )

            # Check convergence
            if np.linalg.norm(e_pos) <= pos_tol and np.linalg.norm(e_rot) <= rot_tol:
                self.get_logger().info("Goal reached within tolerances")
                break

            # PID outputs (per-axis, integrated over dt into small steps)
            # We compute incremental targets relative to current pose
            dt = dt_target  # fixed dt to keep behavior consistent even with blocking motion
            u_pos = np.array([pid.step(e_pos[i], dt) for i, pid in enumerate(self.pid_pos)])
            u_rot = np.array([pid.step(e_rot[i], dt) for i, pid in enumerate(self.pid_rot)])

            next_pos = pos + u_pos
            next_rpy = rpy + u_rot

            # Command small step
            self._move_to_pose(next_pos, next_rpy)

            # Try to maintain approximate loop rate
            spent = time.time() - now
            sleep_t = max(0.0, dt_target - spent)
            if sleep_t > 0.0:
                time.sleep(sleep_t)

        # Final nudge to exact goal (single motion)
        self._move_to_pose(goal_pos, goal_rpy)
        return True


def _read_key(timeout=0.1):
    """Non-blocking single-key read with timeout (seconds). Returns None if no key."""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return None

def teleop_loop(node: AR4ServoPID, linear_step=0.002, angular_step=np.deg2rad(1.0), rate_hz=30.0, max_duration=1.2):
    """
    Teleop: map keys to incremental servo moves.
    q w e r t y -> +[x, y, z, roll, pitch, yaw]
    a s d f g h -> -[x, y, z, roll, pitch, yaw]
    ESC or x    -> exit
    """
    print("\nKeyboard teleop:")
    print("  q/a: +X / -X")
    print("  w/s: +Y / -Y")
    print("  e/d: +Z / -Z")
    print("  r/f: +Roll / -Roll")
    print("  t/g: +Pitch / -Pitch")
    print("  y/h: +Yaw / -Yaw")
    print("  ESC or x to exit\n")
    print(f"Steps: linear={linear_step} m, angular={np.rad2deg(angular_step):.1f} deg")

    key_to_delta = {
        "q": (np.array([+linear_step, 0, 0]), np.zeros(3)),
        "a": (np.array([-linear_step, 0, 0]), np.zeros(3)),
        "w": (np.array([0, +linear_step, 0]), np.zeros(3)),
        "s": (np.array([0, -linear_step, 0]), np.zeros(3)),
        "e": (np.array([0, 0, +linear_step]), np.zeros(3)),
        "d": (np.array([0, 0, -linear_step]), np.zeros(3)),
        "r": (np.zeros(3), np.array([+angular_step, 0, 0])),
        "f": (np.zeros(3), np.array([-angular_step, 0, 0])),
        "t": (np.zeros(3), np.array([0, +angular_step, 0])),
        "g": (np.zeros(3), np.array([0, -angular_step, 0])),
        "y": (np.zeros(3), np.array([0, 0, +angular_step])),
        "h": (np.zeros(3), np.array([0, 0, -angular_step])),
    }

    # Put terminal into raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            k = _read_key(timeout=0.1)
            if k is None:
                continue
            if k == "\x1b" or k.lower() == "x":  # ESC or x
                print("Exiting teleop.")
                break
            if k in key_to_delta:
                dp, drpy = key_to_delta[k]
                print(f"Key '{k}': dp={dp}, drpy(deg)={np.rad2deg(drpy)}")
                node.servo_to_delta(
                    delta_pos=dp,
                    delta_rpy=drpy,
                    rate_hz=rate_hz,
                    max_duration=max_duration,
                )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description="PID servo to a delta pose using MoveIt2")
    parser.add_argument("--dx", type=float, default=0.0, help="Delta X (m)")
    parser.add_argument("--dy", type=float, default=0.0, help="Delta Y (m)")
    parser.add_argument("--dz", type=float, default=0.0, help="Delta Z (m)")
    parser.add_argument("--dr", type=float, default=0.0, help="Delta roll (rad)")
    parser.add_argument("--dp", type=float, default=0.0, help="Delta pitch (rad)")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Delta yaw (rad)")
    parser.add_argument("--rate", type=float, default=20.0, help="Control rate (Hz)")
    parser.add_argument("--maxt", type=float, default=8.0, help="Max duration (s)")
    parser.add_argument("--pos_tol", type=float, default=1e-3, help="Position tolerance (m)")
    parser.add_argument("--rot_tol", type=float, default=5e-3, help="Orientation tolerance (rad)")
    parser.add_argument("--teleop", action="store_true", help="Enable keyboard teleop mode")
    parser.add_argument("--lin_step", type=float, default=0.01, help="Teleop linear step (m)")
    parser.add_argument("--rot_step_deg", type=float, default=3.0, help="Teleop angular step (deg)")
    args = parser.parse_args()

    node = AR4ServoPID()

    # If --teleop is set or all deltas are zero, enter teleop
    if args.teleop or (args.dx == 0.0 and args.dy == 0.0 and args.dz == 0.0 and args.dr == 0.0 and args.dp == 0.0 and args.dyaw == 0.0):
        teleop_loop(
            node,
            linear_step=args.lin_step,
            angular_step=np.deg2rad(args.rot_step_deg),
            rate_hz=max(10.0, args.rate),
            max_duration=min(2.0, args.maxt),
        )
    else:
        ok = node.servo_to_delta(
            delta_pos=np.array([args.dx, args.dy, args.dz], dtype=float),
            delta_rpy=np.array([args.dr, args.dp, args.dyaw], dtype=float),
            pos_tol=args.pos_tol,
            rot_tol=args.rot_tol,
            rate_hz=args.rate,
            max_duration=args.maxt,
        )
        if not ok:
            node.get_logger().warn("Servo did not complete successfully")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
