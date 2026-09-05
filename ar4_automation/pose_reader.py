#!/usr/bin/env python3

import sys
import time
import warnings
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point, Quaternion, Pose
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler, euler_from_quaternion

import math

import tf2_ros
from rclpy.time import Time
from rclpy.duration import Duration

def quat_to_euler(x: float, y: float, z: float, w: float):
	roll, pitch, yaw = R.from_quat([x, y, z, w]).as_euler("XYZ", degrees=False)
	return roll, pitch, yaw

# moveit_msgs error codes -> readable names
_MOVEIT_ERR = {
	1: "SUCCESS", 99999: "FAILURE",
	-1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
	-3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE", -4: "CONTROL_FAILED",
	-5: "UNABLE_TO_AQUIRE_SENSOR_DATA", -6: "TIMED_OUT", -7: "PREEMPTED",
	-10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
	-12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS", -14: "GOAL_CONSTRAINTS_VIOLATED",
	-15: "INVALID_GROUP_NAME", -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
	-18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
	-21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
	-23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE", -25: "COMMUNICATION_FAILURE",
	-31: "NO_IK_SOLUTION",
}

def _moveit_err_str(moveit2):
	"""Readable error string for the last MoveIt execution, or '?'."""
	try:
		code = moveit2.get_last_execution_error_code()
		val = getattr(code, "val", code)
		return f"{_MOVEIT_ERR.get(val, 'UNKNOWN')}({val})"
	except Exception:
		return "?"

# patched local copy of pymoveit2's moveit2.py
from .moveit2 import MoveIt2

class PoseReader(Node):
	"""Node that tracks (and optionally prints) the gripper pose via pymoveit2."""

	def __init__(self, node_name: Optional[str] = None, enable_pose_print: bool = True,
	             robot: str = 'ar4'):
		super().__init__(node_name or "gripper_pose_reader")

		from .robot_config import get_robot_config
		self.robot = robot
		self.robot_config = get_robot_config(robot)
		joint_names = self.robot_config['joint_names']
		base_link_name = self.robot_config['base_link']
		end_effector_name = self.robot_config['end_effector_link']
		group_name = self.robot_config['move_group']

		self._cb_group = ReentrantCallbackGroup()
		self.moveit2 = MoveIt2(
			node=self,
			joint_names=joint_names,
			base_link_name=base_link_name,
			end_effector_name=end_effector_name,
			group_name=group_name,
			use_move_group_action=True,
			callback_group=self._cb_group,
		)

		# 0.0 makes MoveIt warn and fall back to 1.0
		self.moveit2.max_velocity = 0.9
		self.moveit2.max_acceleration = 0.9

		# pause after each move so TrajectoryExecutionManager releases before the next command
		self.move_settle_delay = 0.5


		self.base_link_name = base_link_name
		self.end_effector_name = end_effector_name

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		self._last_joint_msg = None  # list[float] ordered by self.moveit2.joint_names
		self._last_joint_update_monotonic = None
		self.simulation_mode = False
		self._xarm_state = None
		self._xarm_state_update_monotonic = None
		# ros2_control's /joint_states can stop updating while UFACTORY teach
		# mode owns the arm. RobotMsg.angle remains the controller's measured
		# joint feedback and is therefore the authoritative source for
		# hand-guided hand-eye captures.
		self._xarm_joint_msg = None
		self._xarm_joint_update_monotonic = None
		# Cache FK computed from direct xArm feedback.  This is intentionally
		# separate from the TF pose: robot_state_publisher can receive a stale
		# /joint_states stream while the physical arm is in teach mode.
		self._measured_fk_cache = None
		self._measured_fk_cache_joints = None
		self._measured_fk_cache_monotonic = None
		self._xarm_safety_configured = False
		self._xarm_safety_error = None
		self._trajectory_controller_active = None
		self._fk_future = None
		self.pose = np.array([-1,-1,-1,-1,-1,-1])
		self.quat = np.array([-1, -1, -1, -1])
		self.frame = ""

		self.enable_pose_print = enable_pose_print

		self.create_subscription(
			JointState,
			"joint_states",
			self._on_joint_states,
			10,
		)
		self._timer = self.create_timer(0.5, self._on_timer)

		self.get_logger().info(
			f"PoseReader started; base='{base_link_name}', eef='{end_effector_name}'"
		)
		# rotation from Bad Frame to Good Frame + neutral tool euler offset
		self.frameRotationAngles = self.robot_config['frame_rotation_angles']
		self.frameOffsetAngles = self.robot_config['frame_offset_angles']

		# publishes 'stop' to kill any in-flight trajectory before sending a new goal
		self._cancellation_pub = self.create_publisher(String, '/trajectory_execution_event', 1)
		self._init_xarm_safety_monitor()

	def _init_xarm_safety_monitor(self):
		"""Subscribe to an xarm_ros2 physical controller when configured."""
		if 'xarm_safety' not in self.robot_config:
			return
		try:
			from xarm_msgs.msg import RobotMsg
		except ImportError:
			# Gazebo-only workspaces may not install xarm_api/xarm_msgs.
			return
		namespace = self.robot_config['xarm_safety']['namespace'].rstrip('/')
		self.create_subscription(
			RobotMsg, f"{namespace}/robot_states",
			self._on_xarm_state, 10)
		# The driver publishes raw hardware joints under its namespace (for
		# example /xarm/joint_states). The global /joint_states is often
		# republished by MoveIt/robot_state_publisher and can freeze when teach
		# mode stops ros2_control ownership.
		self.create_subscription(
			JointState, f"{namespace}/joint_states",
			self._on_xarm_joint_states, 10)

	def _on_xarm_joint_states(self, msg):
		"""Keep UFACTORY driver's raw measured joints in planning-group order."""
		try:
			angles = [float(msg.position[msg.name.index(name)])
					  for name in self.moveit2.joint_names]
		except (ValueError, IndexError):
			return
		if len(angles) == len(self.moveit2.joint_names) and \
				np.isfinite(angles).all():
			self._xarm_joint_msg = angles
			self._xarm_joint_update_monotonic = time.monotonic()

	def _on_xarm_state(self, msg):
		self._xarm_state = {
			'state': int(msg.state),
			'mode': int(msg.mode),
			'error_code': int(msg.err),
			'warning_code': int(msg.warn),
			'motor_enabled_mask': int(msg.mt_able),
			'brake_mask': int(msg.mt_brake),
		}
		self._xarm_state_update_monotonic = time.monotonic()
		angles = list(getattr(msg, 'angle', []))
		if len(angles) == len(self.moveit2.joint_names) and \
				np.isfinite(angles).all():
			self._xarm_joint_msg = [float(value) for value in angles]
			self._xarm_joint_update_monotonic = time.monotonic()
		if msg.err:
			# Stop the active MoveIt trajectory as soon as the controller
			# reports a collision/fault. Recovery remains an explicit operator
			# action; this code never clears controller errors automatically.
			stop = String()
			stop.data = 'stop'
			self._cancellation_pub.publish(stop)

	def configure_xarm_safety(self, timeout=3.0):
		"""Apply non-motion controller safety settings for a physical UFACTORY arm.

		No errors are cleared and motors are not enabled here.  The operator
		must use UFACTORY's normal recovery/enable procedure after inspecting a
		collision or fault.
		"""
		if 'xarm_safety' not in self.robot_config or self.simulation_mode:
			self._xarm_safety_configured = True
			return True
		# The UFACTORY safety services briefly put the arm in a configuration
		# changed state.  Reapplying an already accepted profile on every press
		# of "Prepare" needlessly creates state=5/mode=0 races with MoveIt.
		# A fresh backend starts with False, so each new ROS session still applies
		# the profile once; this only makes repeated Prepare requests idempotent.
		if self._xarm_safety_configured:
			return True
		try:
			from xarm_msgs.srv import SetFloat32, SetInt16
		except ImportError as exc:
			self._xarm_safety_error = f"xarm_msgs unavailable: {exc}"
			return False
		cfg = self.robot_config['xarm_safety']
		ns = cfg['namespace'].rstrip('/')
		requests = (
			(f"{ns}/set_collision_sensitivity", SetInt16,
			 int(cfg['collision_sensitivity'])),
			(f"{ns}/set_self_collision_detection", SetInt16,
			 int(bool(cfg['self_collision_detection']))),
			(f"{ns}/set_reduced_max_tcp_speed", SetFloat32,
			 float(cfg['reduced_max_tcp_speed_mm_s'])),
			(f"{ns}/set_reduced_max_joint_speed", SetFloat32,
			 float(cfg['reduced_max_joint_speed_rad_s'])),
			(f"{ns}/set_reduced_mode", SetInt16,
			 int(bool(cfg['reduced_mode']))),
		)
		for service_name, service_type, value in requests:
			client = self.create_client(service_type, service_name)
			if not client.wait_for_service(timeout_sec=timeout):
				self._xarm_safety_error = f"service unavailable: {service_name}"
				return False
			req = service_type.Request()
			req.data = value
			future = client.call_async(req)
			deadline = time.monotonic() + timeout
			while not future.done() and time.monotonic() < deadline:
				time.sleep(0.01)
			if not future.done():
				self._xarm_safety_error = f"service timed out: {service_name}"
				return False
			response = future.result()
			if response is None or int(response.ret) != 0:
				ret = None if response is None else int(response.ret)
				self._xarm_safety_error = (
					f"{service_name} rejected safety setting (ret={ret})")
				return False

		# Applying reduced/self-collision settings deliberately leaves UFACTORY
		# controllers in CONFIG_CHANGED (state=5), and some controller firmware
		# also falls back to mode 0.  MoveIt trajectories require mode 1.  Restore
		# both only after this successful configuration sequence and fresh,
		# error-free telemetry; this does not clear errors, enable motors, or
		# initiate motion.
		state = self._xarm_state
		if state is None or state['error_code'] != 0:
			self._xarm_safety_error = (
				"refusing to restore controller state without fresh, error-free telemetry")
			return False
		try:
			self.set_xarm_mode(1, timeout=timeout)
		except Exception as exc:
			self._xarm_safety_error = f"could not restore ROS position mode: {exc}"
			return False
		self._xarm_safety_configured = True
		self._xarm_safety_error = None
		return True

	def set_xarm_mode(self, mode, timeout=3.0):
		"""Change UFACTORY mode without clearing faults or enabling motors."""
		if 'xarm_safety' not in self.robot_config or self.simulation_mode:
			raise RuntimeError('UFACTORY controller mode is physical-only')
		from xarm_msgs.srv import SetInt16
		ns = self.robot_config['xarm_safety']['namespace'].rstrip('/')
		for service, value in ((f'{ns}/set_mode', int(mode)),
		                       (f'{ns}/set_state', 0)):
			client = self.create_client(SetInt16, service)
			if not client.wait_for_service(timeout_sec=timeout):
				raise RuntimeError(f'service unavailable: {service}')
			req = SetInt16.Request()
			req.data = value
			future = client.call_async(req)
			deadline = time.monotonic() + timeout
			while not future.done() and time.monotonic() < deadline:
				time.sleep(0.01)
			if not future.done() or future.result() is None:
				raise RuntimeError(f'no response from {service}')
			if int(future.result().ret) != 0:
				raise RuntimeError(
					f'{service} rejected {value}: ret={future.result().ret}')
		return True

	def set_trajectory_controller_active(self, active, timeout=5.0):
		"""Activate/deactivate the arm trajectory controller via ros2_control."""
		from controller_manager_msgs.srv import ListControllers, SwitchController
		controller = self.robot_config.get('trajectory_controller')
		if not controller:
			raise RuntimeError('trajectory controller is not configured')
		list_client = self.create_client(
			ListControllers, '/controller_manager/list_controllers')
		if not list_client.wait_for_service(timeout_sec=timeout):
			raise RuntimeError(
				'service unavailable: /controller_manager/list_controllers')
		listed = list_client.call_async(ListControllers.Request())
		deadline = time.monotonic() + timeout
		while not listed.done() and time.monotonic() < deadline:
			time.sleep(0.01)
		if not listed.done() or listed.result() is None:
			raise RuntimeError('timed out listing ros2_control controllers')
		current = next(
			(c.state for c in listed.result().controller
			 if c.name == controller), None)
		if current is None:
			raise RuntimeError(f'controller is not loaded: {controller}')
		is_active = current == 'active'
		self._trajectory_controller_active = is_active
		if is_active == bool(active):
			return True

		client = self.create_client(
			SwitchController, '/controller_manager/switch_controller')
		if not client.wait_for_service(timeout_sec=timeout):
			raise RuntimeError(
				'service unavailable: /controller_manager/switch_controller')
		req = SwitchController.Request()
		if active:
			req.activate_controllers = [controller]
		else:
			req.deactivate_controllers = [controller]
		req.strictness = 2  # STRICT
		req.activate_asap = True
		req.timeout.sec = int(timeout)
		future = client.call_async(req)
		deadline = time.monotonic() + timeout
		while not future.done() and time.monotonic() < deadline:
			time.sleep(0.01)
		if not future.done() or future.result() is None:
			raise RuntimeError(
				f'timed out switching controller {controller}')
		if not bool(future.result().ok):
			raise RuntimeError(
				f'controller_manager rejected {"activation" if active else "deactivation"} '
				f'of {controller}')
		self._trajectory_controller_active = bool(active)
		return True

	def enter_teach_mode(self):
		"""Safely hand trajectory ownership from MoveIt to manual guidance."""
		self.set_trajectory_controller_active(False)
		try:
			return self.set_xarm_mode(2)
		except Exception:
			# Best-effort rollback: do not leave a failed teach transition with
			# the normal trajectory controller unnecessarily stopped.
			try:
				self.set_trajectory_controller_active(True)
			except Exception:
				pass
			raise

	def exit_teach_mode(self):
		"""Restore UFACTORY position mode, then return ownership to MoveIt."""
		self.set_xarm_mode(1)
		return self.set_trajectory_controller_active(True)

	def safety_snapshot(self):
		"""Return machine-readable interlocks for the GUI and command guard."""
		now = time.monotonic()
		profile_name = 'simulation_motion' if self.simulation_mode else 'physical_motion'
		profile = self.robot_config.get(profile_name, {})
		checks = []
		joint_age = (
			None if self._last_joint_update_monotonic is None
			else now - self._last_joint_update_monotonic)
		max_age = float(profile.get('joint_state_max_age', 2.0))
		checks.append({
			'name': 'joint_states',
			'ok': joint_age is not None and joint_age <= max_age,
			'detail': 'not received' if joint_age is None else f"age={joint_age:.2f}s",
		})
		if 'xarm_safety' in self.robot_config and not self.simulation_mode:
			state_age = (
				None if self._xarm_state_update_monotonic is None
				else now - self._xarm_state_update_monotonic)
			state = self._xarm_state
			checks.extend([
				{
					'name': 'xarm_state_stream',
					'ok': state is not None and state_age is not None and state_age <= 1.0,
					'detail': 'not received' if state_age is None else f"age={state_age:.2f}s",
				},
				{
					'name': 'xarm_controller',
					# RUNNING(1) and SLEEPING/ready(2) can accept planned motion.
					'ok': state is not None and state['state'] in (1, 2)
					      and state['mode'] == 1 and state['error_code'] == 0,
					'detail': 'unknown' if state is None else (
						f"state={state['state']} mode={state['mode']} "
						f"err={state['error_code']} warn={state['warning_code']}"),
				},
				{
					'name': 'trajectory_controller',
					'ok': self._trajectory_controller_active is True,
					'detail': (
						f"{self.robot_config.get('trajectory_controller')} "
						f"{'active' if self._trajectory_controller_active else 'inactive/unknown'}"),
				},
				{
					'name': 'controller_safety_profile',
					'ok': self._xarm_safety_configured,
					'detail': self._xarm_safety_error or (
						'reduced mode + controller collision protections applied'),
				},
			])
		return {
			'ready': all(c['ok'] for c in checks),
			'profile': 'simulation' if self.simulation_mode else 'physical',
			'checks': checks,
			'xarm_state': self._xarm_state,
		}

	def assert_motion_safe(self):
		safety = self.safety_snapshot()
		if not safety['ready']:
			failed = [c for c in safety['checks'] if not c['ok']]
			raise RuntimeError(
				"motion blocked by safety preflight: " +
				"; ".join(f"{c['name']} ({c['detail']})" for c in failed))

	def validate_joint_target(self, joint_positions, manual=False):
		if len(joint_positions) != len(self.moveit2.joint_names):
			raise ValueError("joint target must contain exactly 6 values")
		if not np.all(np.isfinite(joint_positions)):
			raise ValueError("joint target contains NaN or infinity")
		for name, value, limits in zip(
				self.moveit2.joint_names, joint_positions,
				self.robot_config.get('joint_limits', [])):
			if not limits[0] <= value <= limits[1]:
				raise ValueError(
					f"{name} target {value:.3f} rad is outside safe "
					f"range [{limits[0]:.3f}, {limits[1]:.3f}]")
		if manual and self._last_joint_msg is not None:
			profile_name = 'simulation_motion' if self.simulation_mode else 'physical_motion'
			max_delta = self.robot_config.get(profile_name, {}).get(
				'max_manual_joint_delta')
			if max_delta is not None:
				for name, current, target in zip(
						self.moveit2.joint_names, self._last_joint_msg, joint_positions):
					if abs(target - current) > max_delta:
						raise ValueError(
							f"{name} step is {abs(target-current):.3f} rad; "
							f"manual limit is {max_delta:.3f} rad")
		return True

	def validate_pose_target(self, good_position):
		position = np.asarray(good_position, dtype=float)
		if position.shape != (3,) or not np.all(np.isfinite(position)):
			raise ValueError("Cartesian target must contain three finite values")
		workspace = self.robot_config.get('workspace')
		if not workspace:
			return True
		for index, axis in enumerate(('x', 'y', 'z')):
			low, high = workspace[axis]
			if not low <= position[index] <= high:
				raise ValueError(
					f"{axis}={position[index]:.3f} m is outside safe "
					f"workspace [{low:.3f}, {high:.3f}]")
		radius = float(np.linalg.norm(position))
		low, high = workspace['radius']
		if not low <= radius <= high:
			raise ValueError(
				f"target radius {radius:.3f} m is outside safe "
				f"workspace [{low:.3f}, {high:.3f}]")
		return True

	def _cancel_and_wait(self, wait_timeout=3.0):
		"""Cancel any in-flight MoveIt trajectory and wait for it to go idle."""
		_stop = String()
		_stop.data = 'stop'
		self._cancellation_pub.publish(_stop)
		# let the result callback clear __is_executing before the next goal arrives
		_deadline = time.time() + wait_timeout
		while (getattr(self.moveit2, '_MoveIt2__is_executing', False) or
		       getattr(self.moveit2, '_MoveIt2__is_motion_requested', False)):
			if time.time() > _deadline:
				break
			time.sleep(0.05)
		# force-clear in case the server never sent a result (hard timeout, restart)
		self.moveit2._MoveIt2__is_motion_requested = False
		self.moveit2._MoveIt2__is_executing = False
		# give TrajectoryExecutionManager a beat to release the lock
		time.sleep(0.3)

	def _reached_configuration(self, joint_positions, tol=0.10):
		"""True if measured joints are within tol rad of target.

		Ground-truth completion check, independent of pymoveit2's result callback.
		tol=0.10 rad matches the controller's loosest goal tolerance (J6)."""
		actual = self._last_joint_msg
		if actual is None or len(actual) != len(joint_positions):
			return False
		return all(abs(a - t) <= tol for a, t in zip(actual, joint_positions))

	def _calibration_joint_positions(self):
		"""Return fresh measured joints and their source for hand-eye capture."""
		if (self._xarm_joint_msg is not None and
				self._xarm_joint_update_monotonic is not None and
				time.monotonic() - self._xarm_joint_update_monotonic <= 1.0):
			return list(self._xarm_joint_msg), 'xarm_robot_states'
		if self._last_joint_msg is not None:
			return list(self._last_joint_msg), 'joint_states'
		return None, None

	def _eef_pose_from_joint_state(self, joint_positions=None):
		"""Compute base->EEF from the latest measured joint state.

		This deliberately bypasses TF. During physical teach mode a stale
		robot_state_publisher transform can remain available even after the arm
		has been hand-guided, which is unsafe for hand-eye samples.
		"""
		joint_positions = (list(joint_positions) if joint_positions is not None
						   else self._last_joint_msg)
		if not joint_positions:
			return (None, None)
		try:
			js = JointState()
			js.name = list(self.moveit2.joint_names)
			js.position = list(joint_positions)
			# This node already belongs to a background executor. The
			# synchronous pymoveit2 helper calls rclpy.spin_once(node), which
			# raises when a node is already attached to another executor.
			# Submit asynchronously and let that existing executor complete it.
			future = self.moveit2.compute_fk_async(
				joint_state=js, fk_link_names=[self.end_effector_name])
			if future is None:
				return (None, None)
			deadline = time.monotonic() + 3.0
			while not future.done() and time.monotonic() < deadline:
				time.sleep(0.01)
			if not future.done():
				self.get_logger().warn(
					"FK fallback timed out waiting for /compute_fk",
					throttle_duration_sec=5.0)
				return (None, None)
			ps = self.moveit2.get_compute_fk_result(
				future, fk_link_names=[self.end_effector_name])
		except Exception:
			return (None, None)
		if isinstance(ps, list):
			ps = ps[0] if ps else None
		if ps is None:
			return (None, None)
		# Only trust FK expressed in the base frame we compare against.
		frame = (ps.header.frame_id or self.base_link_name).lstrip('/')
		if frame != self.base_link_name.lstrip('/'):
			return (None, None)
		p = ps.pose.position
		q = ps.pose.orientation
		return (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))

	def _eef_pose_from_measured_joints(self, cache_seconds=0.20):
		"""Return base->EEF from fresh measured feedback, rather than TF.

		For the physical xArm this keeps vision mapping truthful while Teach
		Mode owns the arm and the regular robot_state_publisher TF can lag or
		freeze.  FK is cached briefly because a single camera frame can contain
		several detected markers.
		"""
		joints, source = self._calibration_joint_positions()
		if joints is None:
			return None, None, None
		now = time.monotonic()
		if (self._measured_fk_cache is not None and
				self._measured_fk_cache_joints is not None and
				now - (self._measured_fk_cache_monotonic or 0.0) <= cache_seconds and
				np.allclose(joints, self._measured_fk_cache_joints,
						atol=1e-5, rtol=0.0)):
			pos, quat = self._measured_fk_cache
			return pos.copy(), quat.copy(), source
		pos, quat = self._eef_pose_from_joint_state(joints)
		if pos is None or quat is None:
			return None, None, source
		self._measured_fk_cache = (pos.copy(), quat.copy())
		self._measured_fk_cache_joints = list(joints)
		self._measured_fk_cache_monotonic = now
		return pos, quat, source

	def _eef_pose_truth(self):
		"""link_6 pose in base_link as (pos, quat_xyzw), or (None, None).
		TF first, FK fallback for ordinary motion-completion checks."""
		try:
			tf = self.tf_buffer.lookup_transform(
				self.base_link_name, self.end_effector_name, Time(),
				timeout=Duration(seconds=0.1))
			t = tf.transform.translation
			r = tf.transform.rotation
			return (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
		except Exception:
			return self._eef_pose_from_joint_state()

	def _reached_pose(self, target_pos, target_quat, pos_tol=0.025, ang_tol=0.17):
		"""Ground-truth check that link_6 is within tolerance of the target.
		Deliberately loose (~2.5 cm / ~10 deg): the joint goal tolerances leave
		that much slack on a settled move, tighter would false-fail."""
		cur_pos, cur_q = self._eef_pose_truth()
		if cur_pos is None:
			return False
		if np.linalg.norm(cur_pos - np.asarray(target_pos, dtype=float)) > pos_tol:
			return False
		tq = np.asarray(target_quat, dtype=float)
		# Angle between orientations; |dot| handles the q/-q double cover.
		dot = min(1.0, abs(float(np.dot(cur_q, tq))))
		return (2.0 * np.arccos(dot)) <= ang_tol

	def move_to_configuration(self, joint_positions, timeout=15.0, max_retries=2):
		"""Joint-space move with retries. Polls with a deadline instead of
		wait_until_executed (which has none), and counts the move done once the
		arm physically reaches the config, so a missed result callback can't
		turn a completed move into a timeout."""
		self.assert_motion_safe()
		self.validate_joint_target(joint_positions)
		for attempt in range(max_retries + 1):
			if attempt > 0:
				# a "failed" attempt may actually have arrived (missed result
				# callback); cancelling a completed move is what desyncs the arm
				if self._reached_configuration(joint_positions):
					time.sleep(self.move_settle_delay)
					return True
				self.get_logger().warn(
					f"[move_to_configuration] Retry {attempt}/{max_retries}…"
				)

			self._cancel_and_wait()
			self.moveit2.motion_suceeded = False

			self.moveit2.move_to_configuration(joint_positions=joint_positions)

			_deadline = time.time() + timeout
			timed_out = False
			while (getattr(self.moveit2, '_MoveIt2__is_motion_requested', False) or
			       getattr(self.moveit2, '_MoveIt2__is_executing', False)):
				# ground truth: arm reached the config. A missed result callback
				# leaves the flags stuck set, which would read as a false timeout.
				if self._reached_configuration(joint_positions):
					time.sleep(self.move_settle_delay)
					return True
				if time.time() > _deadline:
					self.get_logger().error(
						f"[move_to_configuration] timed out after {timeout}s."
					)
					timed_out = True
					self._cancel_and_wait()
					break
				time.sleep(0.05)

			# only ground truth counts; motion_suceeded is racy both ways
			if self._reached_configuration(joint_positions):
				time.sleep(self.move_settle_delay)
				return True

			reason = f"timed out after {timeout}s" if timed_out else "motion aborted/failed"
			err = _moveit_err_str(self.moveit2)
			if attempt < max_retries:
				self.get_logger().warn(
					f"[move_to_configuration] {reason} (MoveIt err={err}) on attempt {attempt + 1} — retrying…"
				)
			else:
				self.get_logger().error(
					f"[move_to_configuration] {reason} (MoveIt err={err}) — all attempts exhausted."
				)

		return False

	def _solve_pose_ik(self, bad_pos, quat, timeout=5.0):
		"""Resolve a Cartesian target to the arm's ordered joint vector.

		The pymoveit2 synchronous IK helper tries to spin this node itself,
		which is invalid when the GUI's background executor already owns it.
		Use the async service and let that executor deliver the response.
		"""
		if not self._last_joint_msg:
			self.get_logger().error("IK unavailable: no current joint state")
			return None
		try:
			future = self.moveit2.compute_ik_async(
				position=Point(
					x=float(bad_pos[0]), y=float(bad_pos[1]),
					z=float(bad_pos[2])),
				quat_xyzw=Quaternion(
					x=float(quat[0]), y=float(quat[1]),
					z=float(quat[2]), w=float(quat[3])),
				ik_link_name=self.end_effector_name,
				start_joint_state=list(self._last_joint_msg),
				wait_for_server_timeout_sec=1.0,
			)
			if future is None:
				return None
			deadline = time.monotonic() + timeout
			while not future.done() and time.monotonic() < deadline:
				time.sleep(0.01)
			if not future.done():
				self.get_logger().error(
					f"IK timed out after {timeout:.1f}s")
				return None
			solution = self.moveit2.get_compute_ik_result(future)
			if solution is None:
				return None
			names = list(solution.name)
			return [
				float(solution.position[names.index(joint)])
				for joint in self.moveit2.joint_names
			]
		except (ValueError, RuntimeError) as exc:
			self.get_logger().error(f"IK solution invalid: {exc}")
			return None

	def move_to_pose(self, pos, euler, max_retries=1, timeout=12.0):
		"""Move to a Cartesian pose through collision-aware joint planning.

		IK is seeded from the measured joint state, then the resulting joints
		use the same MoveGroup path already proven for joint goals. This avoids
		the flaky direct pose-action state machine while remaining portable to
		the physical robot (the controller still receives a joint trajectory).
		"""
		self.assert_motion_safe()
		self.validate_pose_target(pos)
		bad_pos, bad_euler = self.to_bad_frame(pos, euler)
		q = R.from_euler("XYZ", bad_euler, degrees=False).as_quat()  # [x, y, z, w]
		self.get_logger().warn(
			f"[move_to_pose] IK target in base_link: "
			f"pos=[{bad_pos[0]:.4f}, {bad_pos[1]:.4f}, {bad_pos[2]:.4f}] "
			f"quat=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
		)
		joint_target = self._solve_pose_ik(bad_pos, q)
		if joint_target is None:
			self.get_logger().error(
				"[move_to_pose] no IK solution for requested pose")
			return False
		self.validate_joint_target(joint_target)
		self.get_logger().info(
			"[move_to_pose] IK joints deg=" +
			str(np.round(np.degrees(joint_target), 2).tolist()))
		return self.move_to_configuration(
			joint_target, timeout=timeout, max_retries=max_retries)


	def to_good_frame(self, bad_position, bad_euler_angles):
		# Transformation from Bad Frame to Good Frame (BF to GF)

		R_BF_GF_Vec = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
		R_BF_GF = R_BF_GF_Vec.as_matrix()
		H_BF_GF = np.eye(4)
		H_BF_GF[:3, :3] = R_BF_GF

		# Create rotation matrix from Euler angles
		RBadFrameVec = R.from_euler("XYZ", bad_euler_angles, degrees=False)
		RBadFrame = RBadFrameVec.as_matrix()
		HBadFrame = np.eye(4)
		HBadFrame[:3, :3] = RBadFrame
		HBadFrame[:3, 3] = bad_position

		HGoodFrame = H_BF_GF @ HBadFrame
		good_position = HGoodFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ("XYZ" order)
		good_euler_angles_vec = R.from_matrix(HGoodFrame[:3, :3])


		good_euler_angles = good_euler_angles_vec.as_euler("XYZ", degrees=False)
		good_euler_angles -= self.frameOffsetAngles

		return good_position, good_euler_angles

	def to_bad_frame(self, good_position, good_euler_angles):
		"""Inverse of to_good_frame."""
		# Inverse rotation from Good Frame to Bad Frame
		R_BF_GF_Vec = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
		R_GF_BF = R_BF_GF_Vec.as_matrix().T
		H_GF_BF = np.eye(4)
		H_GF_BF[:3, :3] = R_GF_BF

		# Create rotation matrix from Euler angles in Good Frame
		# Add back the offset angles that were subtracted in to_good_frame
		good_euler_angles_corrected = good_euler_angles + self.frameOffsetAngles
		RGoodFrameVec = R.from_euler("XYZ", good_euler_angles_corrected, degrees=False)
		RGoodFrame = RGoodFrameVec.as_matrix()
		HGoodFrame = np.eye(4)
		HGoodFrame[:3, :3] = RGoodFrame
		HGoodFrame[:3, 3] = good_position

		# Apply inverse transformation
		HBadFrame = H_GF_BF @ HGoodFrame
		bad_position = HBadFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ("XYZ" order)
		bad_euler_angles_vec = R.from_matrix(HBadFrame[:3, :3])
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", UserWarning)
			bad_euler_angles = bad_euler_angles_vec.as_euler("XYZ", degrees=False)
		return bad_position, bad_euler_angles


	def _on_joint_states(self, msg: JointState):
		# Store joints mapped to the planning group order
		self.jointAngles = msg.position[2:8]
		self.linkNames = msg.name[2:8]

		try:
			self._last_joint_msg = [
				float(msg.position[msg.name.index(j)]) for j in self.moveit2.joint_names
			]
			self._last_joint_update_monotonic = time.monotonic()
		except ValueError:
			# Missing joints in this message; skip
			return
	
	def get_frame(self, frame=None):
		# Use base_link consistently with MoveIt. TF is preferred inside
		# _eef_pose_truth; FK provides a safe fallback when the web backend's
		# TF listener starts after robot_state_publisher.
		bad_pos, quat = self._eef_pose_truth()
		if bad_pos is None:
			self.get_logger().warn(
				f"End-effector pose unavailable for {frame or self.end_effector_name}",
				throttle_duration_sec=5.0)
			return self.pose

		roll, pitch, yaw = quat_to_euler(*quat)
		good_pos, good_euler = self.to_good_frame(
			np.asarray(bad_pos), np.array([roll, pitch, yaw]))
		self.quat = np.asarray(quat, dtype=float)
		self.frame = self.base_link_name
		return np.concatenate((good_pos, good_euler))

	def get_fk(self):
		# Synchronous FK via MoveIt2.compute_fk()
		js = JointState()
		js.name = list(self.moveit2.joint_names)
		js.position = list(self._last_joint_msg)
		pose_stamped = self.moveit2.compute_fk(
			joint_state=js,
			fk_link_names=[self.end_effector_name],
		)
		if pose_stamped is None:
			self.get_logger().warn("FK failed or returned empty result")
			return
		if isinstance(pose_stamped, list):
			pose_stamped = pose_stamped[0] if pose_stamped else None
			if pose_stamped is None:
				self.get_logger().warn("FK returned empty list")
				return

		p = pose_stamped.pose.position
		q = pose_stamped.pose.orientation
		# Compute relative orientation to hardcoded home quaternion so home is (0,0,0)
		qx, qy, qz, qw = q.x, q.y, q.z, q.w
		
		roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)
		frame = pose_stamped.header.frame_id or self.base_link_name

		self.pose = np.array([p.x, p.y, p.z, roll, pitch, yaw])
		self.quat = np.array([qx, qy, qz, qw])
		self.frame = frame
		#print("Computed fk (sync)")


	def _on_timer(self):
		# Ensure joint states have been received at least once
		if not self._last_joint_msg:
			self.get_logger().warn("Waiting for joint_states...")
			return
		# Always update pose
		self.pose = self.get_frame()
		# Only print if enabled
		if self.enable_pose_print:
			print(
				f"[GripperPose] frame={self.frame} pos=({self.pose[0]:.4f}, {self.pose[1]:.4f}, {self.pose[2]:.4f}) "
				f"quat=({self.quat[0]:.4f}, {self.quat[1]:.4f}, {self.quat[2]:.4f}, {self.quat[3]:.4f}) "
				f"rpy=({self.pose[3]:.4f}, {self.pose[4]:.4f}, {self.pose[5]:.4f})"
			)


def main(argv=None):
	rclpy.init(args=argv)
	node = PoseReader()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)
