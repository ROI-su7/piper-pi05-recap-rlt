#!/usr/bin/env python3
"""Fail-closed policy mux for the common two-follower inference session.

This node never opens or commands either leader CAN. /enable_flag only arms
the session; a private /teleop/follower_enable is sent to the two drivers only
after fresh left/right policy plus follower feedback are all present. The first
command is seeded from measured follower poses.
"""

import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String


class TimedJointState:
    def __init__(self):
        self.position = None
        self.received_at = 0.0

    def update(self, message):
        if len(message.position) < 7:
            rospy.logwarn_throttle(2.0, "Ignoring JointState with fewer than 7 positions")
            return
        values = [float(value) for value in message.position[:7]]
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(2.0, "Ignoring non-finite JointState")
            return
        self.position = values
        self.received_at = time.monotonic()

    def sample(self, now, timeout):
        if self.position is None:
            return None, False
        return list(self.position), 0.0 <= now - self.received_at <= timeout


class PolicyFollowerMux:
    def __init__(self):
        self.rlt_stage2 = bool(rospy.get_param("~rlt_stage2", False))
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.input_timeout = float(rospy.get_param("~input_timeout_s", 0.15))
        self.policy_timeout = float(
            rospy.get_param("~policy_input_timeout_s", 1.0)
        )
        self.startup_timeout = float(
            rospy.get_param("~startup_timeout_s", 10.0)
        )
        self.max_joint_velocity = float(
            rospy.get_param("~max_joint_velocity_rad_s", 1.5)
        )
        self.max_gripper_velocity = float(
            rospy.get_param("~max_gripper_velocity_m_s", 0.15)
        )
        self.home_max_joint_velocity = float(
            rospy.get_param("~home_max_joint_velocity_rad_s", 0.25)
        )
        self.home_max_gripper_velocity = float(
            rospy.get_param("~home_max_gripper_velocity_m_s", 0.03)
        )
        self.home_joint_tolerance = float(
            rospy.get_param("~home_joint_tolerance_rad", 0.04)
        )
        self.home_gripper_tolerance = float(
            rospy.get_param("~home_gripper_tolerance_m", 0.005)
        )
        self.home_stable_samples = int(
            rospy.get_param("~home_stable_samples", 20)
        )
        if min(
            self.publish_hz,
            self.input_timeout,
            self.policy_timeout,
            self.startup_timeout,
            self.max_joint_velocity,
            self.max_gripper_velocity,
            self.home_max_joint_velocity,
            self.home_max_gripper_velocity,
            self.home_joint_tolerance,
            self.home_gripper_tolerance,
            self.home_stable_samples,
        ) <= 0.0:
            raise ValueError("follower inference rates, timeouts and limits must be positive")

        self._lock = threading.RLock()
        self._policy = {"left": TimedJointState(), "right": TimedJointState()}
        self._follower = {"left": TimedJointState(), "right": TimedJointState()}
        self._last_output = {"left": None, "right": None}
        self._last_publish_time = time.monotonic()
        self._enable_requested = False
        self._enable_requested_at = 0.0
        self._active = False
        self._operator_active = True
        self._emergency_stop = False
        self._hold_active = self.rlt_stage2
        # HOLD -> I is a two-phase handshake.  The ROS controller keeps the
        # measured pose while the OpenPI client discards its old action chunk;
        # policy commands received before the I request are never replayed.
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._hold_target = {"left": None, "right": None}
        self._home_target = {"left": None, "right": None}
        self._home_active = False
        self._home_stable_count = 0
        self._published_mode = None

        self._command_pub = {
            side: rospy.Publisher(
                "/teleop/command/joint_{}".format(side),
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
            for side in ("left", "right")
        }
        # Preserve the existing collector's executed-action topics.  These are
        # the bounded commands actually selected for the followers, not raw
        # policy messages.
        self._action_pub = {
            side: rospy.Publisher(
                "/master/joint_{}".format(side),
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
            for side in ("left", "right")
        }
        self._mode_pub = rospy.Publisher(
            "/teleop/mode", String, queue_size=1, latch=True
        )
        self._inference_pub = rospy.Publisher(
            "/teleop/inference_active", Bool, queue_size=1, latch=True
        )
        self._emergency_pub = rospy.Publisher(
            "/teleop/emergency_stop", Bool, queue_size=1, latch=True
        )
        self._takeover_pub = rospy.Publisher(
            "/teleop/takeover_ready", Bool, queue_size=1, latch=True
        )
        self._hold_pub = rospy.Publisher(
            "/teleop/hold_active", Bool, queue_size=1, latch=True
        )
        self._home_pub = rospy.Publisher(
            "/teleop/home_active", Bool, queue_size=1, latch=True
        )
        # Follower drivers in start_inference_followers.launch subscribe to
        # this gated topic instead of the public /enable_flag. This prevents a
        # slow first model inference from enabling stationary arms and then
        # tripping the old 10-second startup E-STOP.
        self._follower_enable_pub = rospy.Publisher(
            "/teleop/follower_enable", Bool, queue_size=1, latch=True
        )

        for side in ("left", "right"):
            rospy.Subscriber(
                "/teleop/policy/joint_{}".format(side),
                JointState,
                self._joint_callback,
                callback_args=self._policy[side],
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                "/puppet/joint_{}".format(side),
                JointState,
                self._joint_callback,
                callback_args=self._follower[side],
                queue_size=1,
                tcp_nodelay=True,
            )
        rospy.Subscriber("/enable_flag", Bool, self._enable_callback, queue_size=1)
        rospy.Subscriber(
            "/teleop/emergency_stop_request",
            Empty,
            self._emergency_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/teleop/hold_request", Empty, self._hold_callback, queue_size=1
        )
        rospy.Subscriber(
            "/teleop/mode_request", String, self._mode_callback, queue_size=1
        )
        rospy.Subscriber(
            "/teleop/home_request", Empty, self._home_callback, queue_size=1
        )

        self._set_mode("WAITING_ENABLE")
        self._inference_pub.publish(Bool(data=False))
        self._emergency_pub.publish(Bool(data=False))
        self._takeover_pub.publish(Bool(data=False))
        self._hold_pub.publish(Bool(data=self._hold_active))
        self._home_pub.publish(Bool(data=False))
        self._follower_enable_pub.publish(Bool(data=False))
        rospy.loginfo(
            "Follower-only inference waiting for explicit enable, policy and feedback"
        )

    def _set_mode(self, mode):
        if self.rlt_stage2 and mode in ("INFERENCE_FOLLOWERS", "WAITING_POLICY_RESYNC"):
            mode = "INFERENCE"
        if mode != self._published_mode:
            self._published_mode = mode
            self._mode_pub.publish(String(data=mode))

    def _joint_callback(self, message, storage):
        with self._lock:
            storage.update(message)

    def _enable_callback(self, message):
        with self._lock:
            requested = bool(message.data)
            if requested and self._emergency_stop:
                rospy.logerr("Follower inference enable rejected: E-STOP is latched")
                return
            if requested and not self._operator_active:
                rospy.logerr("Follower inference enable rejected: session is stopped")
                return
            if requested == self._enable_requested:
                return
            self._enable_requested = requested
            self._active = False
            self._resume_pending = False
            self._resume_requested_at = 0.0
            self._inference_pub.publish(Bool(data=False))
            self._last_output = {"left": None, "right": None}
            if requested:
                self._enable_requested_at = time.monotonic()
                if self.rlt_stage2:
                    self._hold_active = True
                    self._hold_pub.publish(Bool(data=True))
                    self._set_mode("HOLD")
                else:
                    self._set_mode("WAITING_POLICY")
                rospy.logwarn(
                    "Follower inference armed; waiting for two policies and two feedback streams"
                )
            else:
                self._hold_active = False
                self._home_active = False
                self._hold_pub.publish(Bool(data=False))
                self._home_pub.publish(Bool(data=False))
                self._follower_enable_pub.publish(Bool(data=False))
                self._set_mode("WAITING_ENABLE")
                rospy.logwarn("Follower inference command output disabled")

    def _mode_callback(self, message):
        requested = message.data.strip().upper()
        with self._lock:
            if requested == "IDLE":
                self._enter_hold("IDLE request converted to safe HOLD")
            elif requested == "INFERENCE":
                if self._emergency_stop:
                    rospy.logerr("Cannot resume INFERENCE: hardware E-STOP is latched")
                elif self.rlt_stage2 and self._active and any(
                    not self._follower[side].sample(time.monotonic(), self.input_timeout)[1]
                    for side in ("left", "right")
                ):
                    self._feedback_timeout_hold("I rejected: follower feedback is stale")
                elif not self._enable_requested or (not self._active and not self.rlt_stage2):
                    rospy.logerr("Cannot resume INFERENCE before followers are active")
                else:
                    # Release the OpenPI pause gate, but continue publishing
                    # the HOLD target until both policy topics contain a
                    # command produced after this resume request.
                    self._hold_active = False
                    self._resume_pending = True
                    self._resume_requested_at = time.monotonic()
                    self._home_active = False
                    self._hold_pub.publish(Bool(data=False))
                    self._home_pub.publish(Bool(data=False))
                    self._set_mode("WAITING_POLICY_RESYNC")
                    self._inference_pub.publish(Bool(data=False))
                    rospy.logwarn(
                        "Follower-only I received; holding until a new post-HOLD policy chunk arrives"
                    )
            else:
                rospy.logwarn(
                    "Follower-only inference ignores %s; TAKEOVER requires the supervised launch",
                    requested,
                )

    def _home_callback(self, _message):
        with self._lock:
            if not self._enable_requested or not self._active:
                rospy.logerr("Cannot start HOME before followers are active")
                return
            if any(self._home_target[side] is None for side in ("left", "right")):
                rospy.logerr("Cannot start HOME: startup follower poses were not captured")
                return
            self._enter_hold("HOME requested")
            self._home_active = True
            self._home_stable_count = 0
            self._home_pub.publish(Bool(data=True))
            self._set_mode("HOME")
            rospy.logwarn("Follower HOME active; moving to startup poses at limited speed")

    def _hold_callback(self, _message):
        with self._lock:
            self._enter_hold("operator software HOLD")

    def _enter_hold(self, reason):
        now = time.monotonic()
        for side in ("left", "right"):
            measured, fresh = self._follower[side].sample(now, self.input_timeout)
            if fresh:
                self._hold_target[side] = measured
            elif self._last_output[side] is not None:
                self._hold_target[side] = list(self._last_output[side])
        self._hold_active = True
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._home_active = False
        self._home_stable_count = 0
        self._inference_pub.publish(Bool(data=False))
        self._hold_pub.publish(Bool(data=True))
        self._home_pub.publish(Bool(data=False))
        self._set_mode("HOLD")
        if self._enable_requested and self._active:
            rospy.logwarn("%s; followers remain enabled and position-held", reason)
        else:
            rospy.logwarn("%s; no enabled follower session exists yet", reason)

    def _emergency_callback(self, _message):
        with self._lock:
            self._latch_emergency("operator QUICK EMERGENCY STOP")

    def _feedback_timeout_hold(self, reason):
        """Stage 2 freezes targets without removing support torque on RX gaps."""
        if not self.rlt_stage2:
            self._latch_emergency(reason)
            return
        now = time.monotonic()
        ages = ", ".join(
            "{}={:.3f}s".format(side, now - self._follower[side].received_at)
            for side in ("left", "right")
        )
        if not self._hold_active or self._resume_pending or self._home_active:
            self._enter_hold(reason)
        rospy.logerr_throttle(
            1.0, "%s; feedback age [%s], limit=%.3fs; powered HOLD retained. "
            "Restore feedback, then press I; no automatic resume.",
            reason, ages, self.input_timeout,
        )

    def _latch_emergency(self, reason):
        if self._emergency_stop:
            return
        self._emergency_stop = True
        self._enable_requested = False
        self._active = False
        self._resume_pending = False
        self._resume_requested_at = 0.0
        self._operator_active = False
        self._inference_pub.publish(Bool(data=False))
        self._follower_enable_pub.publish(Bool(data=False))
        self._emergency_pub.publish(Bool(data=True))
        self._hold_pub.publish(Bool(data=False))
        self._home_pub.publish(Bool(data=False))
        self._set_mode("ESTOP")
        rospy.logerr("Follower-only inference stopped: %s", reason)

    def _samples(self, now):
        values = {"policy": {}, "follower": {}}
        freshness = {"policy": {}, "follower": {}}
        for side in ("left", "right"):
            values["policy"][side], fresh = self._policy[side].sample(
                now, self.policy_timeout
            )
            freshness["policy"][side] = fresh
            values["follower"][side], fresh = self._follower[side].sample(
                now, self.input_timeout
            )
            freshness["follower"][side] = fresh
        return values, freshness

    @staticmethod
    def _missing_streams(freshness):
        return [
            "{}_{}".format(group, side)
            for group in ("policy", "follower")
            for side in ("left", "right")
            if not freshness[group][side]
        ]

    def _bounded(self, side, target, dt, joint_velocity=None, gripper_velocity=None):
        previous = self._last_output[side]
        if previous is None:
            return list(target)
        result = []
        for index, (old, new) in enumerate(zip(previous, target)):
            velocity = (
                self.max_gripper_velocity if gripper_velocity is None else gripper_velocity
            ) if index == 6 else (
                self.max_joint_velocity if joint_velocity is None else joint_velocity
            )
            step = velocity * dt
            result.append(old + max(-step, min(step, new - old)))
        return result

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = list(position)
        return message

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            messages = None
            activate_followers = False
            now = time.monotonic()
            with self._lock:
                # Capture the launch-time follower poses before any model
                # command is released. H always returns to these positions.
                for side in ("left", "right"):
                    if self._home_target[side] is None:
                        position, fresh = self._follower[side].sample(
                            now, self.input_timeout
                        )
                        if fresh:
                            self._home_target[side] = position
                if (
                    self._operator_active
                    and self._enable_requested
                    and not self._emergency_stop
                ):
                    values, freshness = self._samples(now)
                    missing = self._missing_streams(freshness)
                    if not self._active:
                        if self.rlt_stage2 and not self._resume_pending:
                            # No policy motion or motor enable before I.
                            pass
                        elif not missing:
                            for side in ("left", "right"):
                                self._last_output[side] = list(
                                    values["follower"][side]
                                )
                            # Publish an exact measured-pose seed before the
                            # gated enable flag. Policy motion begins on the
                            # next loop iteration.
                            messages = {
                                side: self._message(values["follower"][side])
                                for side in ("left", "right")
                            }
                            self._last_publish_time = now
                            self._active = True
                            self._hold_active = False
                            self._hold_pub.publish(Bool(data=False))
                            self._resume_pending = False
                            activate_followers = True
                            self._set_mode("INFERENCE_FOLLOWERS")
                            self._inference_pub.publish(Bool(data=True))
                            rospy.logwarn(
                                "Follower-only inference active; leader CANs are not opened"
                            )
                        elif now - self._enable_requested_at > self.startup_timeout:
                            rospy.logwarn_throttle(
                                5.0,
                                "Follower inference still waiting; followers remain "
                                "disabled; missing streams: %s",
                                ", ".join(missing),
                            )
                    elif not self._hold_active and not self._resume_pending and any(
                        not freshness["follower"][side] for side in ("left", "right")
                    ):
                        self._feedback_timeout_hold("follower feedback became stale")
                    elif not self._hold_active and not self._resume_pending and any(
                        not freshness["policy"][side] for side in ("left", "right")
                    ):
                        # A model/network pause does not require Piper 0x150.
                        # Freeze the last measured pose and keep torque enabled.
                        self._enter_hold("policy input became stale")

                    if self._active and self._resume_pending:
                        follower_missing = any(
                            not freshness["follower"][side] for side in ("left", "right")
                        )
                        if follower_missing:
                            self._feedback_timeout_hold(
                                "follower feedback became stale while waiting for policy resync"
                            )
                        else:
                            policy_resynced = all(
                                freshness["policy"][side]
                                and self._policy[side].received_at
                                > self._resume_requested_at
                                for side in ("left", "right")
                            )
                            if policy_resynced:
                                self._resume_pending = False
                                self._set_mode("INFERENCE_FOLLOWERS")
                                self._inference_pub.publish(Bool(data=True))
                                rospy.logwarn(
                                    "Follower-only inference resumed with a fresh post-HOLD policy chunk"
                                )

                    if self._active and not activate_followers and (
                        self._hold_active or self._resume_pending
                    ):
                        follower_missing = any(
                            not freshness["follower"][side] for side in ("left", "right")
                        )
                        if follower_missing:
                            self._feedback_timeout_hold("follower feedback became stale during HOLD")
                        if not self._emergency_stop:
                            # Keep sending the frozen last valid targets even
                            # during an RX gap. Never advance HOME or policy.
                            dt = min(
                                0.05,
                                max(1.0 / self.publish_hz, now - self._last_publish_time),
                            )
                            self._last_publish_time = now
                            messages = {}
                            aligned = True
                            for side in ("left", "right"):
                                target = (
                                    self._home_target[side]
                                    if self._home_active
                                    else self._hold_target[side]
                                )
                                output = self._bounded(
                                    side,
                                    target,
                                    dt,
                                    self.home_max_joint_velocity if self._home_active else None,
                                    self.home_max_gripper_velocity if self._home_active else None,
                                )
                                self._last_output[side] = output
                                messages[side] = self._message(output)
                                if self._home_active:
                                    actual = values["follower"][side]
                                    if max(
                                        abs(a - b) for a, b in zip(actual[:6], target[:6])
                                    ) > self.home_joint_tolerance:
                                        aligned = False
                                    if abs(actual[6] - target[6]) > self.home_gripper_tolerance:
                                        aligned = False
                            if self._home_active:
                                self._home_stable_count = (
                                    self._home_stable_count + 1 if aligned else 0
                                )
                                if self._home_stable_count >= self.home_stable_samples:
                                    self._home_active = False
                                    for side in ("left", "right"):
                                        self._hold_target[side] = list(
                                            self._home_target[side]
                                        )
                                    self._home_pub.publish(Bool(data=False))
                                    self._set_mode("HOLD")
                                    rospy.logwarn("Follower HOME complete; remaining in HOLD")
                    elif self._active and not activate_followers:
                        dt = min(
                            0.05,
                            max(1.0 / self.publish_hz, now - self._last_publish_time),
                        )
                        self._last_publish_time = now
                        messages = {}
                        for side in ("left", "right"):
                            output = self._bounded(
                                side, values["policy"][side], dt
                            )
                            self._last_output[side] = output
                            messages[side] = self._message(output)

            if messages is not None:
                for side in ("left", "right"):
                    self._command_pub[side].publish(messages[side])
                    self._action_pub[side].publish(messages[side])
            if activate_followers:
                self._follower_enable_pub.publish(Bool(data=True))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("piper_policy_follower_mux")
    PolicyFollowerMux().run()
