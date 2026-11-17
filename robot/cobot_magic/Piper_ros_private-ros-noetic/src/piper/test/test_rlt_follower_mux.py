"""Exercise actual follower mux state/commands without ROS or CAN."""

import ast
from pathlib import Path
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class FollowerMuxTests(unittest.TestCase):
    def mux(self, stage2=True):
        clock = [1.0]
        stop = [False]
        pubs = {}
        def publisher(topic, *args, **kwargs):
            pubs[topic] = Mock()
            return pubs[topic]
        def message(**kwargs):
            return SimpleNamespace(header=SimpleNamespace(stamp=None), **kwargs)
        rospy = SimpleNamespace(
            get_param=lambda key, default: stage2 if key == '~rlt_stage2' else default,
            Publisher=publisher, Subscriber=Mock(), logwarn=Mock(), logerr=Mock(),
            loginfo=Mock(), logwarn_throttle=Mock(), logerr_throttle=Mock(),
            Time=SimpleNamespace(now=lambda: clock[0]),
            is_shutdown=lambda: stop[0],
            Rate=lambda hz: SimpleNamespace(sleep=lambda: stop.__setitem__(0, True)),
        )
        source = Path(__file__).parents[1] / 'scripts' / 'piper_policy_follower_mux.py'
        nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)]
        env = dict(rospy=rospy, math=math, threading=threading,
                   time=SimpleNamespace(monotonic=lambda: clock[0]),
                   Bool=message, Empty=message, String=message, JointState=message)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
        obj = env['PolicyFollowerMux']()
        for side in ('left', 'right'):
            obj._follower[side].update(message(position=[.1]*7))
            obj._policy[side].update(message(position=[.2]*7))
        obj._enable_callback(message(data=True))
        return obj, clock, stop, pubs, message

    def test_stage2_never_enables_before_i_even_with_policy(self):
        obj, _, _, pubs, _ = self.mux()
        obj.run()
        self.assertFalse(obj._active)
        self.assertFalse(pubs['/teleop/follower_enable'].publish.call_args.args[0].data)
        self.assertEqual(obj._home_target['left'], [.1]*7)

    def test_i_releases_seed_gate_and_captures_launch_home(self):
        obj, _, _, pubs, message = self.mux()
        obj._mode_callback(message(data='INFERENCE'))
        self.assertEqual(obj._published_mode, 'INFERENCE')
        self.assertFalse(obj._hold_active)
        obj.run()
        self.assertTrue(obj._active)
        self.assertEqual(obj._home_target['right'], [.1]*7)
        self.assertEqual(pubs['/teleop/command/joint_right'].publish.call_args.args[0].position, [.1]*7)
        self.assertTrue(pubs['/teleop/follower_enable'].publish.call_args.args[0].data)

    def test_home_uses_launch_pose_after_feedback_changes(self):
        obj, clock, stop, _, message = self.mux()
        obj._mode_callback(message(data='INFERENCE'))
        obj.run()
        clock[0] += .01
        for side in ('left', 'right'):
            obj._follower[side].update(message(position=[.3]*7))
        obj._hold_callback(message())
        obj._home_callback(message())
        self.assertTrue(obj._home_active)
        self.assertEqual(obj._home_target['left'], [.1]*7)
        stop[0] = False
        obj.run()
        self.assertTrue(all(.1 <= value < .3 for value in obj._last_output['left']))
        self.assertTrue(obj._hold_active)

    def test_resume_waits_for_new_policy_and_t_is_ignored(self):
        obj, clock, stop, _, message = self.mux()
        obj._mode_callback(message(data='INFERENCE'))
        obj.run()
        obj._hold_callback(message())
        clock[0] += .01
        obj._mode_callback(message(data='TAKEOVER'))
        self.assertTrue(obj._hold_active)
        obj._mode_callback(message(data='INFERENCE'))
        stop[0] = False
        obj.run()
        self.assertTrue(obj._resume_pending)
        clock[0] += .01
        for side in ('left', 'right'):
            obj._policy[side].update(message(position=[.2]*7))
        stop[0] = False
        obj.run()
        self.assertFalse(obj._resume_pending)

    def test_ordinary_openpi_keeps_automatic_start(self):
        obj, _, _, _, _ = self.mux(stage2=False)
        obj.run()
        self.assertTrue(obj._active)
        self.assertEqual(obj._published_mode, 'INFERENCE_FOLLOWERS')

    def test_feedback_gap_holds_without_estop_or_disable_in_all_stage2_phases(self):
        for phase in ('inference', 'resync', 'hold', 'home'):
            with self.subTest(phase=phase):
                obj, clock, stop, pubs, message = self.mux()
                obj._mode_callback(message(data='INFERENCE'))
                obj.run()
                if phase in ('resync', 'hold', 'home'):
                    obj._hold_callback(message())
                if phase == 'resync':
                    obj._mode_callback(message(data='INFERENCE'))
                if phase == 'home':
                    obj._home_callback(message())
                old = list(obj._last_output['right'])
                clock[0] += .2  # Beyond the existing 150-ms RX threshold.
                obj._follower['left'].update(message(position=[.1]*7))
                stop[0] = False
                obj.run()
                self.assertFalse(obj._emergency_stop)
                self.assertTrue(obj._active and obj._enable_requested)
                self.assertTrue(obj._hold_active)
                self.assertFalse(obj._home_active or obj._resume_pending)
                self.assertEqual(obj._last_output['right'], old)
                self.assertTrue(pubs['/teleop/follower_enable'].publish.call_args.args[0].data)
                self.assertFalse(pubs['/teleop/emergency_stop'].publish.call_args.args[0].data)
                self.assertEqual(pubs['/teleop/command/joint_right'].publish.call_args.args[0].position, old)
                obj._mode_callback(message(data='INFERENCE'))
                self.assertTrue(obj._hold_active)  # Stale I is refused.
                clock[0] += .01
                for side in ('left', 'right'):
                    obj._follower[side].update(message(position=[.1]*7))
                stop[0] = False
                obj.run()
                self.assertTrue(obj._hold_active)  # Recovery is not automatic.
                obj._mode_callback(message(data='INFERENCE'))
                self.assertTrue(obj._resume_pending)

    def test_explicit_emergency_request_still_stops_and_disables(self):
        obj, _, _, pubs, message = self.mux()
        obj._mode_callback(message(data='INFERENCE'))
        obj.run()
        obj._emergency_callback(message())
        self.assertTrue(obj._emergency_stop)
        self.assertFalse(pubs['/teleop/follower_enable'].publish.call_args.args[0].data)
        self.assertTrue(pubs['/teleop/emergency_stop'].publish.call_args.args[0].data)


if __name__ == '__main__':
    unittest.main()
