"""Exercise the shipped native client handoff code with no hardware transport."""

import ast
import importlib.util
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_inflight_result_is_discarded_and_fresh_observation_replanned():
    p = ROOT / "deps/openpi/examples/piper_real/hold_aware_policy.py"
    spec = importlib.util.spec_from_file_location("release_hold_policy", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env:
        generation = 1

        def wait_until_policy_execution_allowed(self):
            return self.generation

        def policy_execution_snapshot(self):
            return True, self.generation

        def get_observation(self):
            return {"epoch": self.generation}

    env = Env()

    class Policy:
        resets = 0
        seen = []

        def reset(self):
            self.resets += 1

        def infer(self, obs):
            self.seen.append(obs["epoch"])
            if len(self.seen) == 1:
                env.generation += 1
            return {"plan": len(self.seen)}

    policy = Policy()
    wrapper = module.HoldAwarePolicy(policy, env)
    result = wrapper.infer({"epoch": 0})
    assert result == {"plan": 2, "_piper_hold_generation": 2}
    assert policy.seen == [1, 2] and policy.resets == 2


def test_native_hold_ack_waits_for_publish_lock_and_clears_cached_targets():
    source = ROOT / "deps/openpi/examples/piper_real/env.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PiperRealEnvironment"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_hold_active_cb")
    space = {"Bool": lambda **kw: SimpleNamespace(**kw), "logging": logging}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), space)
    ack = threading.Event()
    obj = SimpleNamespace(
        _hold_condition=threading.Condition(),
        _command_lock=threading.Lock(),
        _hold_active=False,
        _hold_generation=1,
        _command_start="old",
        _command_target="old",
        _last_smoothed_action="old",
        _policy_hold_ack_pub=SimpleNamespace(publish=lambda msg: ack.set()),
    )
    obj._command_lock.acquire()
    thread = threading.Thread(
        target=space["_hold_active_cb"], args=(obj, SimpleNamespace(data=True))
    )
    thread.start()
    with obj._hold_condition:
        assert obj._hold_condition.wait_for(lambda: obj._hold_active, timeout=2)
    assert not ack.is_set()
    obj._command_lock.release()
    thread.join(2)
    assert not thread.is_alive() and ack.is_set()
    assert obj._hold_generation == 2 and obj._command_target is None
    assert obj._command_start is None and obj._last_smoothed_action is None
