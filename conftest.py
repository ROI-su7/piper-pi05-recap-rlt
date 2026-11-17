"""CPU test guard only: forbid real SDK/SocketCAN use; never imported by robot nodes."""

import importlib.util
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

if importlib.util.find_spec("rospy") is None:
    ros = ModuleType("rospy")
    ros.is_shutdown = lambda: False
    ros.Time = SimpleNamespace(now=lambda: 0)
    for name in [
        "loginfo",
        "logwarn",
        "logerr",
        "logwarn_throttle",
        "logerr_throttle",
        "logdebug",
        "on_shutdown",
    ]:
        setattr(ros, name, Mock())
    sys.modules["rospy"] = ros

    class Message:
        def __init__(self, *args, **kwargs):
            self.header = SimpleNamespace(stamp=None)
            self.position = []
            if args:
                self.data = args[0]
            self.__dict__.update(kwargs)

    for package, names in [
        ("sensor_msgs", ["JointState", "Image"]),
        ("std_msgs", ["Bool", "String", "Empty"]),
    ]:
        module = ModuleType(package)
        msg = ModuleType(package + ".msg")
        for name in names:
            setattr(msg, name, Message)
        module.msg = msg
        sys.modules[package] = module
        sys.modules[package + ".msg"] = msg

if importlib.util.find_spec("piper_sdk") is None:
    sdk = ModuleType("piper_sdk")

    class NoHardwareSDK:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Tests must not instantiate a real Piper SDK")

    sdk.C_PiperInterface_V2 = NoHardwareSDK
    sys.modules["piper_sdk"] = sdk


@pytest.fixture(autouse=True)
def prohibit_can_sockets(monkeypatch):
    original = socket.socket

    def guarded(family=socket.AF_INET, *args, **kwargs):
        if family == getattr(socket, "AF_CAN", 29):
            raise RuntimeError("SocketCAN is forbidden during offline tests")
        return original(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", guarded)
