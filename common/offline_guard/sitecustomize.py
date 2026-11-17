"""Optional per-process SocketCAN prohibition for model-only offline validation."""

import os
import socket

if os.environ.get("PIPER_OFFLINE_VALIDATION") == "1":
    _original_socket = socket.socket

    class _OfflineSocket(_original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, "AF_CAN", 29):
                raise RuntimeError("SocketCAN is prohibited in offline validation")
            super().__init__(family, *args, **kwargs)

    socket.socket = _OfflineSocket
