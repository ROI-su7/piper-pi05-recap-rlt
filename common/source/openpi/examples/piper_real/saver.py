from __future__ import annotations

import logging
import pathlib
import queue
import threading

import imageio.v2 as imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Record I-to-HOLD segments, each with three cameras and a mosaic."""

    IMAGE_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, out_dir: pathlib.Path, *, fps: int, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._fps = max(1, fps)
        self._subsample = max(1, subsample)
        # Control events must never be dropped when the frame budget is full.
        self._queue: queue.Queue = queue.Queue()
        self._frame_slots = threading.BoundedSemaphore(32)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._recording = False
        self._step = 0
        self._dropped = 0
        self._error: Exception | None = None
        self._ros_subscriptions = []

    def attach_ros(self, rospy) -> None:
        """Observe keyboard/mux events; never publish a robot command."""
        from std_msgs.msg import Empty
        from std_msgs.msg import String

        self._ros_subscriptions = [
            rospy.Subscriber("/teleop/mode_request", String, self._mode_request, queue_size=10),
            rospy.Subscriber("/teleop/hold_request", Empty, self._hold_request, queue_size=10),
            rospy.Subscriber("/teleop/home_request", Empty, self._hold_request, queue_size=10),
        ]

    def _mode_request(self, message) -> None:
        if message.data.strip().upper() == "INFERENCE":
            self.start_recording()
        else:
            self.stop_recording()

    def _hold_request(self, _message) -> None:
        self.stop_recording()

    @override
    def on_episode_start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._write_frames, name="piper-video", daemon=True)
            self._thread.start()
        logging.info("Video ready: I starts a new recording; Space ends and saves it.")

    def start_recording(self) -> None:
        with self._lock:
            if self._thread is None or self._recording:
                return
            self._step = 0
            self._recording = True
            self._queue.put_nowait(("start", None))
        logging.info("Video recording started; Space saves this segment.")

    def stop_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._queue.put_nowait(("stop", None))
        logging.info("Video recording stopped; finalizing files in the background.")

    def _write_frames(self) -> None:
        writers = {}
        stem = None
        existing = list(self._out_dir.glob("episode_*.mp4"))
        indices = [int(p.stem.split("_")[1]) for p in existing if p.stem.split("_")[1].isdigit()]
        next_index = max(indices, default=-1) + 1

        def close_writers():
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    self._error = exc
                    logging.exception("Could not finalize a Piper video.")
            if writers:
                logging.info("Piper video segment saved: %s (four MP4s).", stem)
            writers.clear()

        try:
            while True:
                event, panels = self._queue.get()
                if event == "close":
                    break
                if event == "stop":
                    close_writers()
                    stem = None
                    continue
                if event == "start":
                    close_writers()
                    stem = f"episode_{next_index:04d}"
                    next_index += 1
                    self._error = None
                    continue
                try:
                    if stem is None or self._error is not None:
                        continue
                    if not writers:
                        paths = {name: self._out_dir / f"{stem}_{name}.mp4" for name in self.IMAGE_NAMES}
                        paths["combined"] = self._out_dir / f"{stem}.mp4"
                        for name, path in paths.items():
                            writers[name] = imageio.get_writer(path, fps=self._fps / self._subsample)
                            logging.info("Recording Piper video: %s", path)
                    for name, panel in zip(self.IMAGE_NAMES, panels, strict=True):
                        writers[name].append_data(panel)
                    writers["combined"].append_data(np.concatenate(panels, axis=1))
                except Exception as exc:
                    self._error = exc
                    logging.exception("Piper video recording failed; robot control continues.")
                finally:
                    self._frame_slots.release()
        finally:
            close_writers()

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        del action
        with self._lock:
            if not self._recording:
                return
            step = self._step
            self._step += 1
            if step % self._subsample:
                return
            if not self._frame_slots.acquire(blocking=False):
                self._dropped += 1
                if self._dropped == 1:
                    logging.warning("Video encoder is behind; dropping video frames to keep robot control responsive.")
                return
            try:
                panels = tuple(
                    np.transpose(observation["images"][name], (1, 2, 0)).copy() for name in self.IMAGE_NAMES
                )
                self._queue.put_nowait(("frame", panels))
            except Exception:
                self._frame_slots.release()
                raise

    @override
    def on_episode_end(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._recording = False
            self._queue.put_nowait(("close", None))
            self._thread = None
        thread.join()
        logging.info("Piper videos finalized (dropped frames: %d, error: %s).", self._dropped, self._error)
