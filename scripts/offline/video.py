"""CPU software video decoding, including AV1 recordings unsupported by OpenCV."""

from __future__ import annotations


class VideoReader:
    """Decode exact frame order through PyAV; explicitly select dav1d for AV1."""

    def __init__(self, path: str):
        import av

        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        context = self.stream.codec_context
        if context.name == "av1":
            if "libdav1d" not in av.codecs_available:
                self.container.close()
                raise RuntimeError("AV1 recordings require a PyAV build containing libdav1d")
            context = av.CodecContext.create("libdav1d", "r")
            context.extradata = self.stream.codec_context.extradata
        context.thread_count = 2
        self.context = context
        self.frames = self._frames()

    def _frames(self):
        for packet in self.container.demux(self.stream):
            yield from self.context.decode(packet)

    def skip(self, count: int):
        """Skip decoded frames without RGB conversion; retain exact dataset indexing."""
        for _ in range(count):
            next(self.frames)

    def read(self):
        """Return one RGB uint8 frame, or None at the end of the recording."""
        frame = next(self.frames, None)
        return None if frame is None else frame.to_ndarray(format="rgb24")

    def close(self):
        """Release the demuxer and software decoder."""
        self.container.close()
