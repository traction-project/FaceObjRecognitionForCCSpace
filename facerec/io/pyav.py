# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
"""
Video I/O based on libav (via PyAV)
"""
import av
import numpy as np

from .abstract import FrameSource, FrameSink


class VideoSource(FrameSource):
    """
    An encoded video file frame source (e.g. "video.mp4")
    """
    def __init__(self, filename, **kwargs):
        self.video_stream = kwargs.pop('stream', 0)
        self.container = av.open(filename, **kwargs)
        self.frame_iterator = None

    def next(self):
        if self.frame_iterator is None:
            self.frame_iterator = self.container.decode(video=self.video_stream)
        try:
            frame = next(self.frame_iterator)
        except StopIteration:
            raise EOFError("no more frames")
        return frame.to_ndarray(format='rgb24')

    def close(self):
        self.container.close()
        
    def __len__(self):
        return self.container.streams.video[self.video_stream].frames


class VideoSink(FrameSink):
    """
    A sink for frames based on PyAV (libav wrapper)
    """
    def __init__(self, filename, codec='mpeg4', fps=30, pix_fmt='yuv420p'):
        self.container = av.open(filename, mode='w')
        self.stream = None
        self.fps = fps
        self.codec = codec
        self.pix_fmt = pix_fmt

    def write(self, frame):
        if self.container is None:
            raise IOError("video is closed")

        if self.stream is None:
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            self.stream.width = frame.shape[1]
            self.stream.height = frame.shape[0]
            self.stream.pix_fmt = self.pix_fmt

        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def flush(self):
        if self.stream is None or self.container is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)

    def close(self):
        self.flush()
        self.container.close()
        self.container = None


codecs_available = av.codecs_available