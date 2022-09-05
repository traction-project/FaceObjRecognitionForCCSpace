# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
"""
Video input using ffmpeg pipes (slow for larger videos)
"""
import sys
import os
import os.path
import numpy as np
import subprocess as sp
import json

from .abstract import FrameSource


# assumes ffmpeg is available on the path
if sys.platform == 'win32':
    ffmpeg_default_exe = 'ffmpeg.exe'
    ffprobe_default_exe = 'ffprobe.exe'
else:
    ffmpeg_default_exe = 'ffmpeg'
    ffprobe_default_exe = 'ffprobe'


class VideoSource(FrameSource):
    """
    Video input using ffmpeg pipes (slow for larger videos)
    """

    ffmpeg = ffmpeg_default_exe
    ffprobe = ffprobe_default_exe

    def __init__(self, filename):
        self.filename = filename
        self.metadata = get_video_metadata(filename, self.ffprobe)
        self.proc = None

        for index, stream in enumerate(self.metadata['streams']):
            if stream['codec_type'] == 'video':
                self.stream_index = index
                self.stream_metadata = stream
                break

        m = self.stream_metadata
        self.frame_shape = (m['height'], m['width'], 3)
        self.frame_size = int(np.prod(self.frame_shape))

    def _spawn_ffmpeg(self):
        command = make_ffmpeg_command(self.filename, self.stream_index+1, self.ffmpeg)
        self.proc = sp.Popen(
            command, 
            stdout=sp.PIPE, 
            stderr=sp.PIPE, 
            stdin=sp.DEVNULL, 
            bufsize=self.frame_size+100)

    def next(self):
        if self.proc is None:
            self._spawn_ffmpeg()
    
        frame = self.proc.stdout.read(self.frame_size)
        frame = np.fromstring(frame, dtype=np.uint8)
        if frame.size < self.frame_size:
            raise EOFError("no more frames")
        frame = frame.reshape(self.frame_shape)
        return frame

    def skip(self):
        if self.proc is None:
            self._spawn_ffmpeg()
        self.proc.stdout.read(self.frame_size)
    
    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None


def make_ffmpeg_command(filename, stream_index=1, ffmpeg=ffmpeg_default_exe):
    command = [
        ffmpeg,
        '-i', filename,
        '-ss', str(stream_index),
        '-loglevel', 'panic',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec','rawvideo', '-']
    return command


def get_video_metadata(filename, ffprobe=ffprobe_default_exe):
    """
    Use ffprobe to get video metdata.
    """
    command = [
        ffprobe,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', 
        '-show_format',
        filename]
    completed_proc = sp.run(command, capture_output=True)
    if completed_proc.returncode != 0:
        raise IOError("Error running ffprobe")
    data = json.loads(completed_proc.stdout)
    return data
