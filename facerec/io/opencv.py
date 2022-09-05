# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
"""
OpenCV based image and video I/O
"""
import cv2
import os
import os.path

from .abstract import FrameSource, FrameSink
from .util import findfiles


class CaptureFrameSource(FrameSource):
    """
    A frame source based on OpenCV VideoCapture. Produces RGB video frames.
    """
    def __init__(self, filename_or_index):
        self.cap = cv2.VideoCapture(filename_or_index)
        if not self.cap.isOpened():
            raise IOError(f"unable to open video: {filename_or_index}")

    def skip(self):
        # use grab instead of read beacuse it is faster
        if not self.cap.grab():
            raise EOFError("no frames available")

    def next(self):
        retval, im = self.cap.read()
        if not retval:
            raise EOFError("no frames available")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class VideoSource(CaptureFrameSource):
    """
    An encoded video file frame source (e.g. "video.mp4")
    """
    def __init__(self, filename):
        super().__init__(filename)

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


class CameraSource(CaptureFrameSource):
    """
    Attached webcam video frame source.
    """
    def __init__(self, index=0):
        super().__init__(index)


class FileListSource(FrameSource):
    """
    A frame source based on a list of image files. 
    """
    def __init__(self, files):
        self.files = files
        self.index = 0
        self.readimage = readimage

    def skip(self):
        if self.index >= len(self.files):
            raise EOFError("no frames available")
        self.index += 1

    def next(self):
        if self.index >= len(self.files):
            raise EOFError("no frames available")
        filename = self.files[self.index]
        im = self.readimage(filename)
        self.index += 1
        return im

    def __len__(self):
        return len(self.files)


class FolderSource(FileListSource):
    """
    A frame source based on a folder of images. The frames are assumed to be 
    sorted by name in lexicographical order.
    """
    def __init__(self, path, suffix=".jpg"):
        self.path = path
        super().__init__(findfiles(path, suffix))


def folder_or_video_source(path, **kwargs):
    """
    Open either a folder or video source depending on whether the path is 
    a file or a folder.
    """
    if os.path.isdir(path):
        return FolderSource(path, **kwargs)
    elif os.path.isfile(path):
        return VideoSource(path, **kwargs)
    else:
        raise IOError("file not found")


def adaptive_source(path, **kwargs):
    """
    Open either a folder or video or camera depending on whether the path is 
    convertable to an int, is a file, or is a folder a file or a folder.
    """
    try:
        return folder_or_video_source(path, **kwargs)
    except IOError as e:
        try:
            path = int(path)
        except ValueError:
            raise e
        else:
            return CameraSource(path)


class VideoSink(FrameSink):
    """
    A sink for frames based on OpenCV's VideoWriter.
    """
    def __init__(self, filename, fmt='mp4v', fps=30):
        self.filename = filename
        self.fmt = fmt
        self.fps = fps
        self.writer = None
    
    def write(self, frame):
        height, width = frame.shape[:2]
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.fmt)
            self.writer = cv2.VideoWriter(
                self.filename, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise IOError(f"could not open video writer: {self.filename}")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()


class FolderSink(FrameSink):
    """
    A frame sink that writes frames to a folder of images.
    """

    def __init__(self, path, filename_fmt="f{:010d}.jpg", create=False):
        if not os.path.isdir(path):
            if create:
                os.mkdir(path)
            else:
                raise IOError(f'no such directory: {path}')
        self.path = path
        self.filename_fmt = filename_fmt
        self.index = 0
        self.writeimage = writeimage

    def write(self, frame):
        filename = self.filename_fmt.format(self.index)
        filename = os.path.join(self.path, filename)
        self.writeimage(filename, frame)
        self.index += 1


class GUIWindowSink(FrameSink):
    """
    A frame sink that shows frames in a pop up GUI Window.
    """
    def __init__(self, title=None):
        self.winname = title or "Output"
        self.window = cv2.namedWindow(self.winname, cv2.WINDOW_AUTOSIZE)
    
    def write(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.winname, frame)
        if cv2.waitKey(1) == 27:
            raise EOFError("user escape")
    
    def close(self):
        cv2.destroyWindow(self.winname)


def readimage(filename):
    """
    Reads an image from a file using cv2.imread and converts it to RGB
    """
    im = cv2.imread(filename)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def writeimage(filename, image):
    """
    Converts an image to BGR and writes it to the file using cv2.imwrite
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    retval = cv2.imwrite(filename, image)
    if not retval:
        raise IOError("unable to write image")
