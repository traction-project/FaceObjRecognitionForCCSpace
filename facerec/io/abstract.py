# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>

class FrameSource(object):
    """
    Abstract base class for a source of frames. Concrete subclasses could, for 
    example, be videos, folders of images, or filters attached to other sources.
    """

    def next(self):
        """
        Provide the next frame.
        """
        raise NotImplementedError

    def skip(self):
        """
        Skip a frame. By default, this just calls next and returns nothing.
        """
        self.next()

    def __iter__(self):
        while True:
            try:
                frame = self.next()
            except EOFError:
                break
            else:
                yield frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        # not all sources have a len (e.g. webcam, live stream)
        raise NotImplementedError

    def close(self):
        """
        Close the frame source.
        """
        pass


class FrameSink(object):
    """
    Abstract base class for a sink for frames. This could be a video file,
    a folder of images, or a GUI window.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def write(self, frame):
        """
        Write a frame to the sink.
        """
        raise NotImplementedError

    def close(self):
        """
        Close the frame sink.
        """
        pass


class OneToManySink(FrameSink):
    """
    Writes a single frame to multiple sinks.
    """
    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def append(self, sink):
        self.sinks.append(sink)

    def extend(self, sinks):
        self.sinks.extend(sinks)

    def write(self, frame):
        for sink in self.sinks:
            sink.write(frame)
    
    def close(self):
        for sink in self.sinks:
            sink.close()


class ManyToManySink(FrameSink):
    """
    Writes a multiple frames to multiple sinks.
    """
    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def append(self, sink):
        self.sinks.append(sink)

    def extend(self, sinks):
        self.sinks.extend(sinks)

    def write(self, frames):
        for frame, sink in zip(frames, self.sinks):
            sink.write(frame)
    
    def close(self):
        for sink in self.sinks:
            sink.close()


class Pump(object):
    """
    Pumps items by pulling from the source and pushing to the sink.
    """
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink

    def pump(self):
        item = self.source.next()
        self.sink.write(item)

    def run(self, count=None):
        while count != 0:
            try:
                self.pump()
            except EOFError:
                break
            if count is not None:
                count -= 1
        return self


def pump(source, sink, count=None):
    Pump(source, sink).run(count)
