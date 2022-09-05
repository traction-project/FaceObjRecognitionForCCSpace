# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
from ..io.abstract import FrameSource


class Filter(FrameSource):
    """
    A filter is a frame source that can be attached to another frame source to transform
    its output. The output of a filter does not necessarily need to be a frame, but could
    consist of any kind of object (including multiple objects). Obviously, frame sources
    can only be connected to sources that produce an output that they can handle.
    """
    def __init__(self, source=None):
        self.source = None
        self.connect(source)

    def connect(self, source):
        """
        Connect the filter to a source.
        """
        self.source = source
        return self

    def next(self):
        """
        Returns the next filtered frame. This method may consume multiple frames from the 
        source as needed to produce the next frame. By default, this method calls 
        `self.transform(self.source)` and returns the result.
        """
        return self.transform(self.source)

    def transform(self, source):
        """
        Subclasses should implement this to produce the next frames.
        """
        raise NotImplementedError


class IdentityFilter(Filter):
    """
    Passes the source frames through unchanged.
    """
    def transform(self, source):
        return source.next()


class Pipeline(Filter):
    """
    Allows easy chaining together of filters.

    Example:
    pipeline = Pipeline(source, filters=[
        filter1, filter2, filter3
    ])
    """
    def __init__(self, source=None, filters=None):
        super().__init__(source)
        self.extend(filters or [])

    def connect(self, source):
        if self.source is not None:
            s = self.source
            while hasattr(s, 'source') and s.source is not None:
                s = s.source
            s.connect(source)
        else:
            self.source = source
        return self

    def append(self, filter):
        """
        Append a filter to the pipeline.
        """
        self.source = filter.connect(self.source)
        return self

    def extend(self, filters):
        """
        Extend the pipeline with a list of filters.
        """
        for filter in filters:
            self.append(filter)

    def transform(self, source):
        return source.next()

    
class Buffer(Filter):
    """
    Implements an internal frame buffer. Next will produce a list of frames
    from the buffer. Allows later in the pipeline to easily access a sliding
    windows of frames.
    """
    def __init__(self, source=None, size=10, must_fill=False):
        super().__init__(source)
        self.configure(size)
    
    def configure(self, size=10, must_fill=False):
        self.size = size
        self.buffer = []
        self.must_fill = must_fill
        return self

    def next(self):
        # pop frame from buffer
        if len(self.buffer) > 0:
            self.buffer.pop(0)

        # fill buffer
        while len(self.buffer) < self.size:
            try:
                self.buffer.append(self.source.next())
            except EOFError as e:
                if self.must_fill:
                    raise e
                else:
                    break
        
        # check if eof
        if len(self.buffer) == 0:
            raise EOFError("frame buffer empty")

        # return buffer 
        return self.buffer


class DuplicateFilter(Filter):
    """
    Takes a single source and produces [frame1, frame1, frame2, frame2, ...] count times.
    """
    def __init__(self, source=None, count=2):
        super().__init__(source)
        self.configure(count)
        self.index = 0
        self.frame = None

    def configure(self, count=2):
        self.count = count
        return self

    def transform(self, source):
        if self.frame is None:
            self.frame = source.next()
            self.index = 1
            return self.frame
        if self.index < self.count:
            self.index += 1
            return self.frame
        self.frame = source.next()
        self.index = 0
        return self.frame


class Branch(Filter):
    """
    Takes a single source and applies multiple parallel filters
    """
    def __init__(self, source=None, filters=None):
        self.filters = filters or []
        super().__init__(source)

    def connect(self, source):
        self.source = source
        for filter in self.filters:
            filter.connect(self.source)
        return self

    def append(self, filter):
        self.filters.append(filter.connect(self.source))
        return self

    def extend(self, filters):
        for filter in filters:
            self.append(filter)
        return self

    def next(self):
        results = [f.next() for f in self.filters]
        return results


class TemporalSubsample(Filter):
    """
    Downsample a source by dropping frames
    """
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)

    def configure(self, skip=1):
        self.skip = skip
        return self

    def transform(self, source):
        for _ in range(self.skip):
            source.skip()
        return source.next()
