# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import cv2
import numpy as np

from .flow_vis import flow_to_color
from .abstract import Filter, Buffer



class VFlipFilter(Filter):
    """Vertical flip"""
    def transform(self, source):
        return cv2.flip(source.next(), 0)


class HFlipFilter(Filter):
    """Horizontal flip"""
    def transform(self, source):
        return cv2.flip(source.next(), 1)


class Rotate90Filter(Filter):
    """90 degree rotation"""
    def transform(self, source):
        return cv2.rotate(source.next(), cv2.ROTATE_90_CLOCKWISE)


class Rotate180Filter(Filter):
    """180 degree rotation"""
    def transform(self, source):
        return cv2.rotate(source.next(), cv2.ROTATE_180)


class Rotate270Filter(Filter):
    """270 degree rotation"""
    def transform(self, source):
        return cv2.rotate(source.next(), cv2.ROTATE_90_COUNTERCLOCKWISE)


class ColorConversionFilter(Filter):
    """Color space conversion"""
    def __init__(self, source=None, mode='RGB2LAB'):
        super().__init__(source)
        self.configure(mode)

    def configure(self, mode='RGB2LAB'):
        self.mode = mode
        self.conversion_code = getattr(cv2, 'COLOR_' + mode)
        return self
    
    def transform(self, source):
        return cv2.cvtColor(source.next(), self.conversion_code)


class CropFilter(Filter):
    """Crop"""
    def __init__(self, source=None, top=None, left=None, bottom=None, right=None):
        super().__init__(source)
        self.configure(top, left, bottom, right)

    def configure(self, top=None, left=None, bottom=None, right=None):
        self.bounds = top, left, bottom, right
        return self
    
    def transform(self, source):
        frame = source.next()
        top, left, bottom, right = self.bounds
        if top is None:
            top = 0
        if left is None:
            left = 0
        if bottom is None:
            bottom = frame.shape[0]
        if right is None:
            right = frame.shape[1]
        return frame[top:bottom, left:right, ...]


class ResizeFilter(Filter):
    """Spatial resample"""
    def __init__(self, source=None, size=None, interpolation='cubic', scale=None):
        super().__init__(source)
        self.configure(size=size, interpolation=interpolation, scale=scale)

    def configure(self, size=None, interpolation='cubic', scale=None):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        return self

    def transform(self, source):
        frame = source.next()

        try:
            interp = getattr(cv2, 'INTER_' + self.interpolation.upper())
        except AttributeError:
            raise ValueError('invalid interpolation method')

        height, width = frame.shape[:2]

        if self.size is None:

            if self.scale is None:
                raise ValueError("must specify a size or a scale")

            height = int(height * self.scale)
            width = int(width * self.scale)

        elif self.scale is None:

            if hasattr(self.size, '__iter__'):
                # size is a tuple 
                height, width = map(int, self.size)
            
            else:
                # resize long edge to size and preserve aspect ration
                ratio = self.size / height if height > width else self.size / width
                height = int(height * ratio)
                width = int(width * ratio)

        return cv2.resize(frame, (width, height), interpolation=interp)


class OpticalFlow(Filter):
    """Optical Flow (Farneback)"""
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)
        self.prev = None

    def configure(self, pyr_scale=0.5, levels=3, winsize=15, 
                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        return self
        
    def transform(self, source):
        if self.prev is None:
            self.prev = read_grayscale(source)
        frame = read_grayscale(source)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev, frame, None, self.pyr_scale, self.levels, self.winsize, 
            self.iterations, self.poly_n, self.poly_sigma, self.flags)
        self.prev = frame
        return flow


class TemporalSmooth(Filter):
    """Temporal exponential moving average"""
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)
        self.frame = None

    def configure(self, alpha=0.9):
        self.alpha = alpha
        return self

    def transform(self, source):
        frame = source.next()
        if self.frame is None:
            self.frame = frame
        else:
            a = self.alpha
            self.frame = a * frame + (1 - a) * self.frame
        return self.frame.astype(frame.dtype)


class BilateralFilter(Filter):
    """Gaussian bilateral filter"""
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)
        self.frame = None

    def configure(self, color_sigma=200, space_sigma=100, kernel_size=9):
        self.color_sigma = color_sigma
        self.space_sigma = space_sigma
        self.kernel_size = kernel_size
        return self
        
    def transform(self, source):
        frame = source.next()
        return cv2.bilateralFilter(frame, self.kernel_size, self.color_sigma, self.space_sigma)


class OpticalFlowVisualization(Filter):
    """Visualize optical flow image"""

    def __init__(self, source=None, **kwargs):
        super().__init__(source)

    def transform(self, source):
        return flow_to_color(source.next())


class SobelEdges(Filter):
    def __init__(self, source=None):
        super().__init__(source)

    def transform(self, source):
        img = read_grayscale(source)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        edges = (gx**2 + gy**2)**0.5
        return edges.astype(np.uint8)


class CannyEdges(Filter):
    def __init__(self, source=None, thresh1=50, thresh2=150):
        super().__init__(source)
        self.configure(thresh1, thresh2)

    def configure(self, thresh1=50, thresh2=150):
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        return self

    def transform(self, source):
        img = read_grayscale(source)
        edges = cv2.Canny(img, self.thresh1, self.thresh2)
        return edges.astype(np.uint8)


class WarpPerspectiveFilter(Filter):
    """
    Apply a perspective transform to frames.
    """
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)

    def configure(self, transform=None, target_size=None):
        self.matrix = transform
        self.target_size = target_size
        return self

    def transform(self, source):
        frame = source.next()
        if self.matrix is not None:
            dsize = frame.shape[1], frame.shape[0]
            if self.target_size is not None:
                dsize = self.target_size
            frame = cv2.warpPerspective(frame, self.matrix, dsize)
        return frame


class WarpDetectionsFilter(Filter):
    """
    Apply a perspective transform to detections.
    """
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.configure(**kwargs)

    def configure(self, transform=None, target_size=None):
        self.matrix = transform
        self.target_size = target_size
        return self

    def transform(self, source):
        frame, dets, class_names = source.next()
        if self.matrix is not None:
            boxes = np.asarray(dets[0])
            n = boxes.shape[0]
            p1 = boxes[:, :2]
            p2 = boxes[:, 2:]
            pts = np.concatenate([p1, p2])
            pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.matrix)  # pylint: disable=too-many-function-args
            pts = pts.reshape(-1, 2)
            p1 = pts[:n]
            p2 = pts[n:]
            boxes = np.hstack([p1, p2])
        return frame, dets, class_names 
        

def read_grayscale(source):
    """Read from source and convert to grayscale"""
    frame = source.next()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame