# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import numpy as np
import cv2

from typing import Tuple, List, Dict, Callable, Generator
from ..filters import Filter

Point = Tuple[float, float]


class BBoxProp(object):
    def __init__(self, index):
        self.index = index

    def __get__(self, obj, objtype=None):
        return obj.bbox[self.index]

    def __set__(self, obj, value):
        obj.bbox[self.index] = value


class Detection(object):
    bbox : np.ndarray # 4 elements
    landmarks : np.ndarray # N x 2, dtype=np.int64
    confidence : float
    category : str
    identity : str
    recognition_score : float

    x1 = BBoxProp(0)
    y1 = BBoxProp(1)
    x2 = BBoxProp(2)
    y2 = BBoxProp(3)
    top = BBoxProp(1)
    left = BBoxProp(0)
    bottom = BBoxProp(3)
    right = BBoxProp(2)

    def __init__(
            self, 
            bbox : np.ndarray = None, 
            landmarks : np.ndarray = None,
            confidence : float = 0.0, 
            category : str = None, 
            identity : str = None):
        self.bbox = (np.asarray(bbox, np.int64) 
            if bbox is not None else np.zeros(4, dtype=np.int64))
        self.landmarks = (np.asarray(landmarks, np.int64) 
            if landmarks is not None else None)
        self.confidence = confidence
        self.category = category
        self.identity = identity
        self.recognition_score = 0.0

    @property
    def top_left(self) -> Point:
        return self.bbox[0], self.bbox[1]

    @top_left.setter
    def top_left(self, value : Point):
        self.bbox[0] = value[0]
        self.bbox[1] = value[1]

    @property
    def bottom_right(self) -> Point:
        return self.bbox[2], self.bbox[3]

    @bottom_right.setter
    def bottom_right(self, value : Point):
        self.bbox[2] = value[0]
        self.bbox[3] = value[1]

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def size(self):
        return self.width, self.height

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def empty(self):
        return self.width <= 0 or self.height <= 0

    @property
    def area(self):
        return self.width * self.height

    def clip(self, width : float, height : float):
        self.bbox[0] = max(self.bbox[0], 0)
        self.bbox[1] = max(self.bbox[1], 0)
        self.bbox[2] = min(self.bbox[2], width)
        self.bbox[3] = min(self.bbox[3], height)
    
    def clip_to_frame(self, frame : np.array):
        h, w = frame.shape[:2]
        self.clip(w, h)

    def draw(self, frame, 
             bbox_color=(255, 0, 0), 
             label_color=(255, 255, 255),
             landmark_color=(255, 0, 255)):
        
        # draw bounding box
        if bbox_color is not None:
            cv2.rectangle(
                frame, self.top_left, self.bottom_right, bbox_color[::-1], 2)
        
        # draw text
        if label_color is not None:
            if self.identity is not None:
                text = f'{self.identity} ({self.recognition_score*100:.1f})'
            else:
                text = f'face ({self.confidence:.1f})'
            cx, cy = self.x1, self.y1 + 12
            cv2.putText(frame, text, (cx, cy), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color[::-1])
        
        # draw landmarks
        if landmark_color is not None and self.landmarks is not None:
            for pt in self.landmarks:
                cv2.circle(frame, tuple(pt), 1, landmark_color[::-1], 4)
        
        return frame

    def crop(self, frame):
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]


class FrameDetections(object):
    frame : np.array
    detections : List[Detection]

    def __init__(self, frame : np.array = None):
        self.frame = frame
        self.detections = []

    def append(self, detection: Detection, clip=True) -> None:
        if clip:
            detection.clip_to_frame(self.frame)
        self.detections.append(detection)

    def draw(self,
             bbox_color=(255, 0, 0), 
             label_color=(255, 255, 255),
             landmark_color=(255, 0, 255)) -> np.ndarray:
        frame = self.frame.copy()
        for det in self.detections:
            det.draw(frame, bbox_color, label_color, landmark_color)
        return frame

    def __getitem__(self, index):
        return self.detections[index]

    def __iter__(self):
        return self.detections.__iter__()
    
    def __len__(self) -> int:
        return len(self.detections)


class DetectionRenderer(Filter):    
    def transform(self, source):
        dets = source.next()
        return dets.draw()


