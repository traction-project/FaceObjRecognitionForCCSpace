# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import torch
import cv2
from ..filters import Filter
from .deep_sort import DeepSort
from pathlib import Path


class DeepSortFilter(Filter):

    def __init__(self, source=None):
        super().__init__(source)
        path = Path(__file__).absolute().parent / 'deep_sort'
        path = path / 'deepsort_ckpt.t7'
        self.model = DeepSort(str(path))
        self.configure()
        
    def configure(self, device='cuda'):
        self.device = device
        self.model.max_dist = 0.6
        self.model.min_confidence = 0.001
        self.model.nms_max_overlap = 0.3
        self.model.max_iou_distance = 0.5
        self.model.max_age = 10
        self.model.n_init = 1
        self.model.nn_budget = 100
        self.model.use_cuda = bool(self.device=='cuda')
        return self

    def transform(self, source):
        detections = source.next()
        
        if len(detections) == 0:
            self.model.increment_ages()
            return detections

        xywh, confs = self.adapt_input_for_deepsort(detections)
        boxes, ids = self.update_deepsort(xywh, confs, detections.frame)

        for det, id in zip(detections, ids):
            det.identity = id
        return detections
    
    def update_deepsort(self, xywhs, confss, frame):
        outputs = self.model.update(xywhs, confss, frame)
        track_boxes = outputs[:, :4]
        identities = outputs[:, -1]
        return track_boxes, identities
        
    def adapt_input_for_deepsort(self, detections):
        bbox_xywh, confs = [], []
        for detection in detections:
            x_c, y_c = detection.center
            h, w = detection.size
            bbox_xywh.append([x_c, y_c, w, h])
            confs.append([detection.confidence])
        return torch.Tensor(bbox_xywh), torch.Tensor(confs)
