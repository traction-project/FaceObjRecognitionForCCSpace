
# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
from .models.retinaface import RetinaFace
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms
from ..filters import Filter
from .info import Detection, FrameDetections

from pathlib import Path

import torch
import cv2
import numpy as np


class RetinaFaceDetector(object):

    def __init__(self, **kwargs):
        path = Path(__file__).absolute().parent / 'models' 
        path = path / 'mobilenet0.25_Final.pth'
        model = RetinaFace(cfg_mnet, phase='test')
        state_dict = torch.load(path, map_location=lambda s, l: s)
        model.load_state_dict(state_dict)
        self.model = model
        self.configure(**kwargs)

    def configure(self, device='cuda', resize=1, confidence_threshold=0.02, 
                  top_k=5000, nms_threshold=0.4, keep_top_k=750):
        self.device = device
        self.resize = resize
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        return self

    def __call__(self, frame):
        self.model.to(self.device)
        self.model.eval()
        # TODO: does retinanet want BGR?
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cfg = cfg_mnet
        img = frame.astype(np.float32)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        loc, conf, landms = self.model(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        boxes = dets[:, 0:4].astype(np.int64)
        confidences = dets[:, 4]
        landmarks = landms.reshape(landms.shape[0], landms.shape[1]//2, 2).astype(np.int64)

        # convert to FrameDetections object
        detections = FrameDetections(frame)
        for i in range(boxes.shape[0]):
            detection = Detection(
                boxes[i], landmarks[i], confidences[i], category='face')
            detections.append(detection)

        return detections


class RetinaFaceFilter(Filter):
    def __init__(self, source=None, **kwargs):
        super().__init__(source)
        self.detector = RetinaFaceDetector()
        self.configure(**kwargs)

    def configure(self, **kwargs):
        self.detector.configure(**kwargs)
        return self
    
    def transform(self, source):
        frame = source.next()
        with torch.no_grad():
            dets = self.detector(frame)       
        return dets


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
        