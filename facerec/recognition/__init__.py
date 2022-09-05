# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import numpy as np
import json
import torch
import torchvision.transforms as transforms

from PIL import Image
from facenet_pytorch import InceptionResnetV1 as FaceNet
from pathlib import Path
from ..filters import Filter


class FaceRecognitionEngine(object):
    def __init__(self, indexpath, weights='vggface2', device='cuda'):
        indexpath = Path(indexpath)
        index = np.load(str(indexpath / 'descriptors.npy'))
        self.labels = json.load(open(indexpath / 'labels.json'))
        self.index = torch.Tensor(index).to(device)
        self.feature_extractor = FaceNet(pretrained=weights).eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        self.device = device
        self.recognition_threshold = 0.5

    def process(self, detections):
        if len(detections) == 0:
            return detections
        
        with torch.no_grad():
            # extract faces
            faces = torch.zeros(len(detections), 3, 160, 160)
            for i, detection in enumerate(detections):
                if detection.empty:
                    continue
                face = detection.crop(detections.frame)
                faces[i] = self.transform(Image.fromarray(face))
            faces = faces.to(self.device)
        
            # compute descriptors (#faces x 512)
            descriptors = self.feature_extractor(faces).data

            # compute scores (#images x #faces)
            scores = self.index @ descriptors.T

            # best scores (#faces)
            best_scores, indices = torch.max(scores, axis=0)

            for i, detection in enumerate(detections):
                if best_scores[i] >= self.recognition_threshold:
                    detection.identity = self.labels[indices[i]]
                    detection.recognition_score = float(best_scores[i])
                detection.scores = scores[:, i]
                detection.labels = self.labels
                detection.descriptor = descriptors[i]
        
        return detections


class FaceRecognitionFilter(Filter):
    def __init__(self, source=None):
        super().__init__(source)
        self.engine = None
    
    def configure(
            self, indexpath, weights='vggface2', device='cuda', 
            recognition_threshold=0.5):
        self.engine = FaceRecognitionEngine(indexpath, weights, device)
        self.engine.recognition_threshold = recognition_threshold
        return self

    def transform(self, source):
        dets = source.next()
        return self.engine.process(dets)



class TrackStats(object):
    def __init__(self, detection):
        self.scores = detection.scores
        self.labels = detection.labels
        self.alpha = 0.95

    def update(self, scores):
        self.scores = self.alpha * self.scores + (1 - self.alpha) * scores
        idx = torch.argmax(self.scores)
        self.most_likely_identity = self.labels[idx]
        self.recognition_score = self.scores[idx]

    
class TrackingRecognitionFilter(Filter):
    def __init__(self, source=None):
        super().__init__(source)
        self.previous_dets = None
        self.iou_threshold = 0.7
        self.recognition_threshold = 0.5

    def configure(self, iou_threshold=0.7, recognition_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.recognition_threshold = recognition_threshold
        return self

    def transform(self, source):
        dets = source.next()
        if self.previous_dets is None:
            # attach track stats
            for det in dets:
                det.track_stats = TrackStats(det)
            self.previous_dets = dets
            return dets

        for det in dets:

            # find best matching detection from previous frame
            best_iou = 0
            best_match = None
            for pdet in self.previous_dets:
                iou = compute_iou(det.bbox, pdet.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = pdet

            if best_iou > self.iou_threshold:
                # copy track stats
                det.track_stats = pdet.track_stats

                # update track stats
                det.track_stats.update(det.scores)

                # update identity
                det.identity = det.track_stats.most_likely_identity
                det.recognition_score = float(det.track_stats.recognition_score)

                if det.recognition_score < self.recognition_threshold:
                    det.identity = None
            else:
                det.track_stats = TrackStats(det)


        self.previous_dets = dets
        return dets


def area_of_intersection(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x31 = max(x11, x21)
    y31 = max(y11, y21)
    x32 = min(x12, x22)
    y32 = min(y12, y22)
    if y31 > y32 or x31 > x32:
        return 0
    return (y32 - y31) * (x32 - x31)
    

def area_of_union(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x31 = min(x11, x21)
    y31 = min(y11, y21)
    x32 = max(x12, x22)
    y32 = max(y12, y22)
    return (y32 - y31) * (x32 - x31)


def compute_iou(box1, box2):
    i = area_of_intersection(box1, box2)
    u = area_of_union(box1, box2)
    return i / u 

