# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import plac
import numpy as np
import tqdm
import json
import os
import torchvision.transforms as transforms
import facerec.io as io
import facerec.filters as filters
import cv2
import torch


from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from facerec.io.util import download_file, extract_tar
from facerec.detection import RetinaFaceFilter
from facerec.detection import RetinaFaceDetector
from facerec.detection import DetectionRenderer
from facenet_pytorch import InceptionResnetV1 as FaceNet
from facerec.recognition import FaceRecognitionFilter
from facerec.recognition import TrackingRecognitionFilter
from facerec.tracking import DeepSortFilter

commands = []

def command(func):
    commands.append(func.__name__)
    return func


def clip_box(box, shape):
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return x1, y1, x2, y2


class ExtractFace(object):
    """
    Finds and crops out the most confident face box from an image. Returns
    the full image if there is no face detected.
    """
    def __init__(self, **kwargs):
        self.detector = RetinaFaceDetector(**kwargs)
        
    def __call__(self, image):
        detections = self.detector(np.array(image))
        if len(detections) == 0:
            return image
        index = np.argmax([det.confidence for det in detections])
        face = image.crop(detections[index].bbox)
        return face


@command
def index(
    inputpath: "image folder",
    indexpath: "index folder",
    device: ("device", "option", "d")="cuda",
    weights: ("weights", "option", "w")="vggface2"):
    """
    Indexes a dataset of face images.
    """

    # setup index directory
    indexpath = Path(indexpath)
    if not indexpath.is_dir():
        indexpath.mkdir()

    # setup transforms (includes face detection and cropping)
    tf = transforms.Compose([
        ExtractFace(device=device),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    # create face feature function
    facenet = FaceNet(pretrained=weights).eval().to(device)

    # create dataset
    dataset = ImageFolder(inputpath, transform=tf)  
    print(f'found {len(dataset)} images of {len(dataset.classes)} people')  

    # extract descriptors
    descriptors = [[] for _ in range(len(dataset.classes))]
    for image, label in tqdm.tqdm(dataset, total=len(dataset)):
        image = image.to(device).unsqueeze(0)
        descriptor = facenet(image).squeeze()
        descriptors[label].append(descriptor.data.cpu().numpy())
        
    # average pool descriptors and L2 normalize
    descriptors = np.array([np.mean(d, axis=0) for d in descriptors])
    descriptor_norms = np.linalg.norm(descriptors, axis=1)
    descriptors = descriptors / descriptor_norms[:, np.newaxis]
    print(descriptors.shape)

    # save descriptors
    np.save(str(indexpath / 'descriptors.npy'), descriptors)

    # save classes
    with open(indexpath / 'labels.json', 'w') as f:
        json.dump(dataset.classes, f)


@command
def thumbnail(
    inputpath: "image folder",
    indexpath: "index folder",
    resize: ('thumbnail size', 'option', 's')=100):
    """
    Generate thumbnails for an input image folder.
    """

    # setup thumbnail directory
    indexpath = Path(indexpath)
    thumbpath = indexpath / 'thumbs'
    thumbpath.mkdir(exist_ok=True)

    # create dataset
    transform = transforms.Resize(resize)
    dataset = ImageFolder(inputpath, transform=transform)  
    print(f'found {len(dataset)} images of {len(dataset.classes)} people')

    done = set() 
    for image, label in tqdm.tqdm(dataset, total=len(dataset)):
        if label in done:
            continue
        image.save(thumbpath / f'{label}.jpg')
        done.add(label)


@command
def recognize_gui(
    inputpath: "input file or camera number",
    indexpath: "index folder",
    device: ("device", "option", "d")="cuda",
    weights: ("weights", "option", "w")="vggface2",
    rec_thresh: ("recognition confidence threshold", "option", "r")=0.5,
    det_thresh: ("detection confidence threshold", "option", "t")=0.5):
    """
    Detect and recognize faces in images or video.
    """

    pipeline = filters.Pipeline(filters=[
        filters.TemporalSubsample(skip=1),
        RetinaFaceFilter(device=device, confidence_threshold=det_thresh),
        FaceRecognitionFilter().configure(
            indexpath, weights=weights, device=device, 
            recognition_threshold=rec_thresh),
        TrackingRecognitionFilter().configure(
            recognition_threshold=rec_thresh),
        DetectionRenderer()
    ])

    with io.adaptive_source(inputpath) as source,\
         io.GUIWindowSink("Face recognition") as sink:
        
        pipeline.connect(source)
        io.pump(pipeline, sink)


@command
def recognize(
    inputpath: "input file or camera number",
    indexpath: "index folder",
    device: ("device", "option", "d")="cuda",
    weights: ("weights", "option", "w")="vggface2",
    rec_thresh: ("recognition confidence threshold", "option", "r")=0.5,
    det_thresh: ("detection confidence threshold", "option", "t")=0.5):
    """
    Detect and recognize faces in images or video.
    """

    pipeline = filters.Pipeline(filters=[
        filters.TemporalSubsample(skip=1),
        RetinaFaceFilter(device=device, confidence_threshold=det_thresh),
        FaceRecognitionFilter().configure(
            indexpath, weights=weights, device=device, 
            recognition_threshold=rec_thresh),
        TrackingRecognitionFilter().configure(
            recognition_threshold=rec_thresh),
    ])

    with io.adaptive_source(inputpath) as source:
        pipeline.connect(source)
        for frame_num, detections in enumerate(pipeline):
            for det in detections:
                print(frame_num*2, det.bbox, 
                      det.identity, det.recognition_score)


@command
def detect(inputpath: "input file or camera number",
           device: ("device", "option", "d")="cuda",
           det_thresh: ("detection confidence threshold", "option", "t")=0.25):
    """
    Detect faces and write bounding boxes to stdout.
    """
    pipeline = filters.Pipeline()
    pipeline.append(RetinaFaceFilter(
        device=device, confidence_threshold=det_thresh))    

    with io.adaptive_source(inputpath) as source:
        pipeline.connect(source)

        for frame_num, dets in enumerate(pipeline):
            
            for det in dets:
                box_str = ' '.join(str(i) for i in det.bbox)
                print(frame_num, box_str)  


@command
def detect_camera_demo(
    device: ("device", "option", "d")="cuda",
    det_thresh: ("detection confidence threshold", "option", "t")=0.25):
    """
    Detect faces on the webcam and show the results in a GUI window.
    """
    retinaface = RetinaFaceFilter(
        device=device, confidence_threshold=det_thresh)

    with io.CameraSource(0) as source,\
         io.GUIWindowSink("Retina Face") as sink:
         
         source = filters.HFlipFilter(source)
         source = retinaface.connect(source)
         source = DetectionRenderer(source)
         io.pump(source, sink)


@command
def track(inputpath: "input file or camera number",
          device: ("device", "option", "d")="cuda",
          det_thresh: ("detection confidence threshold", "option", "t")=0.5):

    """
    Detect faces and tracks them.
    """
    pipeline = filters.Pipeline()
    pipeline.append(filters.TemporalSubsample(skip=1))
    pipeline.append(RetinaFaceFilter(
        device=device, confidence_threshold=det_thresh))   
    pipeline.append(DeepSortFilter())
    pipeline.append(DetectionRenderer()) 

    with io.adaptive_source(inputpath) as source,\
         io.GUIWindowSink("Retina Face") as sink:
        pipeline.connect(source)
        io.pump(pipeline, sink)
        

@command
def download():
    """
    Downloads and extracts the LFW dataset.
    """
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    path = 'media'
    Path(path).mkdir(exist_ok=True)
    print('Downloading LFW dataset...')
    filename = download_file(url, path=path, progress=True, skip_existing=True)
    print('Extracting tarfile...')
    extract_tar(filename, path=path)
    os.unlink(filename)

