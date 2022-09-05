# TRACTION Face and Object Recognition Module for the Co-Creation Space

Video face detection and recognition software based on RetinaFace and FaceNet. It requires the Co-Creation Space tool available in https://github.com/traction-project/CoCreationSpace.

<img src="https://www.traction-project.eu/wp-content/uploads/sites/3/2020/02/Logo-cabecera-Traction.png" align="left"/><em>This tool was originally developed as part of the <a href="https://www.traction-project.eu/">TRACTION</a> project, funded by the European Commission’s <a hef="http://ec.europa.eu/programmes/horizon2020/">Horizon 2020</a> research and innovation programme under grant agreement No. 870610.</em>

## Documentation

The documentation is available here: https://traction-project.github.io/FaceObjRecognitionForCCSpace

## Installation

Follow the instructions at https://pytorch.org/ to install PyTorch for your
system using ether `conda` or `pip`. If you have a CUDA compatible graphics
card, ensure to install a CUDA comparible version of PyTorch to allow the
deep models to run on the GPU.

To install the `facerec` package and `facetool` console script, run:

```
pip install .
```

This will install the `facetool` script on your environment path.

## Co-Creation Space Integration

The `facerecdb.py` file integrates the face recognition module to Co-Creation Space. Once Co-Creation Space is running with AWS credentials set up in the AWS CLI, simply run the `facerecdb.py` file. Make sure database credentials are correct.
The module will add tags based on faces found to videos in the Co-Creation Space. For faces to be correctly tagged, upload to Co-Creation Space (i.e. create a post) at least one image of the faces of the individuals that appears in videos. Name the post with the name of the individual.

## Standalone Usage

To see the available commands, run:

```
facetool --help
```

The output will look something like:

```
usage: facetool [-h] {index,thumbnail,recognize_gui,recognize,detect,detect_camera_demo,track,download} ...

optional arguments:
  -h, --help            show this help message and exit

subcommands:
  {index,thumbnail,recognize_gui,recognize,detect,detect_camera_demo,track,download}
    index               Indexes a dataset of face images.
    thumbnail           Generate thumbnails for an input image folder.
    recognize_gui       Detect and recognize faces in images or video.
    recognize           Detect and recognize faces in images or video.
    detect              Detect faces and write bounding boxes to stdout.
    detect_camera_demo  Detect faces on the webcam and show the results in a GUI window.
    track               Detect faces and tracks them.
    download            Downloads and extracts the LFW dataset.
```


## Video face recognition

To use the video face recognition software, you'll first need to index a 
collection of faces to recognize. The software assumes that this collection
is stored in a folder with one subfolder per individual, each named with the
label you want to apply to that person. Inside these folders should be one or
more example images of the person.


### Indexing media

To index media run the following:

```
facetool index media/images media/index 
```
to create a `media/index` folder containing the indexed files. Add a 
`--device=cpu` to run this on the CPU. Indexing may take some time, especially
on a CPU (approx 30 mins).

### Outputting the results of recognition to the console

```
facetool recognize media/example.mp4 media/index --device cpu (--device is an optional parameter)
```

The output will look something like:
```
6 [310  61 372 150] Aaron_Tippin 0.5715298652648926
8 [301  54 364 147] Bob_Geldof 0.526273787021637
10 [297  49 359 139] Jimmy_Kimmel 0.6068898439407349
12 [291  48 354 137] Jimmy_Kimmel 0.5837208032608032
14 [286  47 350 135] Jimmy_Kimmel 0.6029414534568787
16 [285  45 347 132] Jimmy_Kimmel 0.6005147099494934
18 [284  44 346 130] Jimmy_Kimmel 0.6477267742156982
20 [282  44 345 131] Jimmy_Kimmel 0.649223268032074
22 [279  43 343 133] Jimmy_Kimmel 0.6489613652229309
24 [277  45 341 132] Jimmy_Kimmel 0.6109910011291504
26 [277  44 339 130] Jimmy_Kimmel 0.6060200929641724
28 [274  44 337 129] Jimmy_Kimmel 0.5980086922645569
30 [273  45 336 129] Jimmy_Kimmel 0.6105645298957825
```
Each line has the format `<frame-number> [<x1> <y1> <x2> <y2>] <label> <confidence>`