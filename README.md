# Proteus

## Goals

This was a personal project, born from following "frustrations":
- There's a lot of open-sourced code available, but often it is written towards training/evaluation on a specific dataset
- It can be time-consuming to rework the code towards an inference oriented scenario
- It's not easy to benchmark accuracy of models in a standardized environment
- It's not easy to benchmark speed of models against eachother.  

With proteus, I wanted to:
- Unify inference for a bunch of models
- Benchmark them easily against eachother
- Have a reusable framework where I can easily add extra models.  

It is my intention to add more vision models to this repository, especially state of the art as they come out.  The repo is open-source and I very much welcome any help !
Unfortunately some models will not fit within this framework.  eg. for Transformer-based architectures it's very hard to convert them to an Onnx model.  Still figuring out how to handle this.  

## Architecture

Proteus works with ONNX model files.  Model inference is handled by a Triton backend, with FastApi providing an API service in front of it.  I also supply Prometheus and Grafana for monitoring, but that's still work in progress.

## Howto (Development)

You will need docker and docker-compose installed.  Development can be done using the docker-compose.yml.  
In development mode, the proteus_api/Dockerfile is used.  This will mount the packages folder and watch file changes.  So you do not have to restart for every code change.  For changes to the API, or new package requirements, you will need to restart docker compose.

packages/package_install.txt contains a list of the packages to install.  If you need only 1 or 2 models for your use-case, I recommend you install only those.

You can create boiler plate code for a new model implementation using tools/templating.  
In general you will need 3 things to deploy a model:
- ONNX model file
- preprocessing code
- postprocessing code

## Howto (Production)

For production, you need to build using proteus_api/Dockerfile.prod.  This will pre-install the model packges.  
I am also providing a Kubernetes example, this is the recommended way to deploy in production.  

## Available models

### Classification
MobileNetV2
Resnet50V2
EfficientNetLite4

### Instance Segmentation
MaskRCNN

### Detection
RetinaNet
EfficientDetD0
EfficientDetD2
YoloV4

### Pose Estimation
EfficientPoseRT
EfficientPoseI
EfficientPoseII
EfficientPoseIII
EfficientPoseIV

### Misc
SuperResolution