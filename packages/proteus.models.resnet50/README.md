# Proteus ResNet50

Package for ResNet50 model usage in Proteus

Model and implementation taken from https://github.com/onnx/models/tree/master/vision/classification/resnet
MIT License

## Pre/Postprocessing
I wrote shared preprocessing for all classification models to keep it simple.  Essentially this is limited to a resize with aspect ratio.
Postprocessing is also coming from ClassificationModel

## Models
I am using the Resnet50 V2 version.

Available Proteus configuration options:
- Num Instances
- Quantization (INT8 precision)
- TritonOptimization