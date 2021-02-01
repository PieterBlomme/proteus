# Proteus EfficientNetLite4

Package for RetinaNet model usage in Proteus

Model and implementation taken from https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
MIT License

## Pre/Postprocessing
I wrote shared preprocessing for all classification models to keep it simple.  Essentially this is limited to a resize with aspect ratio.
Postprocessing is also coming from ClassificationModel

## Models

Available Proteus configuration options:
- Num Instances
- Quantization (INT8 precision)
- TritonOptimization