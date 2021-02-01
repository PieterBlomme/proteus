# Proteus YOLOV4

Package for YOLOV4 model usage in Proteus

Model and code taken from https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4, which in turn based the code on:
- https://github.com/hunglc007/tensorflow-yolov4-tflite

Licensing: the original implementation from hunglc007 has an MIT license 

The model acts on 416x416x3 inputs.  Proteus rescales if needed.

## Pre/Postprocessing
Taken from https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

## Models

Available Proteus configuration options:
- Num Instances
- Quantization (INT8 precision)
- TritonOptimization
- Dynamic Batching