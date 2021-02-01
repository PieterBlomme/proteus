# Proteus EfficientDet

Package for EfficientDet model usage in Proteus

Models and implementation courtesy of the original Google implementation at https://github.com/google/automl/tree/master/efficientdet

Google Automml repo is licensed under Apache License 2.0

## Process for model conversion to ONNX
I used following example: https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb
However, due to recent changes in the automl efficientdet repo, you should use commit 57621e8f3eaddd2c0b421c65c0bbd323ebcf8f2d
when running this notebook

## Pre/Postprocessing
https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb was used as an example. 
However, hardly any pre/postpressing is needed for efficientdet.

## Models
- EfficientPoseD0
- EfficientPoseD2

Available Proteus configuration options:
- Num Instances