# Proteus EfficientPose

Package for EfficientPose model usage in Proteus

Models and implementation courtesy of https://github.com/daniegr/EfficientPose (licensed under Creative Commons Attribution 4.0 International).

## Process for model conversion to ONNX
````
pip install tf2onnx==1.7.2
git clone https://github.com/daniegr/EfficientPose
cd models/tensorflow
python -m tf2onnx.convert --graphdef EfficientPoseIV.pb --output EfficientPoseIV.onnx --inputs input_res1:0 --outputs upscaled_confs/BiasAdd:0 --opset 11
````

## Pre/Postprocessing
Almost exact copy paste from daniegr's repo.
The only change is that the normalized coords are converted to actual pixel values.

## Models
- EfficientPoseRT: 224x224 resolution
- EfficientPoseI: 256x256
- EfficientPoseII: 368x368
- EfficientPoseIII: 480x480
- EfficientPoseIV: 600x600

Available Proteus configuration options:
- Num Instances
- Quantization (INT8 precision)
- TritonOptimization
- Dynamic Batching