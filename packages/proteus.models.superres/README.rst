========
Proteus SuperResolution
========

Package for SuperResolution model usage in Proteus

Model and code taken from https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016, which in turn based the code on:
- https://github.com/pytorch/examples/tree/master/super_resolution
- https://arxiv.org/abs/1609.05158

Licensing: the original Pytorch implementation has a BSD-3 license.  

The model takes a 224x224x3 channel input and upscales it to 672x672x3.  Images of other sizes will be resized first, but ofcourse it makes little sense to use this model if you already have a hi-res image.
