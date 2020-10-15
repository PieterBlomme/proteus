from pathlib import Path

from proteus.models.base import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class MobileNetV2(ClassificationModel):

    CHANNEL_FIRST = True
    DESCRIPTION = (
        "Very efficient model with 70.94 % Top-1 accuracy on ImageNet. "
        " Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
