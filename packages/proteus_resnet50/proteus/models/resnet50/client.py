from pathlib import Path

from proteus.models.base import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class Resnet50V2(ClassificationModel):

    CHANNEL_FIRST = True
    DESCRIPTION = (
        "ResNet models provide very high accuracies with affordable model sizes. "
        "75.81% Top-1 on Imagenet for Resnet50 V2"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx"
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    input_name = "data"
    output_names = ["resnetv24_dense0_fwd"]
    dtype = "FP32"