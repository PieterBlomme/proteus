from pathlib import Path

from proteus.models.base import ClassificationModel
from proteus.models.base.modelconfigs import BaseModelConfig

from .helpers import read_class_names

folder_path = Path(__file__).parent


class ModelConfig(BaseModelConfig):
    pass


class Resnet50V2(ClassificationModel):

    CHANNEL_FIRST = True
    DESCRIPTION = (
        "ResNet models provide very high accuracies with affordable model sizes. "
        "75.81% Top-1 on Imagenet for Resnet50 V2"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_PATH = f"{folder_path}/resnet50-v2-7.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "data"
    OUTPUT_NAMES = ["resnetv24_dense0_fwd"]
    DTYPE = "FP32"
