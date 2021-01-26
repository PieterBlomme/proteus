from pathlib import Path

from proteus.models.base import ClassificationModel
from proteus.models.base.modelconfigs import (
    BaseModelConfig,
    QuantizationModelConfig,
    TritonOptimizationModelConfig,
)

from .helpers import read_class_names

folder_path = Path(__file__).parent


class ModelConfig(
    BaseModelConfig, QuantizationModelConfig, TritonOptimizationModelConfig
):
    pass


class EfficientNetLite4(ClassificationModel):

    CHANNEL_FIRST = False
    DESCRIPTION = (
        "EfficientNet-Lite 4 is the largest variant and most accurate "
        "of the set of EfficientNet-Lite model. It is an integer-only quantized "
        "model that produces the highest accuracy of all of the EfficientNet models. "
        "It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time "
        "(e.g. 30ms/image) on a Pixel 4 CPU.  Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_PATH = "efficientnet-lite4-11.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "images:0"
    OUTPUT_NAMES = ["Softmax:0"]
    DTYPE = "FP32"
    MODEL_CONFIG = ModelConfig
