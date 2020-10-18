from pathlib import Path

from proteus.models.base import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


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
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"
