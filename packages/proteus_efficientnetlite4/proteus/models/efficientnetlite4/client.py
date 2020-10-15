from pathlib import Path

from proteus.models.base import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class EfficientNetLite4(ClassificationModel):

    CHANNEL_FIRST = False
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"


inference_http = EfficientNetLite4.inference_http
load_model = EfficientNetLite4.load_model
