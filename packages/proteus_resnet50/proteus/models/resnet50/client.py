from pathlib import Path

from proteus.models.base import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class Resnet50(ClassificationModel):

    MODEL_NAME = "resnet50"
    CHANNEL_FIRST = True
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx"


inference_http = Resnet50.inference_http
load_model = Resnet50.load_model
