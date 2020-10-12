from pathlib import Path

from proteus.models import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class MobileNetV2(ClassificationModel):

    MODEL_NAME = "mobilenet"
    CHANNEL_FIRST = True
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx"


inference_http = MobileNetV2.inference_http
load_model = MobileNetV2.load_model
