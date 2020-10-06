from pathlib import Path
from .helpers import read_class_names
from proteus.models import ClassificationModel

folder_path = Path(__file__).parent


class MobileNetV2(ClassificationModel):

    MODEL_NAME = 'mobilenet'
    CHANNEL_FIRST = True
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")


inference_http = MobileNetV2.inference_http
