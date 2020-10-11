from pathlib import Path

from proteus.models import ClassificationModel

from .helpers import read_class_names

folder_path = Path(__file__).parent


class Resnet50(ClassificationModel):

    MODEL_NAME = "resnet50"
    CHANNEL_FIRST = True
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")


inference_http = Resnet50.inference_http
