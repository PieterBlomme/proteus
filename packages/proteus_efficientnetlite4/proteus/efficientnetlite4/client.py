from pathlib import Path
from .helpers import read_class_names
from proteus.models import ClassificationModel

folder_path = Path(__file__).parent


class EfficientNetLite4(ClassificationModel):

    MODEL_NAME = 'efficientnetlite4'
    CHANNEL_FIRST = False
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")


inference_http = EfficientNetLite4.inference_http
