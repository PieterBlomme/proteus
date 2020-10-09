from pathlib import Path
from .helpers import read_class_names, get_anchors
from proteus.models import DetectionModel

folder_path = Path(__file__).parent


class YoloV4(DetectionModel):

    MODEL_NAME = 'yolov4'
    CHANNEL_FIRST = False
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    ANCHORS = get_anchors(f"{folder_path}/yolov4_anchors.txt")


inference_http = YoloV4.inference_http
