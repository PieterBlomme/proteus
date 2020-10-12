from pathlib import Path

from proteus.models import ClassificationModel

from .helpers import read_class_names
import os
import requests

folder_path = Path(__file__).parent


def maybe_download():
    target_path = '/models/efficientnetlite4/1/model.onnx'
    if not os.path.isfile(target_path):
        url = 'https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx'
        r = requests.get(url)
        try:
            os.mkdir('/models/efficientnetlite4')
        except Exception as e:
            print(e)
        try:
            os.mkdir('/models/efficientnetlite4/1')
        except Exception as e:
            print(e)
        with open(target_path, 'wb') as f:
            f.write(r.content)


class EfficientNetLite4(ClassificationModel):

    MODEL_NAME = "efficientnetlite4"
    CHANNEL_FIRST = False
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")

    @classmethod
    def load_model(cls, triton_client):
        maybe_download()
        triton_client.load_model(cls.MODEL_NAME)


inference_http = EfficientNetLite4.inference_http
load_model = EfficientNetLite4.load_model
