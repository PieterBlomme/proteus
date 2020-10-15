__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .client import inference_http, load_model, YoloV4

__all__ = ["inference_http", "load_model"]

model_dict = {
    YoloV4.__name__: YoloV4
}