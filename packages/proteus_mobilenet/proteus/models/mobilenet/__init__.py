__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .client import inference_http, load_model, MobileNetV2

__all__ = ["inference_http", "load_model"]

model_dict = {
    MobileNetV2.__name__: MobileNetV2
}