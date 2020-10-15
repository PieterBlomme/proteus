__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .client import EfficientNetLite4, inference_http, load_model

__all__ = ["inference_http", "load_model"]


model_dict = {EfficientNetLite4.__name__: EfficientNetLite4}
