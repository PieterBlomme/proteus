__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .classification import ClassificationModel
from .detection import DetectionModel

__all__ = ["DetectionModel", "ClassificationModel"]

model_dict = {}