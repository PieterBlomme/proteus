__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@gmail.com"
__version__ = "0.0.1"

from .base import BaseModel
from .classification import ClassificationModel

__all__ = ["BaseModel", "ClassificationModel"]

model_dict = {}