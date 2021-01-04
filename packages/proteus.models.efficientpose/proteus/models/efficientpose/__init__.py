__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@gmail.com"
__version__ = "0.0.1"

from .client import (
    EfficientPoseI,
    EfficientPoseII,
    EfficientPoseIII,
    EfficientPoseIV,
    EfficientPoseRT,
)

model_dict = {
    EfficientPoseI.__name__: EfficientPoseI,
    EfficientPoseII.__name__: EfficientPoseII,
    EfficientPoseIII.__name__: EfficientPoseIII,
    EfficientPoseIV.__name__: EfficientPoseIV,
    EfficientPoseRT.__name__: EfficientPoseRT,
}
