__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@gmail.com"
__version__ = "0.0.1"

from .bsds import BSDSSuperRes
from .coco import CocoValBBox, CocoValMask
from .imagenette import ImageNette
from .mpii import MPIIPoseEstimation

__all__ = [
    "ImageNette",
    "CocoValBBox",
    "CocoValMask",
    "BSDSSuperRes",
    "MPIIPoseEstimation",
]
