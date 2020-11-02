__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .coco import CocoValBBox, CocoValMask
from .datasets import ImageNette

__all__ = ["ImageNette", "CocoValBBox", "CocoValMask"]
