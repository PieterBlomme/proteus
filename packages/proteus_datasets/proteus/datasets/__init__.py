__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .coco import CocoValBBox
from .datasets import CocoVal, ImageNette

__all__ = ["ImageNette", "CocoVal", "CocoValBBox"]
