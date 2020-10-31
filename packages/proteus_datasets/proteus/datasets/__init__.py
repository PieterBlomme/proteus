__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .datasets import CocoVal, ImageNette
from .coco import CocoValBBox

__all__ = ["ImageNette", "CocoVal", "CocoValBBox"]
