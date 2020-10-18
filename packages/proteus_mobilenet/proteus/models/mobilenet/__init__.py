__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .client import MobileNetV2

model_dict = {MobileNetV2.__name__: MobileNetV2}
