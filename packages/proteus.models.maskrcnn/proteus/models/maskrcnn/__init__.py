__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@gmail.com"
__version__ = "0.0.1"

from .client import MaskRCNN

model_dict = {
    MaskRCNN.__name__: MaskRCNN,
}
