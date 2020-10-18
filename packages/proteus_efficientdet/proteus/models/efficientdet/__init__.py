__author__ = """Pieter Blomme"""
__email__ = "pieter.blomme@robovision.eu"
__version__ = "0.0.1"

from .client import EfficientDetD0, EfficientDetD2, EfficientDetD7

model_dict = {
    EfficientDetD0.__name__: EfficientDetD0,
    EfficientDetD2.__name__: EfficientDetD2,
    EfficientDetD7.__name__: EfficientDetD7,
}
