import logging
from pathlib import Path

from proteus.models.base import BaseModel
from proteus.models.base.modelconfigs import (
    BaseModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
    TritonOptimizationModelConfig,
)

folder_path = Path(__file__).parent
logger = logging.getLogger(__name__)


class ModelConfig(
    BaseModelConfig,
    TritonOptimizationModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,  # this will require ONNX opset 11
):
    pass


class SuperResolution(BaseModel):

    DESCRIPTION = "Description for model goes here"
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = None
    OUTPUT_NAMES = None
    DTYPE = None
    MODEL_CONFIG = ModelConfig

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format

        :return: processed image
        """
        return img

    @classmethod
    def postprocess(cls, results, original_image_size, batch_size, batching):
        """
        Post-process results to return valid outputs.
        """
        return results
