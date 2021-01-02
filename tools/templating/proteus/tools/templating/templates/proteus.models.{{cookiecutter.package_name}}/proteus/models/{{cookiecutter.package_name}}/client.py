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
    QuantizationModelConfig, # this will require ONNX opset 11
):
    pass

class {{cookiecutter.model_name}}(BaseModel):

    DESCRIPTION = (
        "Description for model goes here"
    )
    MODEL_URL = "{{cookiecutter.model_url}}"
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

        :param img: Pillow image

        :returns:
            - model_input: input as required by the model
            - extra_data: dict of data that is needed by the postprocess function
        """
        extra_data = {}
        return img, extra_data

    @classmethod
    def postprocess(cls, results, extra_data, batch_size, batching):
        """
        Post-process results to return valid outputs.
        :param results: model outputs
        :param extra_data: dict of data that is needed by the postprocess function
        :param batch_size
        :param batching: boolean flag indicating if batching

        :returns: json result
        """
        return results
