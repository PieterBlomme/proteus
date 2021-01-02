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
        "{{cookiecutter.model_description}}"
    )
    MODEL_URL = "{{cookiecutter.model_url}}"

    """
    Note: if CONFIG_PATH is None, Triton will figure out a default configuration from the ONNX file.  
    The {{cookiecutter.model_name}}/load endpoint will return the used configuration, which can then be
    used to fill the actual config.template.  It is not recommended to leave CONFIG_PATH empty in production
    because it will not support features like batching, num_instances and TritonOptimization.
    """
    CONFIG_PATH = None
    #CONFIG_PATH = f"{folder_path}/config.template"
    
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
