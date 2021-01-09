import logging
from pathlib import Path

import numpy as np
from proteus.models.base import BaseModel
from proteus.models.base.modelconfigs import (
    BaseModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
    TritonOptimizationModelConfig,
)
from proteus.types import Coordinate

from .helpers import extract_coordinates, preprocess

folder_path = Path(__file__).parent
logger = logging.getLogger(__name__)


class ModelConfig(
    BaseModelConfig,
    TritonOptimizationModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
):
    pass


class EfficientPoseRT(BaseModel):

    DESCRIPTION = (
        "EfficientPoseRT implementation from https://github.com/daniegr/EfficientPose"
    )
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/effpose/EfficientPoseRT.onnx"

    """
    Note: if CONFIG_PATH is None, Triton will figure out a default configuration from the ONNX file.  
    The EfficientPose/load endpoint will return the used configuration, which can then be
    used to fill the actual config.template.  It is not recommended to leave CONFIG_PATH empty in production
    because it will not support features like batching, num_instances and TritonOptimization.
    """
    CONFIG_PATH = f"{folder_path}/config_RT.template"

    INPUT_NAME = "input_res1:0"
    OUTPUT_NAMES = ["upscaled_confs/BiasAdd:0"]
    DTYPE = "FP32"
    MODEL_CONFIG = ModelConfig
    RESOLUTION = 224

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

        # Load image
        image = np.array(img)
        image_height, image_width = image.shape[:2]
        extra_data["image_height"] = image_height
        extra_data["image_width"] = image_width

        # For simplicity so we don't have to rewrite the original code
        batch = np.expand_dims(image, axis=0)
        # Preprocess batch
        batch = preprocess(batch, cls.RESOLUTION)
        # Pull single image out of batch
        image = batch[0]

        return image, extra_data

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
        image_height = extra_data["image_height"]
        image_width = extra_data["image_width"]

        batch_outputs = results.as_numpy(cls.OUTPUT_NAMES[0])
        logger.debug(f"Shape of outputs: {batch_outputs.shape}")
        coordinates = extract_coordinates(
            batch_outputs[0, ...], image_height, image_width
        )
        logger.debug(f"Coordinates: {coordinates}")

        # Coordinates are normalized, so convert to real pixel values
        coordinates = [
            (name, x * image_width, y * image_height) for (name, x, y) in coordinates
        ]

        # Convert to Proteus type for JSON response
        proteus_coords = [
            Coordinate(name=name, x=x, y=y) for (name, x, y) in coordinates
        ]

        return proteus_coords


class EfficientPoseI(EfficientPoseRT):

    DESCRIPTION = (
        "EfficientPoseI implementation from https://github.com/daniegr/EfficientPose"
    )
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/effpose/EfficientPoseI.onnx"

    CONFIG_PATH = f"{folder_path}/config_I.template"
    RESOLUTION = 256


class EfficientPoseII(EfficientPoseRT):

    DESCRIPTION = (
        "EfficientPoseII implementation from https://github.com/daniegr/EfficientPose"
    )
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/effpose/EfficientPoseII.onnx"

    CONFIG_PATH = f"{folder_path}/config_II.template"
    RESOLUTION = 368


class EfficientPoseIII(EfficientPoseRT):

    DESCRIPTION = (
        "EfficientPoseIII implementation from https://github.com/daniegr/EfficientPose"
    )
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/effpose/EfficientPoseIII.onnx"

    CONFIG_PATH = f"{folder_path}/config_III.template"
    RESOLUTION = 480


class EfficientPoseIV(EfficientPoseRT):

    DESCRIPTION = (
        "EfficientPoseIV implementation from https://github.com/daniegr/EfficientPose"
    )
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/effpose/EfficientPoseIV.onnx"

    CONFIG_PATH = f"{folder_path}/config_IV.template"
    RESOLUTION = 600
