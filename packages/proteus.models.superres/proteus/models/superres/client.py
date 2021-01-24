import logging
from pathlib import Path

import numpy as np
from PIL import Image
from proteus.models.base import BaseModel
from proteus.models.base.modelconfigs import (
    BaseModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
    TritonOptimizationModelConfig,
)
from resizeimage import resizeimage

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

    DESCRIPTION = (
        "Implementation of Sub-Pixel CNN (2016) - https://arxiv.org/abs/1609.05158"
    )
    MODEL_PATH = f"{folder_path}/super-resolution-10.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "input"
    OUTPUT_NAMES = ["output"]
    DTYPE = "FP32"
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

        img = resizeimage.resize_cover(img, [224, 224], validate=False)
        img_ycbcr = img.convert("YCbCr")
        img_y_0, img_cb, img_cr = img_ycbcr.split()
        img_ndarray = np.asarray(img_y_0)
        img_4 = np.expand_dims(img_ndarray, axis=0)
        model_input = img_4.astype(np.float32) / 255.0

        # Save some parts in the PREDICTION_DATA store for postprocess
        extra_data["img_cb"] = img_cb
        extra_data["img_cr"] = img_cr
        return model_input, extra_data

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
        # Fetch from the PREDICTION_DATA store
        img_cb = extra_data["img_cb"]
        img_cr = extra_data["img_cr"]

        output_name = cls.OUTPUT_NAMES[0]
        results = results.as_numpy(output_name)
        logger.debug(results)
        img_out_y = Image.fromarray(
            np.uint8((results[0] * 255.0).clip(0, 255)[0]), mode="L"
        )
        final_img = Image.merge(
            "YCbCr",
            [
                img_out_y,
                img_cb.resize(img_out_y.size, Image.BICUBIC),
                img_cr.resize(img_out_y.size, Image.BICUBIC),
            ],
        ).convert("RGB")
        logger.debug(final_img)
        return final_img
