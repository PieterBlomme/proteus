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

    DESCRIPTION = "Description for model goes here"
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "input"
    OUTPUT_NAMES = ["output"]
    DTYPE = "FP32"
    MODEL_CONFIG = ModelConfig

    @classmethod
    def preprocess(cls, img, pred_ref):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format

        :return: processed image
        """

        img = resizeimage.resize_cover(img, [224, 224], validate=False)
        img_ycbcr = img.convert("YCbCr")
        img_y_0, img_cb, img_cr = img_ycbcr.split()
        img_ndarray = np.asarray(img_y_0)
        img_4 = np.expand_dims(img_ndarray, axis=0)
        img_5 = img_4.astype(np.float32) / 255.0

        # Save some parts in the PREDICTION_DATA store for postprocess
        cls.PREDICTION_DATA[pred_ref]['img_cb'] = img_cb
        cls.PREDICTION_DATA[pred_ref]['img_cr'] = img_cr
        return img_5

    @classmethod
    def postprocess(cls, results, pred_ref, batch_size, batching):
        """
        Post-process results to return valid outputs.
        """
        # Fetch from the PREDICTION_DATA store
        img_cb = cls.PREDICTION_DATA[pred_ref]['img_cb']
        img_cr = cls.PREDICTION_DATA[pred_ref]['img_cr']

        output_name = cls.OUTPUT_NAMES[0]
        results = results.as_numpy(output_name)
        logger.debug(results)
        img_out_y = Image.fromarray(
            np.uint8((results[0] * 255.0).clip(0, 255)[0]), mode="L"
        )
        final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
        logger.debug(final_img)
        return final_img
