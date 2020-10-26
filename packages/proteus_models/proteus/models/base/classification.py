import logging

import cv2
import numpy as np
from proteus.types import Class
from tritonclient.utils import InferenceServerException

from .base import BaseModel

logger = logging.getLogger(__name__)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ClassificationModel(BaseModel):

    # Defaults
    MODEL_VERSION = "1"
    DESCRIPTION = "Base ClassificationModel"
    CHANNEL_FIRST = False
    SHAPE = (224, 224, 3)
    DTYPE = "float32"
    MAX_BATCH_SIZE = 1
    CLASSES = []

    @classmethod
    def _pre_process_edgetpu(cls, img, dims):
        """
        set image file dimensions to 224x224 by resizing and cropping
        image from center

        :param img: image as array in HWC format
        :param dims: dims as tuple in HWC order
        """
        output_height, output_width, _ = dims
        img = cls._resize_with_aspectratio(
            img, output_height, output_width, inter_pol=cv2.INTER_LINEAR
        )
        img = cls._center_crop(img, output_height, output_width)
        img = np.asarray(img, dtype=cls.DTYPE)
        # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
        img -= [127.0, 127.0, 127.0]
        img /= [128.0, 128.0, 128.0]
        return img

    @classmethod
    def _resize_with_aspectratio(
        cls, img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR
    ):
        """
        resize the image with a proportional scale

        :param img: image as array in HWC format
        :param out_height: height after resize
        :param out_width: width after resize:
        :param scale: scale to keep aspect ratio?
        :param inter_pol: type of interpolation for resize
        """
        height, width, _ = img.shape
        new_height = int(100.0 * out_height / scale)
        new_width = int(100.0 * out_width / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img

    @classmethod
    def _center_crop(cls, img, out_height, out_width):
        """
        crop the image around the center based on given height and width

        :param img: image as array in HWC format
        :param out_height: height after resize
        :param out_width: width after resize:
        """
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        See details at
        https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4

        :param img: image as array in HWC format
        """
        if cls.SHAPE[2] == 1:
            sample_img = img.convert("L")
        else:
            sample_img = img.convert("RGB")

        logger.info(f"Original image size: {sample_img.size}")

        # pillow to cv2
        sample_img = np.array(sample_img)
        sample_img = sample_img[:, :, ::-1].copy()

        # preprocess
        img = cls._pre_process_edgetpu(sample_img, cls.SHAPE)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))
        return img

    @classmethod
    def postprocess(
        cls, results, original_image_size, output_names, batch_size, batching, topk=5
    ):
        """
        Post-process results to show classifications.

        :param results: raw results
        :param output_name: name of the output to process
        :param batch_size TODO
        :param batching TODO
        :param topk: how many results to return
        """
        output_array = [results.as_numpy(output_name) for output_name in output_names]

        # Include special handling for non-batching models
        responses = []
        for results in output_array:
            if not batching:
                results = [results]

            # because of "output_names"??
            results = results[0]

            # softmax
            results = softmax(results)

            # get sorted topk
            idx = np.argpartition(results, -topk)[-topk:]
            response = [
                Class(class_name=cls.CLASSES[i], score=float(results[i])) for i in idx
            ]
            response.sort(key=lambda x: x.score, reverse=True)
            responses.append(response)
        return responses
