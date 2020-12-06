import logging
from pathlib import Path

import numpy as np
from proteus.models.base import BaseModel
from proteus.types import BoundingBox
from tritonclient.utils import triton_to_np_dtype

from .helpers import read_class_names

logger = logging.getLogger(__name__)

folder_path = Path(__file__).parent


class EfficientDetD0(BaseModel):

    CHANNEL_FIRST = False
    DESCRIPTION = (
        "EfficientDets are a family of object detection models, which achieve state-of-the-art "
        "55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous"
        " detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors."
        "Converted using https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb"
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/efficientdet/efficientdet-d0.onnx"
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    INPUT_NAME = "image_arrays:0"
    OUTPUT_NAMES = ["detections:0"]
    DTYPE = "UINT8"
    SHAPE = (416, 416, 3)

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        Based on this (very few preprocess needed):
        https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb

        :param img: image as array in HWC format
        """
        if cls.SHAPE[2] == 1:
            sample_img = img.convert("L")
        else:
            sample_img = img.convert("RGB")

        logger.info(f"Original image size: {sample_img.size}")

        # convert to cv2
        open_cv_image = np.array(sample_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        npdtype = triton_to_np_dtype(cls.DTYPE)
        open_cv_image = open_cv_image.astype(npdtype)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))

        return open_cv_image

    @classmethod
    def postprocess(cls, results, original_image_size, batch_size, batching):
        """
        Post-process results to show bounding boxes.
        Based on this (very few postprocess needed):
        https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb
        """
        logger.info(cls.OUTPUT_NAMES)
        detections = [results.as_numpy(output_name) for output_name in cls.OUTPUT_NAMES]
        # only one output, so
        detections = detections[0]
        logger.info(list(map(lambda detection: detection.shape, detections)))

        results = []
        # first dimension is the batch TODO
        for bbox in detections[0]:
            logger.info(bbox)
            # bbox[0] is the image id
            # ymin, xmin, ymax, xmax = bbox[1=5]
            bbox = BoundingBox(
                x1=int(bbox[2]),
                y1=int(bbox[1]),
                x2=int(bbox[4]),
                y2=int(bbox[3]),
                class_name=cls.CLASSES[int(bbox[6])],
                score=float(bbox[5]),
            )
            results.append(bbox)
        return results


class EfficientDetD2(EfficientDetD0):

    CHANNEL_FIRST = False
    DESCRIPTION = (
        "EfficientDets are a family of object detection models, which achieve state-of-the-art "
        "55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous"
        " detectors. Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors."
        "Converted using https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb"
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    MODEL_URL = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/efficientdet/efficientdet-d2.onnx"