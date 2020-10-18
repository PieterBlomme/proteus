import logging
from pathlib import Path

import numpy as np
from proteus.models.base import DetectionModel
from proteus.types import BoundingBox
from tritonclient.utils import triton_to_np_dtype

from .helpers import read_class_names

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

folder_path = Path(__file__).parent


class RetinaNet(DetectionModel):

    CHANNEL_FIRST = False
    DESCRIPTION = (
        "RetinaNet is a single-stage object detection model.  "
        "This version uses ResNet101 backbone.  mAP 0.376"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    NUM_OUTPUTS = 1
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx"

    @classmethod
    def preprocess(cls, img, dtype):
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

        npdtype = triton_to_np_dtype(dtype)
        open_cv_image = open_cv_image.astype(npdtype)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))

        return open_cv_image

    @classmethod
    def postprocess(
        cls, results, original_image_size, output_names, batch_size, batching
    ):
        """
        Post-process results to show bounding boxes.
        Based on this (very few postprocess needed):
        https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb
        """
        logger.info(output_names)
        detections = [results.as_numpy(output_name) for output_name in output_names]
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