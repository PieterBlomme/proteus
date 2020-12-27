import logging
from pathlib import Path

import cv2
import numpy as np
import pydantic
from PIL import Image
from proteus.models.base import BaseModel
from proteus.types import BoundingBox, Segmentation
from tritonclient.utils import triton_to_np_dtype

from .helpers import detection_postprocess, image_preprocess, read_class_names

logger = logging.getLogger(__name__)

folder_path = Path(__file__).parent


class ModelConfig(pydantic.BaseModel):
    num_instances: int = 1

class MaskRCNN(BaseModel):

    DESCRIPTION = (
        "This model is a real-time neural network for object "
        "instance segmentation that detects 80 different classes."
        "mAP of 0.36"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    NUM_OUTPUTS = 4
    MAX_BATCH_SIZE = 0
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "image"
    OUTPUT_NAMES = ["6568", "6570", "6572", "6887"]
    DTYPE = "FP32"
    MODEL_CONFIG = ModelConfig

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format
        """
        img = img.convert("RGB")

        logger.info(f"Original image size: {img.size}")

        img = image_preprocess(img)

        npdtype = triton_to_np_dtype(cls.DTYPE)
        img = img.astype(npdtype)

        return img

    @classmethod
    def postprocess(cls, results, original_image_size, batch_size, batching):
        """
        Post-process results to show bounding boxes.
        https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/retinanet
        """

        # get outputs
        boxes = results.as_numpy(cls.OUTPUT_NAMES[0])
        labels = results.as_numpy(cls.OUTPUT_NAMES[1])
        scores = results.as_numpy(cls.OUTPUT_NAMES[2])
        masks = results.as_numpy(cls.OUTPUT_NAMES[3])

        postprocess_results = detection_postprocess(
            original_image_size, boxes, labels, scores, masks
        )

        results = []
        # TODO add another loop if batching
        for (score, box, cat, mask) in postprocess_results:
            x1, y1, x2, y2 = box

            bbox = BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                class_name=cls.CLASSES[int(cat)],
                score=float(score),
            )

            ret, thresh = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            polygon = contours[0].reshape(-1).tolist()
            if len(polygon) <= 4:
                # not valid, create a dummy
                polygon = [0, 0, 1, 0, 1, 1]

            segmentation = Segmentation(
                segmentation=polygon,
                class_name=cls.CLASSES[int(cat)],
                score=float(score),
            )
            results.append({"bounding_box": bbox, "segmentation": segmentation})
        return results
