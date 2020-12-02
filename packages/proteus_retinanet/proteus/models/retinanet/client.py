import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from proteus.models.base import BaseModel
from proteus.types import BoundingBox
from tritonclient.utils import triton_to_np_dtype

from .helpers import (
    decode,
    detection_postprocess,
    generate_anchors,
    image_resize,
    nms,
    read_class_names,
)

logger = logging.getLogger(__name__)


folder_path = Path(__file__).parent


class RetinaNet(BaseModel):

    DESCRIPTION = (
        "RetinaNet is a single-stage object detection model.  "
        "This version uses ResNet101 backbone.  mAP 0.376"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    SHAPE = (3, 480, 640)
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx"
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    INPUT_NAME = "input"
    OUTPUT_NAMES = [
        "output1",
        "output2",
        "output3",
        "output4",
        "output5",
        "output6",
        "output7",
        "output8",
        "output9",
        "output10",
    ]
    DTYPE = "FP32"

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format
        """
        if cls.SHAPE[2] == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        logger.info(f"Original image size: {img.size}")

        # convert to cv2
        img = np.array(img)
        img = img[:, :, ::-1].copy()

        img = image_resize(img, cls.SHAPE[1:])
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

        cls_heads = [
            torch.from_numpy(results.as_numpy(output_name))
            for output_name in cls.OUTPUT_NAMES[:5]
        ]
        logger.info(list(map(lambda detection: detection.shape, cls_heads)))
        box_heads = [
            torch.from_numpy(results.as_numpy(output_name))
            for output_name in cls.OUTPUT_NAMES[5:]
        ]
        logger.info(list(map(lambda detection: detection.shape, box_heads)))

        # Size here is input size of the model !!
        # Still postprocessing needed to invert padding and scaling.
        scores, boxes, labels = detection_postprocess(
            cls.SHAPE[1:], cls_heads, box_heads
        )

        # scale, delta width, delta height
        _, ih, iw = cls.SHAPE
        h, w = original_image_size
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2

        results = []
        # TODO add another loop if batching
        for score, box, cat in zip(scores[0], boxes[0], labels[0]):
            x1, y1, x2, y2 = box.data.tolist()

            # unpad bbox
            x1 = max(x1 - dw, 0)
            x2 = min(x2 - dw, w)
            y1 = max(y1 - dh, 0)
            y2 = min(y2 - dh, h)

            # scale
            x1, x2, y1, y2 = x1 / scale, x2 / scale, y1 / scale, y2 / scale

            bbox = BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                class_name=cls.CLASSES[int(cat.item())],
                score=float(score.item()),
            )
            results.append(bbox)
        return results
