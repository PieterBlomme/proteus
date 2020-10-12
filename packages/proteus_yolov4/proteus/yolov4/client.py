import logging
from pathlib import Path

import cv2
import numpy as np
from proteus.models import DetectionModel
from proteus.types import BoundingBox
from tritonclient.utils import triton_to_np_dtype

# isort: skip
from .helpers import (
    get_anchors,
    nms,
    postprocess_bbbox,
    postprocess_boxes,
    read_class_names,
)

folder_path = Path(__file__).parent

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


class YoloV4(DetectionModel):

    MODEL_NAME = "yolov4"
    CHANNEL_FIRST = False
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    ANCHORS = get_anchors(f"{folder_path}/yolov4_anchors.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx"

    @classmethod
    def _image_preprocess(cls, image, target_size):

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_padded[dh : nh + dh, dw : nw + dw, :] = image_resized
        image_padded = image_padded / 255.0
        return image_padded

    @classmethod
    def preprocess(cls, img, dtype):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

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

        image = cls._image_preprocess(open_cv_image, (cls.SHAPE[0], cls.SHAPE[1]))

        npdtype = triton_to_np_dtype(dtype)
        image = image.astype(npdtype)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))

        return image

    @classmethod
    def postprocess(
        cls, results, original_image_size, output_names, batch_size, batching
    ):
        """
        Post-process results to show bounding boxes.
        """
        logger.info(output_names)
        detections = [results.as_numpy(output_name) for output_name in output_names]
        logger.info(list(map(lambda detection: detection.shape, detections)))

        STRIDES = np.array([8, 16, 32])
        XYSCALE = [1.2, 1.1, 1.05]

        input_size = cls.SHAPE[0]

        # swap TODO check why this is needed...
        (h, w) = original_image_size

        pred_bbox = postprocess_bbbox(detections, cls.ANCHORS, STRIDES, XYSCALE)
        bboxes = postprocess_boxes(pred_bbox, (w, h), input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method="nms")

        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id]
        results = []
        for i, bbox in enumerate(bboxes):
            bbox = BoundingBox(
                x1=int(bbox[0]),
                y1=int(bbox[1]),
                x2=int(bbox[2]),
                y2=int(bbox[3]),
                class_name=cls.CLASSES[int(bbox[5])],
                score=float(bbox[4]),
            )
            results.append(bbox)

        return results


inference_http = YoloV4.inference_http
