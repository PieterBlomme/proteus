import logging
from pathlib import Path

import cv2
import numpy as np
from proteus.models.base import BaseModel
from proteus.models.base.modelconfigs import (
    BaseModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
    TritonOptimizationModelConfig,
)
from proteus.types import BoundingBox
from tritonclient.utils import triton_to_np_dtype

# isort: skip
from .helpers import (
    get_anchors,
    image_preprocess,
    nms,
    postprocess_bbbox,
    postprocess_boxes,
    read_class_names,
)

folder_path = Path(__file__).parent
logger = logging.getLogger(__name__)


class ModelConfig(
    BaseModelConfig,
    TritonOptimizationModelConfig,
    BatchingModelConfig,
    QuantizationModelConfig,
):
    pass


class YoloV4(BaseModel):

    CHANNEL_FIRST = False
    DESCRIPTION = (
        "YOLOv4 optimizes the speed and accuracy of object detection. "
        "It is two times faster than EfficientDet. It improves YOLOv3's "
        "AP and FPS by 10% and 12%, respectively, with mAP50 of 52.32 "
        "on the COCO 2017 dataset and FPS of 41.7 on Tesla 100."
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    ANCHORS = get_anchors(f"{folder_path}/yolov4_anchors.txt")
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx"
    CONFIG_PATH = f"{folder_path}/config.template"
    INPUT_NAME = "input_1:0"
    OUTPUT_NAMES = ["Identity:0", "Identity_1:0", "Identity_2:0"]
    DTYPE = "FP32"
    SHAPE = (416, 416, 3)
    MODEL_CONFIG = ModelConfig

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

        :param img: Pillow image

        :returns:
            - model_input: input as required by the model
            - extra_data: dict of data that is needed by the postprocess function
        """
        extra_data = {}
        # Careful, Pillow has (w,h) format but most models expect (h,w)
        w, h = img.size
        extra_data["original_image_size"] = (h, w)

        if cls.SHAPE[2] == 1:
            sample_img = img.convert("L")
        else:
            sample_img = img.convert("RGB")

        logger.info(f"Original image size: {sample_img.size}")

        # convert to cv2
        open_cv_image = np.array(sample_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        image = image_preprocess(open_cv_image, (cls.SHAPE[0], cls.SHAPE[1]))

        npdtype = triton_to_np_dtype(cls.DTYPE)
        image = image.astype(npdtype)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))

        return image, extra_data

    @classmethod
    def postprocess(cls, results, extra_data, batch_size, batching):
        """
        Post-process results to show bounding boxes.
        :param results: model outputs
        :param extra_data: dict of data that is needed by the postprocess function
        :param batch_size
        :param batching: boolean flag indicating if batching

        :returns: json result
        """
        original_image_size = extra_data["original_image_size"]

        logger.debug(cls.OUTPUT_NAMES)
        detections = [results.as_numpy(output_name) for output_name in cls.OUTPUT_NAMES]
        logger.debug(list(map(lambda detection: detection.shape, detections)))

        STRIDES = np.array([8, 16, 32])
        XYSCALE = [1.2, 1.1, 1.05]

        input_size = cls.SHAPE[0]

        pred_bbox = postprocess_bbbox(detections, cls.ANCHORS, STRIDES, XYSCALE)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
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
