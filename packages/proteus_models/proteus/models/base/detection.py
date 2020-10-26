import logging

import numpy as np
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from .base import BaseModel

logger = logging.getLogger(__name__)


class DetectionModel(BaseModel):

    # Defaults
    DESCRIPTION = "Base DetectionModel"
    SHAPE = (416, 416, 3)

    @classmethod
    def preprocess(cls, img):
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
        """
        logger.info(cls.OUTPUT_NAMES)
        detections = [results.as_numpy(output_name) for output_name in cls.OUTPUT_NAMES]
        logger.info(list(map(lambda detection: detection.shape, detections)))
        return None
