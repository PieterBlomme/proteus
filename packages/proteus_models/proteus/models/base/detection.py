import logging

import numpy as np
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from .base import BaseModel

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


class DetectionModel(BaseModel):

    # Defaults
    MODEL_VERSION = "1"
    DESCRIPTION = "Base DetectionModel"
    CHANNEL_FIRST = False
    SHAPE = (416, 416, 3)
    DTYPE = "float32"
    MAX_BATCH_SIZE = 1
    CLASSES = []

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

        npdtype = triton_to_np_dtype(cls.dtype)
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
        """
        logger.info(output_names)
        detections = [results.as_numpy(output_name) for output_name in output_names]
        logger.info(list(map(lambda detection: detection.shape, detections)))
        return None