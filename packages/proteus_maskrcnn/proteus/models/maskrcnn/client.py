import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from proteus.models.base import BaseModel
from proteus.types import BoundingBox, Segmentation
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from .helpers import decode, generate_anchors, nms, read_class_names

logger = logging.getLogger(__name__)

folder_path = Path(__file__).parent


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
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    INPUT_NAME = "image"
    OUTPUT_NAMES = ["6568", "6570", "6572", "6887"]
    DTYPE = "FP32"

    @classmethod
    def _image_preprocess(cls, image):
        # Resize
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize(
            (int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR
        )

        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, : image.shape[1], : image.shape[2]] = image
        image = padded_image

        return image

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format
        """
        img = img.convert("RGB")

        logger.info(f"Original image size: {img.size}")

        img = cls._image_preprocess(img)

        npdtype = triton_to_np_dtype(cls.DTYPE)
        img = img.astype(npdtype)

        return img

    @classmethod
    def _detection_postprocess(
        cls, original_image_size, boxes, labels, scores, masks, score_threshold=0.7
    ):
        # Resize boxes
        logger.info(f"original_image_size {original_image_size}")
        ratio = 800.0 / min(original_image_size[0], original_image_size[1])
        boxes /= ratio

        results = []
        for mask, box, label, score in zip(masks, boxes, labels, scores):
            # Showing boxes with score > 0.7
            if score <= score_threshold:
                continue

            # Finding contour based on mask
            mask = mask[0, :, :, None]
            int_box = [int(i) for i in box]
            mask = cv2.resize(
                mask, (int_box[2] - int_box[0] + 1, int_box[3] - int_box[1] + 1)
            )
            mask = mask > 0.5
            im_mask = np.zeros(
                (original_image_size[0], original_image_size[1]), dtype=np.uint8
            )
            x_0 = max(int_box[0], 0)
            x_1 = min(int_box[2] + 1, original_image_size[1])
            y_0 = max(int_box[1], 0)
            y_1 = min(int_box[3] + 1, original_image_size[0])
            mask_y_0 = int(max(y_0 - box[1], 0))
            mask_y_1 = int(mask_y_0 + y_1 - y_0)
            mask_x_0 = int(max(x_0 - box[0], 0))
            mask_x_1 = int(mask_x_0 + x_1 - x_0)
            im_mask[y_0:y_1, x_0:x_1] = mask[mask_y_0:mask_y_1, mask_x_0:mask_x_1]
            im_mask = im_mask[:, :, None]

            bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            results.append((score, bbox, label, im_mask))
        return results

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

        postprocess_results = cls._detection_postprocess(
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

    @classmethod
    def inference_http(cls, triton_client, img):
        """
        Run inference on an img

        :param triton_client : the client to use
        :param img: the img to process (Pillow)

        :return: results
        """
        # Careful, Pillow has (w,h) format but most models expect (h,w)
        w, h = img.size

        # Preprocess the images into input data according to model
        # requirements
        image_data = [cls.preprocess(img)]

        # Send requests of batch_size=1 images. If the number of
        # images isn't an exact multiple of batch_size then just
        # start over with the first images until the batch is filled.
        # TODO batching
        responses = []

        sent_count = 0

        if cls.MAX_BATCH_SIZE > 0:
            batched_image_data = np.stack([image_data[0]], axis=0)
        else:
            batched_image_data = image_data[0]

        # Send request
        try:
            for inputs, outputs in cls._request_generator(batched_image_data):
                sent_count += 1
                responses.append(
                    triton_client.infer(
                        cls.__name__,
                        inputs,
                        request_id=str(sent_count),
                        model_version=cls.MODEL_VERSION,
                        outputs=outputs,
                    )
                )
        except InferenceServerException as e:
            logger.info("inference failed: " + str(e))

        final_responses = []
        for response in responses:
            this_id = response.get_response()["id"]
            logger.info("Request {}, batch size {}".format(this_id, 1))
            final_response = cls.postprocess(
                response, (h, w), 1, cls.MAX_BATCH_SIZE > 0
            )
            final_responses.append(final_response)
        return final_responses
