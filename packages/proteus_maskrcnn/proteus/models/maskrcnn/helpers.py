import logging
import math

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names


def image_preprocess(image):
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


def detection_postprocess(
    original_image_size, boxes, labels, scores, masks, score_threshold=0.7
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
