import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from proteus.models.base import DetectionModel
from proteus.types import BoundingBox
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype

from .helpers import decode, generate_anchors, nms, read_class_names

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

folder_path = Path(__file__).parent


class RetinaNet(DetectionModel):

    DESCRIPTION = (
        "RetinaNet is a single-stage object detection model.  "
        "This version uses ResNet101 backbone.  mAP 0.376"
        "Taken from https://github.com/onnx/models."
    )
    CLASSES = read_class_names(f"{folder_path}/coco_names.txt")
    NUM_OUTPUTS = 10
    SHAPE = (3, 480, 640)
    MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx"
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    input_name = "input"
    output_names = [
        "output1",
        "output10",
        "output2",
        "output3",
        "output4",
        "output5",
        "output6",
        "output7",
        "output8",
        "output9",
    ]
    dtype = "FP32"

    @classmethod
    def _image_resize(cls, image, target_size):

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
    def _image_preprocess(cls, input_image):
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        # Create a mini-batch as expected by the model.
        return input_tensor.numpy()

    @classmethod
    def preprocess(cls, img, dtype):
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

        img = cls._image_resize(img, cls.SHAPE[1:])
        img = cls._image_preprocess(img)

        npdtype = triton_to_np_dtype(dtype)
        img = img.astype(npdtype)

        return img

    @classmethod
    def _detection_postprocess(cls, original_image_size, cls_heads, box_heads):
        # Inference post-processing
        anchors = {}
        decoded = []

        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = original_image_size[-1] // cls_head.shape[-1]
            if stride not in anchors:
                anchors[stride] = generate_anchors(
                    stride,
                    ratio_vals=[1.0, 2.0, 0.5],
                    scales_vals=[4 * 2 ** (i / 3) for i in range(3)],
                )
            # Decode and filter boxes
            decoded.append(
                decode(
                    cls_head,
                    box_head,
                    stride,
                    threshold=0.05,
                    top_n=1000,
                    anchors=anchors[stride],
                )
            )

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        # NMS threshold = 0.5
        scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
        return scores, boxes, labels

    @classmethod
    def postprocess(
        cls, results, original_image_size, output_names, batch_size, batching
    ):
        """
        Post-process results to show bounding boxes.
        https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/retinanet
        """

        # sort output names
        output_names = [f"output{i}" for i in range(1, 11)]

        cls_heads = [
            torch.from_numpy(results.as_numpy(output_name))
            for output_name in output_names[:5]
        ]
        logger.info(list(map(lambda detection: detection.shape, cls_heads)))
        box_heads = [
            torch.from_numpy(results.as_numpy(output_name))
            for output_name in output_names[5:]
        ]
        logger.info(list(map(lambda detection: detection.shape, box_heads)))

        # Size here is input size of the model !!
        # Still postprocessing needed to invert padding and scaling.
        scores, boxes, labels = cls._detection_postprocess(
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
