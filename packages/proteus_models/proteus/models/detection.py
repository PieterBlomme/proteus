import cv2
import numpy as np
import logging
from proteus.types import BoundingBox
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
from .detection_helpers import postprocess_bbbox, postprocess_boxes, nms
from .base import BaseModel
# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


class DetectionModel(BaseModel):

    # Defaults
    MODEL_NAME = 'detection'
    MODEL_VERSION = '1'
    CHANNEL_FIRST = False
    SHAPE = (416, 416, 3)
    DTYPE = 'float32'
    MAX_BATCH_SIZE = 1
    CLASSES = []
    ANCHORS = None
    NUM_OUTPUTS = 3

    @classmethod
    def _image_preprocess(cls, image, target_size):

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw/w, ih/h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_padded = image_padded / 255.
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
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        logger.info(f'Original image size: {sample_img.size}')

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
    def postprocess(cls, results, original_image_size, output_names, 
                    batch_size, batching):
        """
        Post-process results to show bounding boxes.
        """
        logger.info(output_names)
        detections = [results.as_numpy(output_name) for
                      output_name in output_names]
        logger.info(list(map(lambda detection: detection.shape, detections)))

        STRIDES = np.array([8, 16, 32])
        XYSCALE = [1.2, 1.1, 1.05]

        input_size = cls.SHAPE[0]

        # swap TODO check why this is needed...
        (h, w) = original_image_size

        pred_bbox = postprocess_bbbox(detections, cls.ANCHORS, 
                                      STRIDES, XYSCALE)
        bboxes = postprocess_boxes(pred_bbox, (w, h),
                                   input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')

        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id]
        results = []
        for i, bbox in enumerate(bboxes):
            bbox = BoundingBox(x1=int(bbox[0]),
                               y1=int(bbox[1]),
                               x2=int(bbox[2]),
                               y2=int(bbox[3]),
                               class_name=cls.CLASSES[int(bbox[5])],
                               score=float(bbox[4])
                               )
            results.append(bbox)

        return results


    @classmethod
    def requestGenerator(cls, batched_image_data, input_name,
                         output_names, dtype):
        """ Set the input data """
        inputs = [httpclient.InferInput(input_name, batched_image_data.shape,
                                        dtype)]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [httpclient.InferRequestedOutput(output_name,
                                                   binary_data=True)
                   for output_name in output_names]
        yield inputs, outputs

    @classmethod
    def inference_http(cls, triton_client, img):
        """
        Run inference on an img

        :param triton_client : the client to use
        :param img: the img to process

        :return: results
        """
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=cls.MODEL_NAME, model_version=cls.MODEL_VERSION)
        except InferenceServerException as e:
            raise Exception("failed to retrieve the metadata: " + str(e))

        try:
            model_config = triton_client.get_model_config(
                model_name=cls.MODEL_NAME, model_version=cls.MODEL_VERSION)
        except InferenceServerException as e:
            raise Exception("failed to retrieve the config: " + str(e))

        logger.info(f'Model metadata: {model_metadata}')
        logger.info(f'Model config: {model_config}')

        input_name, output_names, dtype = cls.parse_model_http(model_metadata, 
                                                        model_config)

        # Preprocess the images into input data according to model
        # requirements
        image_data = [cls.preprocess(img, dtype)]

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
            for inputs, outputs in cls.requestGenerator(
                        batched_image_data, input_name, output_names, dtype):
                sent_count += 1
                responses.append(
                        triton_client.infer(cls.MODEL_NAME,
                                            inputs,
                                            request_id=str(sent_count),
                                            model_version=cls.MODEL_VERSION,
                                            outputs=outputs))
        except InferenceServerException as e:
            logger.info("inference failed: " + str(e))

        final_responses = []
        for response in responses:
            this_id = response.get_response()["id"]
            logger.info("Request {}, batch size {}".format(this_id, 1))
            final_response = cls.postprocess(response, img.size, output_names, 1,
                                        cls.MAX_BATCH_SIZE > 0)
            final_responses.append(final_response)
        return final_responses
