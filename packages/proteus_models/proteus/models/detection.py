import numpy as np
import logging
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
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
    NUM_OUTPUTS = 3


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

        npdtype = triton_to_np_dtype(dtype)
        open_cv_image = open_cv_image.astype(npdtype)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))

        return open_cv_image


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
        return None

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
            final_response = cls.postprocess(response, img.size,
                                             output_names, 1,
                                             cls.MAX_BATCH_SIZE > 0)
            final_responses.append(final_response)
        return final_responses
