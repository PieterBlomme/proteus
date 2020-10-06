import cv2
import numpy as np
import logging
from proteus.types import Class
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ClassificationModel:

    # Defaults
    MODEL_NAME = 'classification'
    MODEL_VERSION = '1'
    CHANNEL_FIRST = False
    SHAPE = (224, 224, 3)
    DTYPE = 'float32'
    MAX_BATCH_SIZE = 1
    CLASSES = []

    @classmethod
    def parse_model_http(cls, model_metadata, model_config):
        """
        TODO check if this is still relevant
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata['inputs']) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata['inputs'])))
        if len(model_metadata['outputs']) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata['outputs'])))

        if len(model_config['input']) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config['input'])))

        input_metadata = model_metadata['inputs'][0]
        all_output_metadata = model_metadata['outputs']

        for output_metadata in all_output_metadata:
            if output_metadata['datatype'] != "FP32":
                raise Exception("expecting output datatype to be FP32, model '" +
                                model_metadata['name'] + "' output type is " +
                                output_metadata['datatype'])

        # Model input must have 3 dims (not counting the batch dimension),
        # either CHW or HWC
        input_batch_dim = (cls.MAX_BATCH_SIZE > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata['shape']) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata['name'],
                       len(input_metadata['shape'])))

        return (input_metadata['name'], output_metadata['name'], input_metadata['datatype'])

    @classmethod
    def _pre_process_edgetpu(cls, img, dims):
        """
        set image file dimensions to 224x224 by resizing and cropping
        image from center

        :param img: image as array in HWC format
        :param dims: dims as tuple in HWC order
        """
        output_height, output_width, _ = dims
        img = cls._resize_with_aspectratio(img, output_height, output_width,
                                           inter_pol=cv2.INTER_LINEAR)
        img = cls._center_crop(img, output_height, output_width)
        img = np.asarray(img, dtype=cls.DTYPE)
        # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
        img -= [127.0, 127.0, 127.0]
        img /= [128.0, 128.0, 128.0]
        return img

    @classmethod
    def _resize_with_aspectratio(cls, img, out_height, out_width, scale=87.5,
                                 inter_pol=cv2.INTER_LINEAR):
        """
        resize the image with a proportional scale

        :param img: image as array in HWC format
        :param out_height: height after resize
        :param out_width: width after resize:
        :param scale: scale to keep aspect ratio?
        :param inter_pol: type of interpolation for resize
        """
        height, width, _ = img.shape
        new_height = int(100. * out_height / scale)
        new_width = int(100. * out_width / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img

    @classmethod
    def _center_crop(cls, img, out_height, out_width):
        """
        crop the image around the center based on given height and width

        :param img: image as array in HWC format
        :param out_height: height after resize
        :param out_width: width after resize:
        """
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        See details at
        https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4

        :param img: image as array in HWC format
        """
        if cls.SHAPE[2] == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        logger.info(f'Original image size: {sample_img.size}')

        # pillow to cv2
        sample_img = np.array(sample_img)
        sample_img = sample_img[:, :, ::-1].copy()

        # preprocess
        img = cls._pre_process_edgetpu(sample_img, cls.SHAPE)

        # channels first if needed
        if cls.CHANNEL_FIRST:
            img = np.transpose(img, (2, 0, 1))
        return img


    @classmethod
    def postprocess(cls, results, output_name, batch_size,
                    batching, topk=5):
        """
        Post-process results to show classifications.

        :param results: raw results
        :param output_name: name of the output to process
        :param batch_size TODO
        :param batching TODO
        :param topk: how many results to return
        """
        output_array = results.as_numpy(output_name)

        # Include special handling for non-batching models
        responses = []
        for results in output_array:
            if not batching:
                results = [results]

            # softmax
            results = softmax(results)

            # get sorted topk
            idx = np.argpartition(results, -topk)[-topk:]
            response = [Class(class_name=cls.CLASSES[i], score=float(results[i]))
                        for i in idx]
            response.sort(key=lambda x: x.score, reverse=True)
            responses.append(response)
        return responses

    @classmethod
    def requestGenerator(cls, batched_image_data, input_name, 
                         output_name, dtype):
        """ Set the input data """
        inputs = [httpclient.InferInput(input_name, batched_image_data.shape,
                                        dtype)]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = []
        outputs.append(
                httpclient.InferRequestedOutput(output_name, binary_data=True))
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

        input_name, output_name, dtype = cls.parse_model_http(model_metadata,
                                                              model_config)

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
            for inputs, outputs in cls.requestGenerator(
                        batched_image_data, input_name, output_name, dtype):
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
            final_response = cls.postprocess(response, output_name,
                                             1,
                                             cls.MAX_BATCH_SIZE > 0)
            logger.info(final_response)
            final_responses.append(final_response)

        return final_responses