# TODO clean up and split out
import numpy as np
from PIL import Image
import logging
import cv2
import math

import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype, InferenceServerException
from pathlib import Path
from .helpers import read_class_names
from proteus.types import Class

MODEL_NAME = 'mobilenet'
MODEL_VERSION = '1'

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

folder_path = Path(__file__).parent
classes = read_class_names(f"{folder_path}/imagenet_labels.txt")


def parse_model_http(model_metadata, model_config):
    """
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
    input_config = model_config['input'][0]
    all_output_metadata = model_metadata['outputs']

    max_batch_size = 1

    for output_metadata in all_output_metadata:
        if output_metadata['datatype'] != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata['name'] + "' output type is " +
                            output_metadata['datatype'])

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))

    # FORMAT_NCHW
    c = input_metadata['shape'][1 if input_batch_dim else 0]
    h = input_metadata['shape'][2 if input_batch_dim else 1]
    w = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], output_metadata['name'], c,
            h, w, input_config['format'], input_metadata['datatype'])


# set image file dimensions to 224x224 by resizing and cropping image from center 
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img
    
# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
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

# crop the image around the center based on given height and width 
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
    """
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    logger.info(f'Original image size: {sample_img.size}')

    #pillow to cv2
    sample_img = np.array(sample_img)
    sample_img = sample_img[:, :, ::-1].copy()

    #preprocess
    img = pre_process_edgetpu(sample_img, (224, 224, 3))
    img = np.transpose(img, (2, 0, 1))
    return img

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def postprocess(results, output_name, batch_size, batching, topk=5):
    """
    Post-process results to show classifications.
    """
    output_array = results.as_numpy(output_name)
    results = output_array[0]

    # Include special handling for non-batching models
    responses = []
    for results in output_array:
        if not batching:
            results = [results]

        #softmax
        results = softmax(results)

        # get sorted topk
        idx = np.argpartition(results, -topk)[-topk:]
        response = [Class(class_name=classes[i], score=float(results[i])) for i in idx]
        response.sort(key=lambda x: x.score, reverse=True)
        responses.append(response)
    return responses


def requestGenerator(batched_image_data, input_name, output_name, dtype):
    # Set the input data
    inputs = [httpclient.InferInput(input_name, batched_image_data.shape,
                                    dtype)]
    inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    outputs = []
    outputs.append(
            httpclient.InferRequestedOutput(output_name, binary_data=True))
    yield inputs, outputs


def inference_http(triton_client, img):
    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=MODEL_NAME, model_version=MODEL_VERSION)
    except InferenceServerException as e:
        raise Exception("failed to retrieve the metadata: " + str(e))

    try:
        model_config = triton_client.get_model_config(
            model_name=MODEL_NAME, model_version=MODEL_VERSION)
    except InferenceServerException as e:
        raise Exception("failed to retrieve the config: " + str(e))

    logger.info(f'Model metadata: {model_metadata}')
    logger.info(f'Model config: {model_config}')

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_http(
            model_metadata, model_config)

    # Preprocess the images into input data according to model
    # requirements
    # TODO scaling should be param
    image_data = [preprocess(img, format, dtype, c, h, w, 'INCEPTION')]

    # Send requests of batch_size=1 images. If the number of
    # images isn't an exact multiple of batch_size then just
    # start over with the first images until the batch is filled.
    # TODO batching
    responses = []

    sent_count = 0

    if max_batch_size > 0:
        batched_image_data = np.stack([image_data[0]], axis=0)
    else:
        batched_image_data = image_data[0]

    # Send request
    try:
        for inputs, outputs in requestGenerator(
                    batched_image_data, input_name, output_name, dtype):
            sent_count += 1
            responses.append(
                    triton_client.infer(MODEL_NAME,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=MODEL_VERSION,
                                        outputs=outputs))
    except InferenceServerException as e:
        logger.info("inference failed: " + str(e))

    final_responses = []
    for response in responses:
        this_id = response.get_response()["id"]
        logger.info("Request {}, batch size {}".format(this_id, 1))
        final_response = postprocess(response, output_name, 1,
                                     max_batch_size > 0)
        logger.info(final_response)
        final_responses.append(final_response)

    return final_responses
