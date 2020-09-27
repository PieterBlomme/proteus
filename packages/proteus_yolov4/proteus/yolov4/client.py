# TODO clean up and split out
import numpy as np
from PIL import Image
import logging

import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype, InferenceServerException

from .helpers import get_anchors, postprocess_bbbox, postprocess_boxes, nms, print_bbox


MODEL_NAME = 'yolov4'
MODEL_VERSION = '1'

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 3:
        raise Exception("expecting 3 outputs, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    all_output_metadata = model_metadata['outputs']

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

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

    # FORMAT_NHWC
    h = input_metadata['shape'][1 if input_batch_dim else 0]
    w = input_metadata['shape'][2 if input_batch_dim else 1]
    c = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], [output_metadata['name'] 
            for output_metadata in all_output_metadata], c,
            h, w, input_config['format'], input_metadata['datatype'])


def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    TODO: yolov4 preprocess https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
    """
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    logger.info(f'Original image size: {sample_img.size}')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == "FORMAT_NCHW":
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, original_image_size, output_names, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    detections = [results.as_numpy(output_name) for
                  output_name in output_names]
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05]

    ANCHORS = get_anchors()
    STRIDES = np.array(STRIDES)

    input_size = 416
    pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size,
                               input_size, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')
    bboxes = print_bbox(bboxes)
    return bboxes


def requestGenerator(batched_image_data, input_name, output_names, dtype):
    # Set the input data
    inputs = [httpclient.InferInput(input_name, batched_image_data.shape,
                                    dtype)]
    inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    outputs = []
    for output_name in output_names:
        outputs.append(
                httpclient.InferRequestedOutput(output_name,
                                                binary_data=True))

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

    max_batch_size, input_name, output_names, c, h, w, format, dtype = parse_model_http(
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
                    batched_image_data, input_name, output_names, dtype):
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
        final_response = postprocess(response, img.size, output_names, 1,
                                     max_batch_size > 0)
        final_responses.append(final_response)

    return final_responses
