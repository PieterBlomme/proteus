from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
import tritonclient.http as httpclient
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

#TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

app = FastAPI()

#set up Triton connection
TRITONURL = 'host.docker.internal:8000'

try:
    # Specify large enough concurrency to handle the
    # the number of requests.
    concurrency = 1
    triton_client = httpclient.InferenceServerClient(
                    url=TRITONURL, concurrency=concurrency, verbose=True)
    logger.info(f'Server ready? {triton_client.is_server_ready()}')
except Exception as e:
    logger.error("client creation failed: " + str(e))


@app.get("/health")
async def get_server_health():
    if triton_client.is_server_live():
        logger.info(f'Server is alive')
        return {"success": True}
    else:
        logger.info(f'Server is dead')
        return {"success": False}

@app.get("/models")
async def get_model_repository():
    return triton_client.get_model_repository_index()

@app.post("/load/")
async def load_model(model: str):
    if model == 'yolov4':
        logger.info(f'Loading model {model}')
        triton_client.load_model(model)
        if not triton_client.is_model_ready(model):
            return {"success": False, "message": f"model {model} not ready - check logs"}
        else:
            return {"success": True, "message": f"model {model} loaded"}
    else:
        return {"success": False, "message": "unknown model"}

@app.post("/unload/")
async def unload_model(model: str):
    if not triton_client.is_model_ready(model):
        logger.info(f'No model with name {model} loaded')
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f'Unloading model {model}')
        triton_client.unload_model(model)
        return {"success": True, "message": f"model {model} unloaded"}

import argparse
import numpy as np
from PIL import Image
import sys
from functools import partial
import os

import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

MODEL_NAME = 'yolov4'
MODEL_VERSION = 1


FLAGS = None

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

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (max_batch_size > 0)
    non_one_cnt = 0
    for output_metadata in all_output_metadata:
        for dim in output_metadata['shape']:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    #TODO
                    print ("expecting model output to be a vector")

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))

    #FORMAT_NHWC
    h = input_metadata['shape'][1 if input_batch_dim else 0]
    w = input_metadata['shape'][2 if input_batch_dim else 1]
    c = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'], [output_metadata['name'] for output_metadata in all_output_metadata], c,
            h, w, input_config['format'], input_metadata['datatype'])


def preprocess(img, format, dtype, c, h, w, scaling):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    TODO: yolov4 preprocess https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    logger.info(sample_img.size)
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

from scipy import special
import colorsys
import random


def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    '''define anchor boxes'''
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def print_bbox(bboxes, classes=read_class_names("coco.names"), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # random.seed(0)
    # random.shuffle(colors)
    # random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        # fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        # bbox_color = colors[class_ind]
        # bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        # cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            # t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            # cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  

            # cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        # fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    logger.info(c1, c2, bbox_mess)

def postprocess(results, output_names, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    detections = [results.as_numpy(output_name) for output_name in output_names]
    ANCHORS = "./yolov4_anchors.txt"
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05]

    ANCHORS = get_anchors(ANCHORS)
    STRIDES = np.array(STRIDES)

    original_image_size = (3361, 2521) #TODO
    input_size = 416
    pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')
    print_bbox(bboxes)
    return bboxes


def requestGenerator(batched_image_data, input_name, output_names, dtype):
    # Set the input data
    inputs = []
    inputs.append(
            tritonhttpclient.InferInput(input_name, batched_image_data.shape,
                                        dtype))
    inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

    outputs = []
    for output_name in output_names:
        outputs.append(
                tritonhttpclient.InferRequestedOutput(output_name,
                                                    binary_data=True))

    yield inputs, outputs, MODEL_NAME, MODEL_VERSION

def predict(triton_client, img):
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

    logger.info(model_metadata)
    logger.info(model_config)
    
    max_batch_size, input_name, output_names, c, h, w, format, dtype = parse_model_http(
            model_metadata, model_config)

    # Preprocess the images into input data according to model
    # requirements
    image_data = [preprocess(img, format, dtype, c, h, w, 'INCEPTION')]#TODO scaling should be param

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0

    while not last_request:
        input_filenames = []
        repeated_image_data = []

        for idx in range(1):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
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
        final_response = postprocess(response, output_names, 1, max_batch_size > 0)
        final_responses.append(final_response)

    return final_response

@app.post("/{model}/predict")
async def predict(model: str, file: bytes = File(...)):
    if not triton_client.is_model_ready(model):
        raise HTTPException(status_code=404, detail="model not available")

    #TODO validation of the file
    try:
        img = Image.open(BytesIO(file))
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file",
        )
    response = predict(triton_client, img)
    logger.info(response)
    return {"img": img.size}