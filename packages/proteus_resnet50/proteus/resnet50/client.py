# TODO clean up and split out
import numpy as np
import logging

from tritonclient.utils import triton_to_np_dtype, InferenceServerException
from pathlib import Path
from .helpers import read_class_names
from proteus.models import ClassificationModel

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

folder_path = Path(__file__).parent


class Resnet50(ClassificationModel):

    MODEL_NAME = 'resnet50'
    CHANNEL_FIRST = True
    CLASSES = read_class_names(f"{folder_path}/imagenet_labels.txt")


def inference_http(triton_client, img):
    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=Resnet50.MODEL_NAME, model_version=Resnet50.MODEL_VERSION)
    except InferenceServerException as e:
        raise Exception("failed to retrieve the metadata: " + str(e))

    try:
        model_config = triton_client.get_model_config(
            model_name=Resnet50.MODEL_NAME, model_version=Resnet50.MODEL_VERSION)
    except InferenceServerException as e:
        raise Exception("failed to retrieve the config: " + str(e))

    logger.info(f'Model metadata: {model_metadata}')
    logger.info(f'Model config: {model_config}')

    input_name, output_name, dtype = Resnet50.parse_model_http(model_metadata, model_config)

    # Preprocess the images into input data according to model
    # requirements
    # TODO scaling should be param
    image_data = [Resnet50.preprocess(img)]

    # Send requests of batch_size=1 images. If the number of
    # images isn't an exact multiple of batch_size then just
    # start over with the first images until the batch is filled.
    # TODO batching
    responses = []

    sent_count = 0

    if Resnet50.MAX_BATCH_SIZE > 0:
        batched_image_data = np.stack([image_data[0]], axis=0)
    else:
        batched_image_data = image_data[0]

    # Send request
    try:
        for inputs, outputs in Resnet50.requestGenerator(
                    batched_image_data, input_name, output_name, dtype):
            sent_count += 1
            responses.append(
                    triton_client.infer(Resnet50.MODEL_NAME,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=Resnet50.MODEL_VERSION,
                                        outputs=outputs))
    except InferenceServerException as e:
        logger.info("inference failed: " + str(e))

    final_responses = []
    for response in responses:
        this_id = response.get_response()["id"]
        logger.info("Request {}, batch size {}".format(this_id, 1))
        final_response = Resnet50.postprocess(response, output_name,
                                              Resnet50.CLASSES, 1,
                                              Resnet50.MAX_BATCH_SIZE > 0)
        logger.info(final_response)
        final_responses.append(final_response)

    return final_responses
