import logging
import os
import shutil

import numpy as np
import onnx
import requests
import tritonclient.http as httpclient
from jinja2 import Template
from onnxruntime.quantization import QuantType, quantize_dynamic
from tritonclient.utils import InferenceServerException

from .modelconfigs import BaseModelConfig, TritonOptimizationModelConfig

logger = logging.getLogger(__name__)


class ModelConfig(BaseModelConfig, TritonOptimizationModelConfig):
    pass


class BaseModel:
    """
    Abstract base for models.
    Submodels should:
    - implement preprocess classmethod to prepare an image
    - implement postprocess classmethod to parse results
    - define DESCRIPTION, MODEL_URL and some other parameters
    """

    # Defaults
    MODEL_VERSION = "1"
    MAX_BATCH_SIZE = 1
    MODEL_URL = ""
    CONFIG_PATH = None
    DESCRIPTION = "This is a model"
    CHANNEL_FIRST = False
    CLASSES = []
    INPUT_NAME = None
    OUTPUT_NAMES = None
    DTYPE = None

    MODEL_CONFIG = ModelConfig

    @classmethod
    def _maybe_download(cls):
        # Download model
        target_path = f"/tmp_models/{cls.__name__}/model.onnx"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if not os.path.isfile(target_path):
            url = cls.MODEL_URL
            r = requests.get(url)
            with open(target_path, "wb") as f:
                f.write(r.content)

    @classmethod
    def _maybe_quantize(cls):
        logger.info("Saving quantized model")
        # Download model
        model_fp32 = f"/tmp_models/{cls.__name__}/model.onnx"
        model_quant = f"/models/{cls.__name__}/1/model.onnx"
        os.makedirs(os.path.dirname(model_quant), exist_ok=True)

        url = cls.MODEL_URL
        r = requests.get(url)
        quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
        logger.info(f"Quantized model saved to:{model_quant}")
        full_size = os.path.getsize(model_fp32) / (1024 * 1024)
        quant_size = os.path.getsize(model_quant) / (1024 * 1024)
        logger.info(f"ONNX full precision model size (MB): {full_size}")
        logger.info(f"ONNX quantized model size (MB): {quant_size}")

    @classmethod
    def _write_model(cls):
        # Download model
        model_source = f"/tmp_models/{cls.__name__}/model.onnx"
        model_target = f"/models/{cls.__name__}/1/model.onnx"
        os.makedirs(os.path.dirname(model_target), exist_ok=True)

        shutil.copy(model_source, model_target)

    @classmethod
    def _request_generator(cls, batched_image_data):
        """ Set the input data """
        inputs = [
            httpclient.InferInput(cls.INPUT_NAME, batched_image_data.shape, cls.DTYPE)
        ]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput(output_name, binary_data=True)
            for output_name in cls.OUTPUT_NAMES
        ]
        yield inputs, outputs

    @classmethod
    def generate_config_file(cls, model_config):
        # Generate config
        targetfile = f"/models/{cls.__name__}/config.pbtxt"
        os.makedirs(os.path.dirname(targetfile), exist_ok=True)
        with open(cls.CONFIG_PATH) as f:
            template = Template(f.read())

        if getattr(model_config, "triton_optimization", False):
            triton_optimization = (
                "optimization { execution_accelerators {\n"
                '  gpu_execution_accelerator : [ { name : "tensorrt" } ]\n'
                "}}"
            )
        else:
            triton_optimization = ""

        if getattr(model_config, "dynamic_batching", False):
            dynamic_batching = "dynamic_batching { }"
        else:
            dynamic_batching = ""

        num_instances = getattr(model_config, "num_instances", 1)
        num_instances = "instance_group [ { count: " + str(num_instances) + " }]"

        with open(targetfile, "w") as fh:
            rendered_template = template.render(
                triton_optimization=triton_optimization,
                dynamic_batching=dynamic_batching,
                num_instances=num_instances,
            )
            logger.info(rendered_template)
            fh.write(rendered_template)

    @classmethod
    def load_model(cls, model_config, triton_client):
        """
        Download (if needed) model files and load model in Triton

        :param triton_client : the client to use
        """
        logger.debug(model_config)
        cls._maybe_download()

        if cls.CONFIG_PATH is None:
            # Load and get the metadata
            triton_client.load_model(cls.__name__)
            logger.info(cls.load_model_info(triton_client))
        else:
            cls.generate_config_file(model_config)
            if getattr(model_config, "quantize", False):
                # quantize model and write
                cls._maybe_quantize()
            else:
                cls._write_model()

            triton_client.load_model(cls.__name__)

    @classmethod
    def load_model_info(cls, triton_client):
        """
        Function to be called to get model_metadata from Triton
        Useful if config.pbtxt is not available

        :param triton_client : the client to use
        """
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=cls.__name__, model_version=cls.MODEL_VERSION
            )
            return model_metadata
        except InferenceServerException as e:
            raise Exception("failed to retrieve the metadata: " + str(e))

        raise Exception(
            f"Please create a config.pbtxt with model metadata {model_metadata}"
        )

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
            logger.error("inference failed: " + str(e))
            raise e

        final_responses = []
        for response in responses:
            this_id = response.get_response()["id"]
            logger.debug("Request {}, batch size {}".format(this_id, 1))
            final_response = cls.postprocess(
                response, (h, w), 1, cls.MAX_BATCH_SIZE > 0
            )
            final_responses.append(final_response)

        return final_responses
