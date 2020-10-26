import logging
import os
from shutil import copyfile

import requests
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)


class BaseModel:

    # Defaults
    MODEL_VERSION = "1"
    MAX_BATCH_SIZE = 1
    NUM_OUTPUTS = 1
    MODEL_URL = ""
    CONFIG_PATH = None
    DESCRIPTION = "This is a model"
    input_name = None
    output_names = None
    dtype = None

    @classmethod
    def _maybe_download(cls):
        # Download model
        target_path = f"/models/{cls.__name__}/1/model.onnx"
        if not os.path.isfile(target_path):
            url = cls.MODEL_URL
            r = requests.get(url)
            try:
                os.mkdir(f"/models/{cls.__name__}")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"/models/{cls.__name__}/1")
            except Exception as e:
                print(e)
            with open(target_path, "wb") as f:
                f.write(r.content)
        # Download config
        target_path = f"/models/{cls.__name__}/config.pbtxt"
        if cls.CONFIG_PATH and not os.path.isfile(target_path):
            # if cls has MODEL_CONFIG:
            # no api calls to triton_client to load model metadata
            # load_model_info and parse_model_http should never be needed
            # this should, in its own, already give a performance boost
            copyfile(cls.CONFIG_PATH, target_path)

    @classmethod
    def load_model(cls, triton_client):
        cls._maybe_download()
        triton_client.load_model(cls.__name__)

    @classmethod
    def load_model_info(cls, triton_client):
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=cls.__name__, model_version=cls.MODEL_VERSION
            )
        except InferenceServerException as e:
            raise Exception("failed to retrieve the metadata: " + str(e))

        try:
            model_config = triton_client.get_model_config(
                model_name=cls.__name__, model_version=cls.MODEL_VERSION
            )
        except InferenceServerException as e:
            raise Exception("failed to retrieve the config: " + str(e))

        logger.info(f"Model metadata: {model_metadata}")
        logger.info(f"Model config: {model_config}")

        cls.input_name, cls.output_names, cls.dtype = cls.parse_model_http(
            model_metadata, model_config
        )
        logger.info(f"Input name: {cls.input_name}")
        logger.info(f"output names: {cls.output_names}")
        logger.info(f"dtype: {cls.dtype}")

    @classmethod
    def parse_model_http(cls, model_metadata, model_config):
        """
        TODO check if this is still relevant
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata["inputs"]) != 1:
            raise Exception(
                "expecting 1 input, got {}".format(len(model_metadata["inputs"]))
            )
        if len(model_metadata["outputs"]) != cls.NUM_OUTPUTS:
            raise Exception(
                "expecting {} outputs, got {}".format(
                    cls.NUM_OUTPUTS, len(model_metadata["outputs"])
                )
            )

        if len(model_config["input"]) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config["input"])
                )
            )

        input_metadata = model_metadata["inputs"][0]
        output_metadatas = model_metadata["outputs"]

        for output_metadata in output_metadatas:
            if output_metadata["datatype"] != "FP32":
                raise Exception(
                    "expecting output datatype to be FP32, model '"
                    + model_metadata["name"]
                    + "' output type is "
                    + output_metadata["datatype"]
                )

        # sort to make sure that order of outputs is fixed
        output_metadatas = sorted(
            [output_metadata["name"] for output_metadata in output_metadatas]
        )

        # Model input must have 3 dims (not counting the batch dimension),
        # either CHW or HWC
        input_batch_dim = cls.MAX_BATCH_SIZE > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata["shape"]) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims,
                    model_metadata["name"],
                    len(input_metadata["shape"]),
                )
            )

        return (input_metadata["name"], output_metadatas, input_metadata["datatype"])

    @classmethod
    def requestGenerator(cls, batched_image_data, input_name, output_names, dtype):
        """ Set the input data """
        inputs = [httpclient.InferInput(input_name, batched_image_data.shape, dtype)]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput(output_name, binary_data=True)
            for output_name in output_names
        ]
        yield inputs, outputs
