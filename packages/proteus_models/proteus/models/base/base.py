import logging
import os
from shutil import copyfile

import requests
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")


class BaseModel:

    # Defaults
    MODEL_VERSION = "1"
    MAX_BATCH_SIZE = 1
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
            copyfile(cls.CONFIG_PATH, target_path)

    @classmethod
    def load_model(cls, triton_client):
        cls._maybe_download()
        triton_client.load_model(cls.__name__)

    @classmethod
    def load_model_info(cls, triton_client):
        """ 
        Function to be called to get model_metadata from Triton
        Useful if config.pbtxt is not available
        """
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=cls.__name__, model_version=cls.MODEL_VERSION
            )
        except InferenceServerException as e:
            raise Exception("failed to retrieve the metadata: " + str(e))

        raise Exception(f'Please create a config.pbtxt with model metadata {model_metadata}')

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
