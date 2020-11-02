import logging
from pathlib import Path

from proteus.models.base import BaseModel

folder_path = Path(__file__).parent
logger = logging.getLogger(__name__)


class {{cookiecutter.model_name}}(BaseModel):

    DESCRIPTION = (
        "Description for model goes here"
    )
    MODEL_URL = None
    CONFIG_PATH = f"{folder_path}/config.pbtxt"
    INPUT_NAME = None
    OUTPUT_NAMES = None
    DTYPE = None

    @classmethod
    def preprocess(cls, img):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        :param img: image as array in HWC format

        :return: processed image
        """
        return img

    @classmethod
    def postprocess(cls, results, original_image_size, batch_size, batching):
        """
        Post-process results to return valid outputs.
        """
        return results
