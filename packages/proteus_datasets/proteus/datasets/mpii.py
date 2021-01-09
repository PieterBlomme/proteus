import tempfile

import numpy as np
import requests
from PIL import Image

from .imagenette import ImageNette

tmpfolder = tempfile.gettempdir()


class MPIIPoseEstimation(ImageNette):
    """
    Just a dummy implementation.  MPII dataset is huge and
    the evaluation code is written in matlab.  Can't be bothered to work out
    scoring code.  Using ImageNet instead
    """

    def eval(self, preds):
        return 0.0
