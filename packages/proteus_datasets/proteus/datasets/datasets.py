import glob
import json
import os
import random
import tarfile
import tempfile
import urllib.request

import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

tmpfolder = tempfile.gettempdir()


class Dataset:
    """
    Dataset interface:
    - implement __getitem__ to generate files
    - implement eval(preds) to predict a score
    """

    def __getitem__(self, index):
        None

    def __len__(self):
        return 1

    def eval(self, preds):
        return 1.0
