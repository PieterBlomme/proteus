import glob
import os
import random
import tarfile
import tempfile
import urllib.request

import numpy as np
import requests
from PIL import Image

from .datasets import Dataset

tmpfolder = tempfile.gettempdir()


class BSDSSuperRes(Dataset):
    """
    Will yield resized (224,224) images of BSDS500
    For  evaluation, we resize to 572,572 and check average MSE
    It's not a very good metric, but it'll do ...
    """

    def __init__(self, k=50):
        self.maybe_download()
        files = glob.glob(f"{tmpfolder}/datasets/BSR/BSDS500/data/images/test/*.jpg")
        random.shuffle(files)
        self.files = files[:k]

    def maybe_download(self):
        if not os.path.isdir(f"{tmpfolder}/datasets/BSR"):
            print("Downloading BSDS500")
            thetarfile = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
            ftpstream = urllib.request.urlopen(thetarfile)
            thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
            thetarfile.extractall(path=f"{tmpfolder}/datasets")

    def __getitem__(self, index):
        fpath = self.files[index]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.open(fpath).resize((224, 224)).save(tmp.name)
        Image.open(tmp.name)
        return tmp.name, None

    def __len__(self):
        return len(self.files)

    def eval(self, preds):
        originals = [self.files[i] for i in range(self.__len__())]
        mses = []
        for original, pred in zip(originals, preds):
            np_original = np.array(Image.open(original).resize((672, 672)))
            np_pred = np.array(Image.open(pred))
            mse = np.square(np.subtract(np_original, np_pred)).mean()
            mses.append(mse)
        return np.array(mses).mean()
