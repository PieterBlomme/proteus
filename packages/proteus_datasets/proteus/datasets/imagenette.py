import glob
import json
import os
import random
import tarfile
import tempfile
import urllib.request

import requests

from .datasets import Dataset

tmpfolder = tempfile.gettempdir()


class ImageNette(Dataset):

    lbl_dict = dict(
        n01440764="tench",
        n02102040="english springer",
        n02979186="cassette player",
        n03000684="chain saw",
        n03028079="church",
        n03394916="french horn",
        n03417042="garbage truck",
        n03425413="gas pump",
        n03445777="golf ball",
        n03888257="parachute",
    )

    def __init__(self, k=50):
        self.maybe_download()
        files = glob.glob(f"{tmpfolder}/datasets/imagenette2-320/val/*/*")
        random.shuffle(files)
        self.files = files[:k]

    def maybe_download(self):
        if not os.path.isdir(f"{tmpfolder}/datasets/imagenette2-320"):
            print("Downloading ImageNette 320 px")
            thetarfile = (
                "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
            )
            ftpstream = urllib.request.urlopen(thetarfile)
            thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
            thetarfile.extractall(path=f"{tmpfolder}/datasets")

    def __getitem__(self, index):
        fpath = self.files[index]
        synset = fpath.split("/")[-2]
        target = self.lbl_dict[synset].lower()
        return fpath, target

    def __len__(self):
        return len(self.files)

    def eval(self, preds):
        preds = [pred[0][0]["class_name"].lower() for pred in preds]
        targets = [self.__getitem__(i)[1] for i in range(self.__len__())]
        correct = 0
        for p, t in zip(preds, targets):
            if p == t:
                correct += 1
        return correct / self.__len__()
