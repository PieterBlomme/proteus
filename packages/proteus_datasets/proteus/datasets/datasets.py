import glob
import json
import os
import random
import tarfile
import urllib.request

import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ImageNette:

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

    def maybe_download(self):
        if not os.path.isdir("./datasets/imagenette2-320"):
            print("Downloading ImageNette 320 px")
            thetarfile = (
                "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
            )
            ftpstream = urllib.request.urlopen(thetarfile)
            thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
            thetarfile.extractall(path="./datasets")

    def __init__(self):
        self.maybe_download()
        self.files = glob.glob("./datasets/imagenette2-320/val/*/*")

    def __getitem__(self, index):
        fpath = self.files[index]
        synset = fpath.split("/")[-2]
        target = self.lbl_dict[synset].lower()
        return fpath, target

    def __len__(self):
        return len(self.files)


class CocoVal:
    def getfile(self, url):
        filename = "./coco_imgs/" + url.split("/")[-1]
        if not os.path.isdir("./coco_imgs"):
            try:
                os.mkdir("./coco_imgs")
            except Exception as e:
                print(e)
        if not os.path.isfile(filename):
            response = requests.get(url, allow_redirects=True)
            open(filename, "wb").write(response.content)
        return filename

    def maybe_download(self):
        target = "./datasets/coco/instances_val2017.json"
        if not os.path.isfile(target):
            try:
                os.mkdir("./datasets")
            except Exception as e:
                print(e)
            try:
                os.mkdir("./datasets/coco")
            except Exception as e:
                print(e)
            print("Downloading COCO validation json")
            url = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/coco/instances_val2017.json"
            response = requests.get(url, allow_redirects=True)
            open(target, "wb").write(response.content)

    def __init__(self, k=50):
        self.maybe_download()
        self.coco = COCO("./datasets/coco/instances_val2017.json")
        self.cats = {
            cat["name"]: cat["id"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        imgs = self.coco.loadImgs(self.coco.getImgIds())
        self.imgs = random.sample(imgs, k)

    def __getitem__(self, index):
        url = self.imgs[index]["coco_url"]
        fpath = self.getfile(url)
        return fpath, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def eval(self, preds):
        with open("results_coco.json", "w") as f:
            json.dump(preds, f)
        cocoDT = self.coco.loadRes("results_coco.json")
        cocoEval = COCOeval(self.coco, cocoDT, "bbox")
        cocoEval.params.imgIds = [s["id"] for s in self.imgs]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print(cocoEval.stats)
        return cocoEval.stats[1]
