import json
import os
import random
import tempfile
from pathlib import Path

import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .datasets import Dataset

tmpfolder = tempfile.gettempdir()
# We want a good random sample from CocoVal
# But we want pseudorandom to get consistent test results
random.seed(42)


class CocoValBBox(Dataset):
    def __init__(self, k=50):
        self.maybe_download()
        self.coco = COCO(f"{tmpfolder}/datasets/coco/instances_val2017.json")
        self.cats = {
            cat["name"]: cat["id"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        imgs = self.coco.loadImgs(self.coco.getImgIds())
        random.shuffle(imgs)
        self.imgs = imgs[:k]

    def maybe_download(self):
        target = f"{tmpfolder}/datasets/coco/instances_val2017.json"
        if not os.path.isfile(target):
            Path(f"{tmpfolder}/datasets/coco").mkdir(parents=True, exist_ok=True)
            print("Downloading COCO validation json")
            url = "https://pieterblomme-models.s3.us-east-2.amazonaws.com/coco/instances_val2017.json"
            response = requests.get(url, allow_redirects=True)
            open(target, "wb").write(response.content)

        Path(f"{tmpfolder}/coco_imgs").mkdir(parents=True, exist_ok=True)

    def _getfile(self, url):
        filename = f"{tmpfolder}/coco_imgs/" + url.split("/")[-1]
        if not os.path.isfile(filename):
            print(f"Downloading {url}")
            response = requests.get(url, allow_redirects=True)
            open(filename, "wb").write(response.content)
        return filename

    def _prepare_preds(self, preds_in):
        preds_out = []
        for index, pred in enumerate(preds_in):
            img_id = self.imgs[index]["id"]
            for box in pred:
                try:
                    result = {
                        "image_id": img_id,
                        "category_id": self.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"],
                            box["y1"],
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                    }
                    preds_out.append(result)
                except Exception as e:
                    print(e)
        return preds_out

    def __getitem__(self, index):
        url = self.imgs[index]["coco_url"]
        fpath = self._getfile(url)
        return fpath, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def eval(self, preds):
        preds = self._prepare_preds(preds)
        with open(f"{tmpfolder}/results_coco.json", "w") as f:
            json.dump(preds, f)
        cocoDT = self.coco.loadRes(f"{tmpfolder}/results_coco.json")
        cocoEval = COCOeval(self.coco, cocoDT, "bbox")
        cocoEval.params.imgIds = [s["id"] for s in self.imgs]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]


class CocoValMask(CocoValBBox):
    def _prepare_preds(self, preds_in):
        preds_out = []
        for index, pred in enumerate(preds_in):
            img_id = self.imgs[index]["id"]
            for ann in pred:
                segm = ann["segmentation"]
                box = ann["bounding_box"]
                try:
                    result = {
                        "image_id": img_id,
                        "category_id": self.cats[box["class_name"]],
                        "score": segm["score"],
                        "bbox": [
                            box["x1"],
                            box["y1"],
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                        "segmentation": [segm["segmentation"]],
                    }
                    preds_out.append(result)
                except Exception as e:
                    print(e)
        return preds_out

    def eval(self, preds):
        preds = self._prepare_preds(preds)
        with open(f"{tmpfolder}/results_coco.json", "w") as f:
            json.dump(preds, f)
        cocoDT = self.coco.loadRes(f"{tmpfolder}/results_coco.json")
        cocoEval = COCOeval(self.coco, cocoDT, "segm")
        cocoEval.params.imgIds = [s["id"] for s in self.imgs]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]
