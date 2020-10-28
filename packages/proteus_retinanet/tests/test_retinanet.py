import json

import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from proteus.datasets import CocoVal


@pytest.fixture
def model():
    model = "RetinaNet"
    response = requests.post("http://localhost/load", json.dumps({"name": model}))
    assert response.json()["success"]

    yield model
    response = requests.post("http://localhost/unload", json.dumps({"name": model}))
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return CocoVal(k=50)


@pytest.fixture
def small_dataset():
    return CocoVal(k=10)


def test_speed(dataset, model):
    fpath, _ = dataset[0]
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.elapsed.total_seconds() < 25.0


def test_jpg(model):
    fpath = "image.jpg"
    Image.new("RGB", (800, 1280)).save(fpath)

    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.status_code == requests.codes.ok


def test_png(model):
    fpath = "image.png"
    Image.new("RGBA", (800, 1280)).save(fpath)

    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.status_code == requests.codes.ok


def test_bmp(model):
    fpath = "image.bmp"
    Image.new("RGB", (800, 1280)).save(fpath)

    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.status_code == requests.codes.ok


@pytest.mark.long_running
def test_score(dataset, model):
    preds = []
    for (fpath, img) in dataset:
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
            for box in response.json()[0]:
                try:
                    result = {
                        "image_id": img["id"],
                        "category_id": dataset.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"],
                            box["y1"],
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                    }
                    if box["score"] > 0.2:
                        preds.append(result)
                except Exception as e:
                    print(e)
    mAP = dataset.eval(preds)
    assert mAP > 0.25


def test_resize(small_dataset, model):
    # mAP should be similar after doubling image size
    preds_normal = []
    preds_resize = []
    for (fpath, img) in small_dataset:
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
            for box in response.json()[0]:
                try:
                    result = {
                        "image_id": img["id"],
                        "category_id": small_dataset.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"],
                            box["y1"],
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                    }
                    if box["score"] > 0.2:
                        preds_normal.append(result)
                except Exception as e:
                    print(e)

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        tmp_img.resize((w * 2, h * 2)).save(fpath)
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
            for box in response.json()[0]:
                # half every box
                try:
                    result = {
                        "image_id": img["id"],
                        "category_id": small_dataset.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"] / 2,
                            box["y1"] / 2,
                            box["x2"] / 2 - box["x1"] / 2,
                            box["y2"] / 2 - box["y1"] / 2,
                        ],
                    }
                    if box["score"] > 0.2:
                        preds_resize.append(result)
                except Exception as e:
                    print(e)
    mAP_normal = small_dataset.eval(preds_normal)
    mAP_resize = small_dataset.eval(preds_resize)
    print(abs(mAP_normal - mAP_resize))
    assert abs(mAP_normal - mAP_resize) < 0.02  # 2% diff seems acceptable


def test_padding(small_dataset, model):
    # mAP should be similar after padding to a square
    preds_normal = []
    preds_padded = []
    for (fpath, img) in small_dataset:
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
            for box in response.json()[0]:
                try:
                    result = {
                        "image_id": img["id"],
                        "category_id": small_dataset.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"],
                            box["y1"],
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                    }
                    if box["score"] > 0.2:
                        preds_normal.append(result)
                except Exception as e:
                    print(e)

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        target = max((w, h))
        dw = (target - w) / 2
        dh = (target - h) / 2
        tmp_img = pad(tmp_img, (target, target))
        tmp_img.save(fpath)
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
            for box in response.json()[0]:
                try:
                    result = {
                        "image_id": img["id"],
                        "category_id": small_dataset.cats[box["class_name"]],
                        "score": box["score"],
                        "bbox": [
                            box["x1"] - dw,
                            box["y1"] - dh,
                            box["x2"] - box["x1"],
                            box["y2"] - box["y1"],
                        ],
                    }
                    if box["score"] > 0.2:
                        preds_padded.append(result)
                except Exception as e:
                    print(e)
    mAP_normal = small_dataset.eval(preds_normal)
    mAP_padded = small_dataset.eval(preds_padded)
    print(abs(mAP_normal - mAP_padded))
    assert abs(mAP_normal - mAP_padded) < 0.05  # 5% diff seems acceptable
