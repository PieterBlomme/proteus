import json

import pytest
import requests
from PIL import Image
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
