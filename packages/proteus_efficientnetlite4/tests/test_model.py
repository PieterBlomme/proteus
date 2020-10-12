import json
import random

import pytest
import requests
from proteus.datasets import ImageNette
from PIL import Image

model = "efficientnetlite4"


@pytest.fixture
def setup():
    response = requests.post("http://localhost/load", json.dumps({"name": model}))
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return ImageNette()


def test_speed(dataset):
    fpath, _ = dataset[0]
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.elapsed.total_seconds() < 0.25


def test_jpg():
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


def test_png():
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


def test_bmp():
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


def test_score(dataset):
    ids = [i for i in range(len(dataset))]
    ids = random.sample(ids, 100)

    correct = 0
    for i in ids:
        fpath, target = dataset[i]
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://localhost/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
        if response.json()[0][0][0]["class_name"].lower() == target:
            correct += 1

    assert correct > 70


@pytest.fixture
def teardown():
    response = requests.post("http://localhost/unload", json.dumps({"name": model}))
    assert response.json()["success"]
