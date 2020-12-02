import json
import tempfile

import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from proteus.datasets import {{cookiecutter.test_dataset}}


def get_prediction(fpath, model):
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    return response


@pytest.fixture
def model():
    model = "{{cookiecutter.model_name}}"
    response = requests.post(f"http://localhost/{model}/load")
    assert response.json()["success"]

    yield model
    response = requests.post(f"http://localhost/{model}/unload")
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return {{cookiecutter.test_dataset}}(k=100)


@pytest.fixture
def small_dataset():
    return {{cookiecutter.test_dataset}}(k=10)


def test_jpg(model):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        Image.new("RGB", (800, 1280)).save(tmp.name)
        response = get_prediction(tmp.name, model)
    assert response.status_code == requests.codes.ok


def test_png(model):
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.new("RGB", (800, 1280)).save(tmp.name)
        response = get_prediction(tmp.name, model)
    assert response.status_code == requests.codes.ok


def test_bmp(model):
    with tempfile.NamedTemporaryFile(suffix=".bmp") as tmp:
        Image.new("RGB", (800, 1280)).save(tmp.name)
        response = get_prediction(tmp.name, model)
    assert response.status_code == requests.codes.ok


@pytest.mark.slow
def test_score(dataset, model):
    preds = []
    for (fpath, img) in dataset:
        response = get_prediction(fpath, model)
        result = [box for box in response.json()[0]]
        preds.append(result)
    score = dataset.eval(preds)
    assert score > 0.0