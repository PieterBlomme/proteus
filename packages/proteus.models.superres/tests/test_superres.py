import itertools
import tempfile
import time

import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from io import BytesIO
from proteus.datasets import BSDSSuperRes
from proteus.models.superres.client import ModelConfig

MODEL = "SuperResolution"

# Check liveness
for i in range(10):
    try:
        response = requests.get("http://localhost/health")
        if response.status_code == requests.codes.ok:
            break
    except:
        time.sleep(25)


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
    payload = {"triton_optimization": True}
    response = requests.post(
        f"http://localhost/{MODEL}/load",
        json=payload,
    )
    assert response.json()["success"]

    yield MODEL
    response = requests.post(f"http://localhost/{MODEL}/unload")
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return BSDSSuperRes(k=10)


@pytest.fixture
def small_dataset():
    return BSDSSuperRes(k=10)


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


def test_modelconfig():
    # Figure out which config parameters are defined
    schema = ModelConfig().dict()

    # Find all combinations that we want to test
    test_parameters = []
    test_values = []
    for k, v in schema.items():
        test_parameters.append(k)
        if type(v) == bool:
            test_values.append([True, False])
        elif type(v) == int:
            test_values.append([1, 2])
        else:
            raise NotImplementedError(
                f"Config parameter of type {type(v)} not yet implemented"
            )
    test_combinations = list(itertools.product(*test_values))

    # Test load + prediction for each combination
    for test_config in test_combinations:
        mc = {k: v for k, v in zip(test_parameters, test_config)}
        response = requests.post(
            f"http://localhost/{MODEL}/load",
            json=mc,
        )
        assert response.status_code == requests.codes.ok

        with tempfile.NamedTemporaryFile(suffix=".bmp") as tmp:
            Image.new("RGB", (800, 1280)).save(tmp.name)
            response = get_prediction(tmp.name, MODEL)
            assert response.status_code == requests.codes.ok

        response = requests.post(f"http://localhost/{MODEL}/unload")
        assert response.status_code == requests.codes.ok


@pytest.mark.slow
def test_score(dataset, model):
    preds = []
    for (fpath, img) in dataset:
        response = get_prediction(fpath, model)
        img_byte_arr = BytesIO(response.content)
        img_byte_arr.seek(0)  # important here!
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.open(img_byte_arr).save(tmp.name)
            preds.append(tmp.name)
            
    score = dataset.eval(preds)
    assert score < 100.0
