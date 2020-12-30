import itertools
import tempfile
import time

import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from proteus.datasets import ImageNette
from proteus.models.base.base import ModelConfig # probably should have his own version

MODEL = "MobileNetV2"

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
    return ImageNette(k=1000)


@pytest.fixture
def small_dataset():
    return ImageNette(k=250)


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
    for fpath, target in dataset:
        response = get_prediction(fpath, model)
        preds.append(response.json()[0])

    score = dataset.eval(preds)
    print(f"Accuracy: {score}")
    assert score >= 0.47


@pytest.mark.slow
def test_resize(small_dataset, model):
    # mAP should be similar after increasing image size
    preds_normal = []
    preds_resize = []
    for fpath, img in small_dataset:
        response = get_prediction(fpath, model)
        preds_normal.append(response.json()[0])

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            resize_path = tmp.name
        tmp_img.resize((w * 2, h * 2)).save(resize_path)
        response = get_prediction(resize_path, model)
        preds_resize.append(response.json()[0])

    score_normal = small_dataset.eval(preds_normal)
    score_resize = small_dataset.eval(preds_resize)
    print(f"Resize diff: {abs(score_normal - score_resize)}")
    assert abs(score_normal - score_resize) < 0.025  # 2% diff seems acceptable


@pytest.mark.slow
def test_padding(small_dataset, model):
    # mAP should be similar after padding to a square
    preds_normal = []
    preds_padded = []
    for fpath, img in small_dataset:
        response = get_prediction(fpath, model)
        preds_normal.append(response.json()[0])

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        target = max((w, h))
        dw = (target - w) / 2
        dh = (target - h) / 2
        tmp_img = pad(tmp_img, (target, target))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            padded_path = tmp.name
        tmp_img.save(padded_path)
        response = get_prediction(padded_path, model)
        preds_padded.append(response.json()[0])

    score_normal = small_dataset.eval(preds_normal)
    score_padded = small_dataset.eval(preds_padded)
    print(f"Padding diff: {abs(score_normal - score_padded)}")
    assert abs(score_normal - score_padded) < 0.15  # 15% diff seems acceptable
