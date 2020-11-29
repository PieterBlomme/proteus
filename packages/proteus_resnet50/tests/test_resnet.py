import tempfile

import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from proteus.datasets import ImageNette


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


def test_health():
    for i in range(10):
        try:
            response = requests.get("http://localhost/health")
            if response.status_code == requests.codes.ok:
                return
        except:
            time.sleep(25)


@pytest.fixture
def model():
    test_health()
    model = "Resnet50V2"
    response = requests.post(f"http://localhost/{model}/load")
    assert response.json()["success"]

    yield model
    response = requests.post(f"http://localhost/{model}/unload")
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return ImageNette(k=500)


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


@pytest.mark.long_running
def test_score(dataset, model):
    preds = []
    for fpath, target in dataset:
        response = get_prediction(fpath, model)
        preds.append(response.json()[0])

    score = dataset.eval(preds)
    print(f"Accuracy: {score}")
    assert score >= 0.67


def test_resize(dataset, model):
    # mAP should be similar after increasing image size
    preds_normal = []
    preds_resize = []
    for fpath, img in dataset:
        response = get_prediction(fpath, model)
        preds_normal.append(response.json()[0])

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            resize_path = tmp.name
        tmp_img.resize((w * 2, h * 2)).save(resize_path)
        response = get_prediction(resize_path, model)
        preds_resize.append(response.json()[0])

    score_normal = dataset.eval(preds_normal)
    score_resize = dataset.eval(preds_resize)
    print(f"Resize diff: {abs(score_normal - score_resize)}")
    assert abs(score_normal - score_resize) < 0.025  # 2% diff seems acceptable


def test_padding(dataset, model):
    # mAP should be similar after padding to a square
    preds_normal = []
    preds_padded = []
    for fpath, img in dataset:
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

    score_normal = dataset.eval(preds_normal)
    score_padded = dataset.eval(preds_padded)
    print(f"Padding diff: {abs(score_normal - score_padded)}")
    assert abs(score_normal - score_padded) < 0.1  # 10% diff seems acceptable
