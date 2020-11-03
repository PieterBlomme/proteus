import json
import tempfile
import time
import pytest
import requests
from PIL import Image
from PIL.ImageOps import pad
from proteus.datasets import CocoValBBox


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
            response = requests.get('http://localhost/health')
            if response.status_code == requests.codes.ok:
                return
        except:
            time.sleep(25)

@pytest.fixture
def model():
    test_health()
    model = "EfficientDetD2"
    response = requests.post("http://localhost/load", json.dumps({"name": model}))
    assert response.json()["success"]

    yield model
    response = requests.post("http://localhost/unload", json.dumps({"name": model}))
    assert response.json()["success"]


@pytest.fixture
def dataset():
    return CocoValBBox(k=100)


@pytest.fixture
def small_dataset():
    return CocoValBBox(k=10)


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
    for (fpath, img) in dataset:
        response = get_prediction(fpath, model)
        result = [box for box in response.json()[0]]
        preds.append(result)
    mAP = dataset.eval(preds)
    print(f"mAP score: {mAP}")
    assert mAP > 0.41


def test_resize(small_dataset, model):
    # mAP should be similar after increasing image size
    preds_normal = []
    preds_resize = []
    for (fpath, img) in small_dataset:
        response = get_prediction(fpath, model)
        result = [box for box in response.json()[0]]
        preds_normal.append(result)

        tmp_img = Image.open(fpath)
        w, h = tmp_img.size
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            resize_path = tmp.name
        tmp_img.resize((w * 2, h * 2)).save(resize_path)
        response = get_prediction(resize_path, model)

        result = [box for box in response.json()[0]]
        # half every box:
        for box in result:
            box["x1"] /= 2
            box["y1"] /= 2
            box["x2"] /= 2
            box["y2"] /= 2
        preds_resize.append(result)

    mAP_normal = small_dataset.eval(preds_normal)
    mAP_resize = small_dataset.eval(preds_resize)
    print(f"Resize diff: {abs(mAP_normal - mAP_resize)}")
    assert abs(mAP_normal - mAP_resize) < 0.02  # 2% diff seems acceptable


def test_padding(small_dataset, model):
    # mAP should be similar after padding to a square
    preds_normal = []
    preds_padded = []
    for (fpath, img) in small_dataset:
        response = get_prediction(fpath, model)
        result = [box for box in response.json()[0]]
        preds_normal.append(result)

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
        result = [box for box in response.json()[0]]
        # half every box:
        for box in result:
            box["x1"] -= dw
            box["y1"] -= dh
            box["x2"] -= dw
            box["y2"] -= dh

        preds_padded.append(result)
    mAP_normal = small_dataset.eval(preds_normal)
    mAP_padded = small_dataset.eval(preds_padded)
    print(f"Padding diff: {abs(mAP_normal - mAP_padded)}")
    assert abs(mAP_normal - mAP_padded) < 0.05  # 5% diff seems acceptable
