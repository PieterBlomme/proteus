import json
import random

import pytest
import requests
from proteus.datasets import ImageNette

model = "efficientnetlite4"


@pytest.fixture
def dataset():
    return ImageNette()


def test_speed(dataset):
    response = requests.post(
        "http://host.docker.internal/load", json.dumps({"name": model})
    )
    assert response.json()["success"]

    fpath, _ = dataset[0]
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://host.docker.internal/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    assert response.elapsed.total_seconds() < 1.0

    response = requests.post(
        "http://host.docker.internal/unload", json.dumps({"name": model})
    )
    assert response.json()["success"]


def test_score(dataset):
    response = requests.post(
        "http://host.docker.internal/load", json.dumps({"name": model})
    )
    assert response.json()["success"]

    ids = [i for i in range(len(dataset))]
    ids = random.sample(ids, 100)

    correct = 0
    for i in ids:
        fpath, target = dataset[i]
        with open(fpath, "rb") as f:
            jsonfiles = {"file": f}
            payload = {"file_id": fpath}
            response = requests.post(
                f"http://host.docker.internal/{model}/predict",
                files=jsonfiles,
                data=payload,
            )
        if response.json()[0][0][0]["class_name"].lower() == target:
            correct += 1

    assert correct > 70

    response = requests.post(
        "http://host.docker.internal/unload", json.dumps({"name": model})
    )
    assert response.json()["success"]
