import tempfile
import time

import pytest
import requests
from PIL import Image

MAX_ACTIVE_MODELS = 3  # see docker-compose.yml.  can be different in production
MODEL_INACTIVITY = 1  # see docker-compose.yml.  can be different in production


def test_max_active_models():
    """
    If max_active_models is reached, api should return a clean error message instead of 200 on model load
    """
    model_dict = requests.get(f"http://localhost/models").json()

    loaded_models = []
    for model, desc in model_dict.items():
        response = requests.post(f"http://localhost/{model}/load")
        if len(loaded_models) < MAX_ACTIVE_MODELS:
            assert response.status_code == requests.codes.ok
            assert response.json()["success"]
            loaded_models.append(model)
        else:
            assert response.status_code == requests.codes.forbidden
            break


def test_inactivity_no_requests():
    """
    If MODEL_INACTIVITY minutes is reached, model should be unloaded
    """
    model_dict = requests.get(f"http://localhost/models").json()

    for model, desc in model_dict.items():
        response = requests.post(f"http://localhost/{model}/load")

        # Model should be available for at least a grace period
        # Even though no predictions are done
        time.sleep(MODEL_INACTIVITY * 60)
        response = requests.get(f"http://localhost/models/status")
        model_status = [m for m in response.json() if m.get("name") == model][0]
        assert model_status.get("state", "UNAVAILABLE") == "READY"

        # But expire even then
        time.sleep(60)
        response = requests.get(f"http://localhost/models/status")
        model_status = [m for m in response.json() if m.get("name") == model][0]
        assert model_status.get("state", "UNAVAILABLE") != "READY"

        # cleanup
        requests.post(f"http://localhost/{model}/unload")
        break


def test_inactivity_with_request():
    """
    If MODEL_INACTIVITY minutes is reached, model should be unloaded
    """
    model_dict = requests.get(f"http://localhost/models").json()

    for model, desc in model_dict.items():
        response = requests.post(f"http://localhost/{model}/load").json()

        # Dummy prediction
        fpath = None
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            Image.new("RGB", (800, 1280)).save(tmp.name)

            with open(tmp.name, "rb") as f:
                jsonfiles = {"file": f}
                payload = {"file_id": fpath}
                response = requests.post(
                    f"http://localhost/{model}/predict",
                    files=jsonfiles,
                    data=payload,
                )
                assert response.status_code == requests.codes.ok

        # Model should be expire after a while
        time.sleep(MODEL_INACTIVITY * 60 + 60)
        response = requests.get(f"http://localhost/models/status")
        model_status = [m for m in response.json() if m.get("name") == model]
        assert model_status.get("state", "UNAVAILABLE") != "READY"

        # cleanup
        requests.post(f"http://localhost/{model}/unload")
        break


def teardown_function(test_max_active_models):
    model_dict = requests.get(f"http://localhost/models").json()

    for model, desc in model_dict.items():
        response = requests.post(f"http://localhost/{model}/unload")
