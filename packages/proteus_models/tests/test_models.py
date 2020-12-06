import pytest
import requests

MAX_ACTIVE_MODELS = 3  # see docker-compose.yml.  can be different in production


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


def teardown_function(test_max_active_models):
    model_dict = requests.get(f"http://localhost/models").json()

    for model, desc in model_dict.items():
        response = requests.post(f"http://localhost/{model}/unload")
