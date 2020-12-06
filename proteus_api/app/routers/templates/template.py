import importlib
import logging
import os
import pkgutil
from io import BytesIO

import proteus.models
import tritonclient.http as httpclient
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException
from PIL import Image
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

logger = logging.getLogger(__name__)

router = APIRouter()


def get_model_dict():
    # discover models
    def iter_namespace(ns_pkg):
        # Specifying the second argument (prefix) to iter_modules makes the
        # returned name an absolute name instead of a relative one. This allows
        # import_module to work without having to do additional modification to
        # the name.
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    model_dict = {}
    for finder, name, ispkg in iter_namespace(proteus.models):
        module = importlib.import_module(name)
        model_dict.update(module.model_dict)
    return model_dict


def get_triton_client():
    # set up Triton connection
    TRITONURL = "triton:8000"
    # TODO check that always available ...
    try:
        # Specify large enough concurrency to handle the
        # the number of requests.
        concurrency = 1
        triton_client = httpclient.InferenceServerClient(
            url=TRITONURL, concurrency=concurrency
        )
        logger.info(f"Server ready? {triton_client.is_server_ready()}")
    except Exception as e:
        logger.error("client creation failed: " + str(e))
    return triton_client


triton_client = get_triton_client()
model_dict = get_model_dict()


@router.post(f"/load")
async def load_model():
    # Check if there's room for more models
    max_active_models = int(os.environ.get("MAX_ACTIVE_MODELS", "1"))
    loaded_models = [
        m.get("name")
        for m in triton_client.get_model_repository_index()
        if m.get("state", "UNAVAILABLE") == "READY"
    ]
    if len(loaded_models) >= max_active_models:
        raise HTTPException(
            status_code=403,
            detail=f"Max active models ({max_active_models}) reached.  A model needs to be unloaded before adding another.",
        )

    name = "{{name}}"
    model = model_dict[name]
    try:
        logger.info(f"Loading model {{name}}")
        model.load_model(triton_client)

        if not triton_client.is_model_ready(name):
            return {
                "success": False,
                "message": f"model {{name}} not ready - check logs",
            }
        else:
            return {"success": True, "message": f"model {{name}} loaded"}
    except ImportError as e:
        logger.info(e)
        return {"success": False, "message": f"unknown model {{name}}"}


@router.post(f"/unload")
async def unload_model():
    name = "{{name}}"
    if not triton_client.is_model_ready(name):
        logger.info(f"No model with name {{name}} loaded")
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f"Unloading model {{name}}")
        triton_client.unload_model(name)
        return {"success": True, "message": f"model {{name}} unloaded"}


@router.post(f"/predict")
async def predict(file: bytes = File(...)):
    name = "{{name}}"
    model = model_dict[name]

    if not triton_client.is_model_ready(name):
        raise HTTPException(status_code=404, detail="model not available")

    # TODO validation of the file
    try:
        img = Image.open(BytesIO(file))
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to process file",
        )
    response = model.inference_http(triton_client, img)
    return response
