import importlib
import logging
import os
import pkgutil
from io import BytesIO

import proteus.models
import tritonclient.http as httpclient
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

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
        CONCURRENCY = int(os.environ.get("TRITON_CONCURRENCY", "1"))
        TRITON_CLIENT_TIMEOUT = int(os.environ.get("TRITON_CLIENT_TIMEOUT", "300"))
        triton_client = httpclient.InferenceServerClient(
            url=TRITONURL,
            concurrency=CONCURRENCY,
            connection_timeout=TRITON_CLIENT_TIMEOUT,
            network_timeout=TRITON_CLIENT_TIMEOUT,
        )
        logger.info(f"Concurrency set to {CONCURRENCY}")
        logger.info(f"TritonClient timeout set to {TRITON_CLIENT_TIMEOUT}")
        logger.info(f"Server ready? {triton_client.is_server_ready()}")
    except Exception as e:
        logger.error("client creation failed: " + str(e))
    return triton_client


triton_client = get_triton_client()
model_dict = get_model_dict()
config_class = model_dict["{{name}}"].MODEL_CONFIG


@router.post(f"/load")
async def load_model(model_config: config_class):
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
        logging.getLogger("predictions").info("{{name}}|LOADING")
        model.load_model(model_config, triton_client)
        logging.getLogger("predictions").info("{{name}}|LOADED")

        if not triton_client.is_model_ready(name):
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Triton could not load model",
            )
        else:
            model_config = model.load_model_info(triton_client)
            return {
                "success": True,
                "message": f"model {{name}} loaded",
                "model_config": model_config,
            }
    except ImportError as e:
        logger.info(e)
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Unkwown model"
        )


@router.post(f"/unload")
async def unload_model():
    # log prediction call to file
    logging.getLogger("predictions").info("{{name}}|UNLOAD")
    name = "{{name}}"
    if not triton_client.is_model_ready(name):
        logger.info(f"No model with name {{name}} loaded")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    else:
        logger.info(f"Unloading model {{name}}")
        triton_client.unload_model(name)
        return {"success": True, "message": f"model {{name}} unloaded"}


@router.post(f"/predict")
async def predict(file: bytes = File(...)):
    # log prediction call to file
    logging.getLogger("predictions").info("{{name}}|PREDICT")

    name = "{{name}}"
    model = model_dict[name]

    if not triton_client.is_model_ready(name):
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Model not available"
        )

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

    if type(response[0]) == Image.Image:
        logger.warning("returning StreamingResponse")
        # return file response
        img_byte_arr = BytesIO()
        response[0].save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # important here!
        return StreamingResponse(img_byte_arr, media_type="image/png")
    else:
        return response
