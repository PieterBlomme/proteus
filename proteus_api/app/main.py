import importlib
import logging
import os
import pkgutil
from io import BytesIO

import proteus.models
import tritonclient.http as httpclient
from fastapi import FastAPI, File, HTTPException
from PIL import Image
from pydantic import BaseModel
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# global logging level
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if os.environ.get("DEBUG") == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

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
logger.info(model_dict)


class Model(BaseModel):
    name: str


app = FastAPI()

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


@app.get("/health")
async def get_server_health():
    if triton_client.is_server_live():
        logger.info("Server is alive")
        return {"success": True}
    else:
        logger.info(f"Server is dead")
        return {"success": False}


@app.get("/models")
async def get_models():
    return {k: v.DESCRIPTION for (k, v) in model_dict.items()}


@app.get("/models/status")
async def get_model_repository():
    return triton_client.get_model_repository_index()


@app.post("/load/")
async def load_model(model: Model):

    try:
        MODEL = model_dict[model.name]
        logger.info(f"Loading model {model.name}")
        MODEL.load_model(triton_client)

        if not triton_client.is_model_ready(model.name):
            return {
                "success": False,
                "message": f"model {model.name} not ready - check logs",
            }
        else:
            return {"success": True, "message": f"model {model.name} loaded"}
    except ImportError as e:
        logger.info(e)
        return {"success": False, "message": f"unknown model {model.name}"}


@app.post("/unload/")
async def unload_model(model: Model):
    if not triton_client.is_model_ready(model.name):
        logger.info(f"No model with name {model.name} loaded")
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f"Unloading model {model.name}")
        triton_client.unload_model(model.name)
        return {"success": True, "message": f"model {model.name} unloaded"}


@app.post("/{model}/predict")
async def predict(model: str, file: bytes = File(...)):
    if not triton_client.is_model_ready(model):
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
    MODEL = model_dict[model]
    response = MODEL.inference_http(triton_client, img)
    return response
