import importlib
import logging
import os
import pkgutil
from io import BytesIO

import proteus.models
import tritonclient.http as httpclient
from fastapi import APIRouter, FastAPI, File, HTTPException
from PIL import Image
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

app = FastAPI()

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

# global logging level
logging.basicConfig(level=logging.INFO)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if os.environ.get("DEBUG") == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.info(model_dict)

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


# build model-specific routers
for name, model in model_dict.items():
    router = APIRouter()

    @router.post("/load")
    async def load_model():

        try:
            logger.info(f"Loading model {name}")
            model.load_model(triton_client)

            if not triton_client.is_model_ready(name):
                return {
                    "success": False,
                    "message": f"model {name} not ready - check logs",
                }
            else:
                return {"success": True, "message": f"model {name} loaded"}
        except ImportError as e:
            logger.info(e)
            return {"success": False, "message": f"unknown model {name}"}

    @router.post("/unload")
    async def unload_model():
        if not triton_client.is_model_ready(name):
            logger.info(f"No model with name {name} loaded")
            return {"success": False, "message": "model not loaded"}
        else:
            logger.info(f"Unloading model {name}")
            triton_client.unload_model(name)
            return {"success": True, "message": f"model {name} unloaded"}

    @router.post("/predict")
    async def predict(file: bytes = File(...)):
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

    app.include_router(
        router,
        prefix=f"/{name}",
        tags=[f"{name}"],
    )
