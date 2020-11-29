import logging
import os
from io import BytesIO

from fastapi import APIRouter, FastAPI, File, HTTPException
from PIL import Image
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from .helper import get_model_dict, get_triton_client

app = FastAPI()

# global logging level
logging.basicConfig(level=logging.INFO)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if os.environ.get("DEBUG") == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

triton_client = get_triton_client()
model_dict = get_model_dict()
logger.info(model_dict)


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
