from fastapi import FastAPI, File, HTTPException
import tritonclient.http as httpclient

import logging
import importlib
from PIL import Image
from io import BytesIO
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

#factory
def get_inference_http(model):
    module = importlib.import_module(f"proteus.{model}")
    return module.inference_http


from pydantic import BaseModel
class Model(BaseModel):
    name: str

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

app = FastAPI()

# set up Triton connection
TRITONURL = 'triton:8000'

try:
    # Specify large enough concurrency to handle the
    # the number of requests.
    concurrency = 1
    triton_client = httpclient.InferenceServerClient(
                    url=TRITONURL, concurrency=concurrency)
    logger.info(f'Server ready? {triton_client.is_server_ready()}')
except Exception as e:
    logger.error("client creation failed: " + str(e))


@app.get("/health")
async def get_server_health():
    if triton_client.is_server_live():
        logger.info('Server is alive')
        return {"success": True}
    else:
        logger.info(f'Server is dead')
        return {"success": False}


@app.get("/models")
async def get_model_repository():
    return triton_client.get_model_repository_index()


@app.post("/load/")
async def load_model(model: Model):
    if model.name in ('yolov4', 'mobilenet'):
        logger.info(f'Loading model {model.name}')
        triton_client.load_model(model.name)
        if not triton_client.is_model_ready(model.name):
            return {"success": False,
                    "message": f"model {model.name} not ready - check logs"}
        else:
            return {"success": True, "message": f"model {model.name} loaded"}
    else:
        return {"success": False, "message": "unknown model"}


@app.post("/unload/")
async def unload_model(model: Model):
    if not triton_client.is_model_ready(model.name):
        logger.info(f'No model with name {model.name} loaded')
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f'Unloading model {model.name}')
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
    inference_http = get_inference_http(model)
    response = inference_http(triton_client, img)
    return response
