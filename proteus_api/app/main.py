from fastapi import FastAPI, File, HTTPException
import tritonclient.http as httpclient

import logging
from PIL import Image
from io import BytesIO
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from proteus.yolov4 import inference_http as inference_http_yolov4
from proteus.mobilenet import inference_http as inference_http_mobilenet

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
async def load_model(model: str):
    if model in ('yolov4', 'mobilenet'):
        logger.info(f'Loading model {model}')
        triton_client.load_model(model)
        if not triton_client.is_model_ready(model):
            return {"success": False,
                    "message": f"model {model} not ready - check logs"}
        else:
            return {"success": True, "message": f"model {model} loaded"}
    else:
        return {"success": False, "message": "unknown model"}


@app.post("/unload/")
async def unload_model(model: str):
    if not triton_client.is_model_ready(model):
        logger.info(f'No model with name {model} loaded')
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f'Unloading model {model}')
        triton_client.unload_model(model)
        return {"success": True, "message": f"model {model} unloaded"}


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
    if model == 'yolov4':
        response = inference_http_yolov4(triton_client, img)
    else:
        response = inference_http_mobilenet(triton_client, img)
    return response
