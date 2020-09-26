from typing import Optional
from fastapi import FastAPI
import tritonclient.http as httpclient
import logging
#TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

app = FastAPI()

#set up Triton connection
TRITONURL = 'host.docker.internal:8000'

try:
    # Specify large enough concurrency to handle the
    # the number of requests.
    concurrency = 1
    triton_client = httpclient.InferenceServerClient(
                    url=TRITONURL, concurrency=concurrency, verbose=True)
    logger.info(f'Server ready? {triton_client.is_server_ready()}')
except Exception as e:
    logger.error("client creation failed: " + str(e))


@app.get("/health")
async def get_server_health():
    if triton_client.is_server_live():
        logger.info(f'Server is alive')
        return {"success": True}
    else:
        logger.info(f'Server is dead')
        return {"success": False}

@app.get("/models")
async def get_model_repository():
    return triton_client.get_model_repository_index()

@app.get("/load/")
async def load_model(model: str):
    if model == 'yolov4':
        logger.info(f'Loading model {model}')
        triton_client.load_model(model)
        if not triton_client.is_model_ready(model):
            return {"success": False, "message": f"model {model} not ready - check logs"}
        else:
            return {"success": True, "message": f"model {model} loaded"}
    else:
        return {"success": False, "message": "unknown model"}

@app.get("/unload/")
async def unload_model(model: str):
    if not triton_client.is_model_ready(model):
        logger.info(f'No model with name {model} loaded')
        return {"success": False, "message": "model not loaded"}
    else:
        logger.info(f'Unloading model {model}')
        triton_client.unload_model(model)
        return {"success": True, "message": f"model {model} unloaded"}