from typing import Optional

from fastapi import FastAPI

import logging

app = FastAPI()
logger = logging.getLogger("api")

import tritonclient.http as httpclient
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/load_model/")
async def load_model(model: str):

    if model == 'yolov4':
        logger.error('before load')
        triton_client.load_model(model)
        logger.error('after load')
        return {"success": True, "message": f"model {model} loaded"}
    else:
        return {"success": False, "message": "unknown model"}