from typing import Optional

from fastapi import FastAPI

import tritonhttpclient
TRITONURL = 'host.docker.internal:8000'

app = FastAPI()

try:
    # Specify large enough concurrency to handle the
    # the number of requests.
    concurrency = 1
    triton_client = tritonhttpclient.InferenceServerClient(
                    url=TRITONURL, concurrency=concurrency)
    print(f'Server ready? {triton_client.is_server_ready()}')
except Exception as e:
    print("client creation failed: " + str(e))



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/load_model/")
def load_model(model: str, version: Optional[int] = 1):

    if model == 'yolov4':
        triton_client.load_model(model)
        return {"success": True, "message": f"model {model} loaded"}
    else:
        return {"success": False, "message": "unknown model"}