import importlib.util
import logging
import os

from fastapi import FastAPI

from .helper import generate_endpoints, get_model_dict, get_triton_client

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
    generate_endpoints(name)
    currdir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        f"routers.{name}", f"{currdir}/routers/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    router = module.router
    app.include_router(
        router,
        prefix=f"/{name}",
        tags=[f"{name}"],
    )
