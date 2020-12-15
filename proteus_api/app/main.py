import importlib.util
import logging
import os

from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from fastapi_utils.timing import add_timing_middleware
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from .helper import (
    check_last_active,
    generate_endpoints,
    get_model_dict,
    get_triton_client,
)

# Env vars
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG")
MODEL_INACTIVITY = int(os.environ.get("MODEL_INACTIVITY", "10"))

# Setup logging
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG")
logging.config.fileConfig(
    "/app/logging.conf", disable_existing_loggers=False, defaults={"level": LOGLEVEL}
)
logger = logging.getLogger(__name__)

# Setup FastAPI
app = FastAPI()
add_timing_middleware(app, record=logger.info)

triton_client = get_triton_client()
model_dict = get_model_dict()


@app.on_event("startup")
@repeat_every(seconds=10)
async def remove_expired_models():
    # Get loaded models
    loaded_models = [
        m.get("name")
        for m in triton_client.get_model_repository_index()
        if m.get("state", "UNAVAILABLE") == "READY"
    ]
    for model in loaded_models:
        last_active = check_last_active(model)
        if last_active > MODEL_INACTIVITY:
            logger.warning(
                f"Model was last active {last_active} minutes ago.  Automatic shutdown because larger than {MODEL_INACTIVITY} MODEL_INACTIVITY"
            )
            triton_client.unload_model(model)


@app.get("/health")
async def get_server_health():
    if triton_client.is_server_live():
        logger.info("Server is alive")
        return {"success": True}
    else:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server not available",
        )


@app.get("/models")
async def get_models():
    return {k: v.DESCRIPTION for (k, v) in model_dict.items()}


@app.get("/models/status")
async def get_model_repository():
    try:
        return triton_client.get_model_repository_index()
    except:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server not available",
        )


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
