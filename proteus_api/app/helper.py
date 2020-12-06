import importlib
import logging
import os
import pkgutil
from pathlib import Path

import proteus.models
import tritonclient.http as httpclient
from jinja2 import Environment, FileSystemLoader

currdir = os.path.dirname(os.path.abspath(__file__))

env = Environment(
    loader=FileSystemLoader([f"{currdir}/routers/templates"]),
)
template = env.get_template("template.py")

logger = logging.getLogger(__name__)


def get_triton_client():
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
    return triton_client


def get_model_dict():
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
    return model_dict


def generate_endpoints(model):
    targetfile = Path(f"{currdir}/routers/{model}.py")
    if not targetfile.is_file():
        # file does not exist yet
        with open(targetfile, "w") as fh:
            fh.write(template.render(name=model))