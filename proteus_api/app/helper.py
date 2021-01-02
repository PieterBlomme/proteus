import datetime
import importlib
import logging
import os
import pkgutil
from pathlib import Path

import proteus.models
import tritonclient.http as httpclient
from file_read_backwards import FileReadBackwards
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
    logger.debug(model_dict)
    return model_dict


def generate_endpoints(model):
    targetfile = Path(f"{currdir}/routers/{model}.py")
    if not targetfile.is_file():
        # file does not exist yet
        with open(targetfile, "w") as fh:
            fh.write(template.render(name=model))


def check_last_active(model):
    with FileReadBackwards("/logs/predictions.log") as frb:

        # getting lines by lines starting from the last line up
        for l in frb:
            ts, name, action = l.split("|")[0], l.split("|")[1], l.split("|")[2]
            if model == name and action == "LOADING":
                # Never trigger unload if still loading ...
                return 0
            elif model == name:
                last_call = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
                elapsed = datetime.datetime.now() - last_call
                return elapsed.total_seconds() / 60
    return 60 * 60 * 24  # some very large number, eg. 1 day
