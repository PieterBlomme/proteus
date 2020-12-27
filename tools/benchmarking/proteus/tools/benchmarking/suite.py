import importlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

folder_path = Path(__file__).parent

mod = importlib.import_module("proteus.datasets")


def load_model(model, model_config):
    response = requests.post(
        f"http://localhost/{model}/load",
        json=model_config,
    )
    assert response.json()["success"]


def unload_model(model):
    response = requests.post(f"http://localhost/{model}/unload")
    assert response.json()["success"]


def load_dataset(dataset, num_samples):
    dataset = getattr(mod, dataset)
    return dataset(k=num_samples)


def get_prediction(fpath, model, i):
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    return response, i


def calculate_throughput(model, dataset, parms):
    num_samples = len(dataset)
    preds = [None for i in range(num_samples)]
    ds = [s for s in dataset]  # pre-download

    model_config = {
        k: v
        for k, v in parms.items()
        if k in ["quantize", "triton_optimization", "num_instances"]
    }
    load_model(model, model_config)

    start = time.time()
    with ThreadPoolExecutor(max_workers=parms["num_workers"]) as executor:
        # Start the load operations and mark each future with its index
        futures = [
            executor.submit(get_prediction, fpath, model, i)
            for i, (fpath, img) in enumerate(ds)
        ]

        for fut in as_completed(futures):
            response, i = fut.result()
            preds[i] = response.json()[0]
    end = time.time()
    throughput = num_samples / (end - start)
    print(parms)
    print(throughput)
    unload_model(model)


def calculate_latency(model, dataset, parms):
    num_samples = len(dataset)
    preds = [None for i in range(num_samples)]
    ds = [s for s in dataset][:num_samples]  # pre-download

    model_config = {
        k: v
        for k, v in parms.items()
        if k in ["quantize", "triton_optimization", "num_instances"]
    }
    load_model(model, model_config)

    latencies = []
    for i, (fpath, img) in enumerate(ds):
        latency, _ = get_prediction(fpath, model, i)
        latency = latency.elapsed.total_seconds() * 1000
        latencies.append(latency)
    latency = sum(latencies) / (num_samples)
    print(parms)
    print(latency)
    unload_model(model)

def calculate_score(model, dataset, parms):
    num_samples = len(dataset)
    preds = [None for i in range(num_samples)]
    ds = [s for s in dataset]  # pre-download

    model_config = {
        k: v
        for k, v in parms.items()
        if k in ["quantize", "triton_optimization", "num_instances"]
    }
    load_model(model, model_config)

    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Start the load operations and mark each future with its index
        futures = [
            executor.submit(get_prediction, fpath, model, i)
            for i, (fpath, img) in enumerate(ds)
        ]

        for fut in as_completed(futures):
            response, i = fut.result()
            preds[i] = response.json()[0]
    score = dataset.eval(preds)
    end = time.time()
    print(parms)
    print(score)
    unload_model(model)

def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Missing model argument")
        return

    model = args[0]
    with open(f"{folder_path}/configs/{args[0]}.json", "rb") as f:
        data = json.load(f)

    dataset = load_dataset(data["Dataset"], 10)
    for parms in data["Latency"]:
        calculate_latency(model, dataset, parms)

    dataset = load_dataset(data["Dataset"], 5)
    for parms in data["Throughput"]:
        calculate_throughput(model, dataset, parms)

    dataset = load_dataset(data["Dataset"], 20)
    for parms in data["Score"]:
        calculate_score(model, dataset, parms)