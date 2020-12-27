import importlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from jinja2 import Template
import pandas as pd
import requests

folder_path = Path(__file__).parent
TEMPLATE_PATH = f'{folder_path}/templates/Benchmark.md'

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
    unload_model(model)
    parms['throughput'] = throughput
    return parms

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
    unload_model(model)
    parms['latency'] = latency
    return parms

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
    unload_model(model)
    parms['score'] = score
    return parms


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Missing path to config json")
        return

    with open(args[0], "rb") as f:
        data = json.load(f)

    model = data["Model"]

    num_samples_latency = 10
    dataset = load_dataset(data["Dataset"], num_samples_latency)
    results = []
    for parms in data["Latency"]:
        result = calculate_latency(model, dataset, parms)
        results.append(result)
    score_df = pd.DataFrame(results).sort_values(by='latency', ascending=False)
    print(score_df.to_markdown())

    num_samples_throughput = 5
    dataset = load_dataset(data["Dataset"], num_samples_throughput)
    results = []
    for parms in data["Throughput"]:
        result = calculate_throughput(model, dataset, parms)
        results.append(result)
    throughput_df = pd.DataFrame(results).sort_values(by='throughput', ascending=False)
    print(throughput_df.to_markdown())

    num_samples_score = 20
    dataset = load_dataset(data["Dataset"], num_samples_score)
    results = []
    for parms in data["Score"]:
        result = calculate_score(model, dataset, parms)
        results.append(result)
    latency_df = pd.DataFrame(results).sort_values(by='score', ascending=False)
    print(latency_df.to_markdown())

    with open(TEMPLATE_PATH) as f:
        template = Template(f.read())
    
    targetfile = 'Benchmark.md'
    with open(targetfile, "w") as fh:
        rendered_template = template.render(
            score_table=score_df.to_markdown(),
            latency_table=latency_df.to_markdown(),
            throughput_table=throughput_df.to_markdown(),
            dataset=data["Dataset"],
            num_samples_score=num_samples_score,
            num_samples_latency=num_samples_latency,
            num_samples_throughput=num_samples_throughput            
        )
        fh.write(rendered_template)