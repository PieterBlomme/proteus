import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

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


def main():
    print("Welcome to the benchmark tool")
    model = input("Model? ")
    
    model_config = {}
    triton_optimization = input("Triton optimization? (y/n)")
    if triton_optimization == 'y':
        model_config['triton_optimization'] = True
    else:
        model_config['triton_optimization'] = False
    load_model(model, model_config)

    dataset = input("Dataset? ")
    num_samples = input("Number of samples? ")
    num_samples = int(num_samples)
    dataset = load_dataset(dataset, num_samples)

    num_workers = input("Number of workers? ")
    num_workers = int(num_workers)

    start = time.time()
    preds = [None for i in range(num_samples)]
    ds = [s for s in dataset]  # pre-download
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
    throughput = num_samples / (end - start)
    # latency
    fpath, img = dataset[0]
    latency, _ = get_prediction(fpath, model, 0)
    latency = latency.elapsed.total_seconds() * 1000

    unload_model(model)

    print("Results")
    print(f"Throughput: {throughput} FPS")
    print(f"Latency: {latency} ms")
    print(f"Score: {score}")
