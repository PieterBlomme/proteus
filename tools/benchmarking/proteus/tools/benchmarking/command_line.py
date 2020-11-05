import importlib
import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

mod = importlib.import_module("proteus.datasets")


def load_model(model):
    response = requests.post("http://localhost/load", json.dumps({"name": model}))
    assert response.json()["success"]


def load_dataset(dataset, num_samples):
    dataset = getattr(mod, dataset)
    return dataset(k=num_samples)

def get_prediction(fpath, model):
    with open(fpath, "rb") as f:
        jsonfiles = {"file": f}
        payload = {"file_id": fpath}
        response = requests.post(
            f"http://localhost/{model}/predict",
            files=jsonfiles,
            data=payload,
        )
    return response


def main():
    print("Welcome to the benchmark tool")
    model = input("Model? ")
    load_model(model)

    dataset = input("Dataset? ")
    num_samples = input("Number of samples? ")
    num_samples = int(num_samples)
    dataset = load_dataset(dataset, num_samples)

    num_workers = input("Number of workers? ")
    num_workers = int(num_workers)

    start = time.time()
    preds = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start the load operations and mark each future with its URL
        futures = [
                executor.submit(get_prediction, fpath, model)
                for (fpath, img) in dataset
                ]
        for fut in as_completed(futures):
            response = fut.result()
            preds.append(response.json()[0])
    score = dataset.eval(preds)
    end = time.time()
    throughput = (end - start)*1000 / num_samples

    #latency
    fpath, img = dataset[0]
    latency = get_prediction(fpath, model).elapsed.total_seconds()*1000

    print('Results')
    print(f'Throughput: {throughput} FPS')
    print(f'Latency: {latency} ms')
    print(f'Score: {score}')