"""Profile CPU inference and four activation choices."""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from common import datasets, model


def run_profile(network, loader, batches):
    network.eval()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU],
                                  profile_memory=True, record_shapes=True) as prof:
        for index, (features, _) in enumerate(loader):
            if index >= batches:
                break
            network(features)
    events = prof.key_averages()
    interesting = ("linear", "batch_norm", "relu", "sigmoid", "tanh", "gelu")
    return [(event.key, round(event.self_cpu_time_total / 1000, 3),
             event.self_cpu_memory_usage)
            for event in events if any(name in event.key.lower() for name in interesting)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    if min(a.samples, a.batches, a.batch_size) < 1:
        p.error("samples, batches and batch size must be positive")
    _, test, classes = datasets(a.data_dir, a.samples)
    loader = DataLoader(test, batch_size=a.batch_size)
    for activation in ("relu", "sigmoid", "tanh", "gelu"):
        print(activation, run_profile(model(classes, activation), loader, a.batches))


if __name__ == "__main__":
    main()
