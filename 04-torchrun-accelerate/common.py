"""Shared data loading and model construction for assignment 4."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset


def datasets(data_dir=None, samples=128):
    if data_dir:
        root = Path(data_dir)
        arrays = [np.load(root / f"{split}_{suffix}.npy", mmap_mode="r")
                  for split in ("train", "test") for suffix in ("x", "y")]
        train_x, train_y, test_x, test_y = arrays
        def convert(x, y):
            return TensorDataset(torch.from_numpy(np.array(x[:samples], dtype=np.float32)),
                                 torch.from_numpy(np.array(y[:samples], dtype=np.int64)))
        training, test = convert(train_x, train_y), convert(test_x, test_y)
        classes = int(max(np.max(train_y), np.max(test_y))) + 1
    else:
        generator = torch.Generator().manual_seed(93)
        x = torch.randn(samples + max(32, samples // 4), 512, generator=generator)
        y = ((x[:, :10].argmax(dim=1) + (x[:, 10] > 0).long()) % 10).long()
        training = TensorDataset(x[:samples], y[:samples])
        test = TensorDataset(x[samples:], y[samples:])
        classes = 10
    return training, test, classes


def model(classes, activation="relu"):
    activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid,
                   "tanh": nn.Tanh, "gelu": nn.GELU}
    return nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128),
                         activations[activation](), nn.Linear(128, 64),
                         nn.BatchNorm1d(64), activations[activation](),
                         nn.Linear(64, classes))


def evaluate(network, loader, device):
    network.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            logits = network(features.to(device))
            correct += int((logits.argmax(1) == labels.to(device)).sum())
            total += len(labels)
    return correct / total
