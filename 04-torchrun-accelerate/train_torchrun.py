"""Single-node or multi-node PyTorch DDP training via torchrun."""
import argparse
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common import datasets, evaluate, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--checkpoint")
    a = p.parse_args()
    if min(a.samples, a.epochs, a.batch_size) < 1:
        p.error("samples, epochs and batch size must be positive")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(55)
    training, test, classes = datasets(a.data_dir, a.samples)
    sampler = DistributedSampler(training) if world > 1 else None
    loader = DataLoader(training, batch_size=a.batch_size,
                        sampler=sampler, shuffle=sampler is None, drop_last=True)
    network = model(classes).to(device)
    if world > 1:
        network = nn.parallel.DistributedDataParallel(
            network, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    started = time.perf_counter()
    for epoch in range(a.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        network.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(network(features), labels).backward()
            optimizer.step()
    if world > 1:
        dist.barrier()
    if rank == 0:
        bare = network.module if world > 1 else network
        accuracy = evaluate(bare, DataLoader(test, batch_size=a.batch_size), device)
        if a.checkpoint:
            torch.save(bare.state_dict(), a.checkpoint)
        print(f"device={device.type} ranks={world} accuracy={accuracy:.4f} "
              f"train_seconds={time.perf_counter()-started:.3f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
