"""A portable single-process/DDP CNN training experiment."""
import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler


def build_dataset(kind, root, samples):
    if kind == "stl10":
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        training = datasets.STL10(root, split="train", transform=transform)
        test = datasets.STL10(root, split="test", transform=transform)
        return Subset(training, range(min(samples, len(training)))), test
    rng = torch.Generator().manual_seed(31)
    x = torch.randn(samples + max(16, samples // 4), 3, 32, 32, generator=rng)
    y = x.mean(dim=(2, 3)).argmax(dim=1)
    train = TensorDataset(x[:samples], y[:samples])
    test = TensorDataset(x[samples:], y[samples:])
    return train, test


def model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "stl10"], default="synthetic")
    p.add_argument("--data-root", default="data")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if min(args.samples, args.epochs, args.batch_size) < 1:
        p.error("samples, epochs and batch size must be positive")
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(41)
    train_set, test_set = build_dataset(args.dataset, args.data_root, args.samples)
    sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, shuffle=sampler is None)
    network = model().to(device)
    if distributed:
        network = nn.parallel.DistributedDataParallel(
            network, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        network.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(network(features), labels)
            loss.backward()
            optimizer.step()
    if distributed:
        dist.barrier()
    elapsed = time.perf_counter() - start
    if rank == 0:
        bare = network.module if distributed else network
        bare.eval()
        correct = total = 0
        with torch.no_grad():
            for features, labels in DataLoader(test_set, batch_size=args.batch_size):
                prediction = bare(features.to(device)).argmax(1).cpu()
                correct += int((prediction == labels).sum())
                total += len(labels)
        memory = (torch.cuda.max_memory_allocated(device) / 2**20
                  if device.type == "cuda" else None)
        print(f"backend={device.type} ranks={world} dataset={args.dataset} "
              f"accuracy={correct/total:.4f} train_seconds={elapsed:.3f} "
              f"peak_gpu_mib={memory}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
