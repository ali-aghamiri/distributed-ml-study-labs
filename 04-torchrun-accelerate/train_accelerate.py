"""The same classifier trained with Hugging Face Accelerate."""
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import datasets, evaluate, model


def main():
    from accelerate import Accelerator
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    if min(a.samples, a.epochs, a.batch_size) < 1:
        p.error("samples, epochs and batch size must be positive")
    accelerator = Accelerator(mixed_precision=a.mixed_precision)
    torch.manual_seed(55)
    training, test, classes = datasets(a.data_dir, a.samples)
    loader = DataLoader(training, batch_size=a.batch_size, shuffle=True, drop_last=True)
    network = model(classes)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    network, optimizer, loader = accelerator.prepare(network, optimizer, loader)
    started = time.perf_counter()
    for _ in range(a.epochs):
        network.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss = nn.functional.cross_entropy(network(features), labels)
            accelerator.backward(loss)
            optimizer.step()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bare = accelerator.unwrap_model(network)
        accuracy = evaluate(bare, DataLoader(test, batch_size=a.batch_size),
                            accelerator.device)
        if a.checkpoint:
            accelerator.save(bare.state_dict(), a.checkpoint)
        print(f"device={accelerator.device} processes={accelerator.num_processes} "
              f"precision={a.mixed_precision} accuracy={accuracy:.4f} "
              f"train_seconds={time.perf_counter()-started:.3f}")


if __name__ == "__main__":
    main()
