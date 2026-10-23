"""Pooled logistic regression and sample-weighted FedAvg over three clients."""
import argparse
from pathlib import Path
import time

import numpy as np


def make_clients(data_dir, seed):
    rng = np.random.default_rng(seed)
    if data_dir:
        root = Path(data_dir)
        pairs = [(np.load(root / f"data{i}.npy"),
                  np.load(root / f"labels{i}.npy")) for i in (1, 2, 3)]
    else:
        # Three shifted distributions, so local models are not IID.
        true_w = rng.normal(size=12)
        pairs = []
        for bias in (-0.8, 0.0, 0.8):
            x = rng.normal(loc=bias, size=(300, 12))
            logits = x @ true_w / 4 + bias + rng.normal(scale=0.3, size=len(x))
            pairs.append((x, (logits > 0).astype(np.float64)))
    clients = []
    for x, y in pairs:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 5:
            raise ValueError("Each client needs 2D features and matching labels")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("Labels must be binary 0/1")
        order = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        clients.append(((x[order[:cut]], y[order[:cut]]),
                        (x[order[cut:]], y[order[cut:]])))
    if len({c[0][0].shape[1] for c in clients}) != 1:
        raise ValueError("Clients have different feature counts")
    return clients


def train(w, x, y, epochs, lr):
    w = w.copy()
    xb = np.column_stack((x, np.ones(len(x))))
    for _ in range(epochs):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -35, 35)))
        w -= lr * (xb.T @ (probabilities - y)) / len(y)
    return w


def accuracy(w, sample):
    x, y = sample
    predicted = (x @ w[:-1] + w[-1] >= 0).astype(np.float64)
    return float(np.mean(predicted == y))


def local_fedavg(clients, rounds, epochs, lr, bad_client):
    w = np.zeros(clients[0][0][0].shape[1] + 1)
    for _ in range(rounds):
        updates = []
        for i, (training, _) in enumerate(clients):
            x, y = training
            rate = 0.5 if i == bad_client else lr
            updates.append((len(y), train(w, x, y, epochs, rate)))
        w = sum(n * u for n, u in updates) / sum(n for n, _ in updates)
    return w


def mpi_fedavg(clients, rounds, epochs, lr, bad_client):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 4:
        raise ValueError("Run with exactly four MPI ranks: one server, three clients")
    chunks = [None] + clients if rank == 0 else None
    client = comm.scatter(chunks, root=0)
    dimension = clients[0][0][0].shape[1] if rank == 0 else None
    dimension = comm.bcast(dimension, root=0)
    w = np.zeros(dimension + 1)
    for _ in range(rounds):
        w = comm.bcast(w if rank == 0 else None, root=0)
        if rank:
            x, y = client[0]
            update = (len(y), train(w, x, y, epochs,
                                    0.5 if rank - 1 == bad_client else lr))
        else:
            update = None
        gathered = comm.gather(update, root=0)
        if rank == 0:
            updates = gathered[1:]
            w = sum(n * u for n, u in updates) / sum(n for n, _ in updates)
    w = comm.bcast(w if rank == 0 else None, root=0)
    score = accuracy(w, client[1]) if rank else None
    scores = comm.gather(score, root=0)
    return (w, scores[1:]) if rank == 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--backend", choices=["local", "mpi"], default="local")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--bad-client", type=int, choices=[0, 1, 2])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.lr <= 0:
        p.error("rounds, epochs and learning rate must be positive")
    if args.backend == "mpi":
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    clients = make_clients(args.data_dir, args.seed) if rank == 0 else None
    started = time.perf_counter()
    if args.backend == "mpi":
        result = mpi_fedavg(clients, args.rounds, args.local_epochs,
                            args.lr, args.bad_client)
        if rank:
            return
        fed_w, scores = result
    else:
        fed_w = local_fedavg(clients, args.rounds, args.local_epochs,
                             args.lr, args.bad_client)
        scores = [accuracy(fed_w, test) for _, test in clients]
    print(f"fedavg client_accuracy={np.round(scores, 4).tolist()} "
          f"mean={np.mean(scores):.4f} elapsed={time.perf_counter()-started:.3f}s")
    pooled_x = np.concatenate([part[0][0] for part in clients])
    pooled_y = np.concatenate([part[0][1] for part in clients])
    baseline = train(np.zeros(pooled_x.shape[1] + 1), pooled_x, pooled_y,
                     args.rounds * args.local_epochs, args.lr)
    print(f"pooled_baseline client_accuracy="
          f"{np.round([accuracy(baseline, test) for _, test in clients], 4).tolist()}")


if __name__ == "__main__":
    main()
