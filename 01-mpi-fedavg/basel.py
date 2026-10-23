"""Basel-series π estimate with measurable static work partitioning."""
import argparse
import math
import time


def term(k: int) -> float:
    # Deliberately variable computation cost, as in the course problem.
    work = 0
    for i in range(k // 1000):
        work += i * i
    return 1.0 / (k * k)


def indexes(n: int, rank: int, size: int, strategy: str):
    if strategy == "cyclic":
        return range(rank + 1, n + 1, size)
    width, extra = divmod(n, size)
    start = rank * width + min(rank, extra) + 1
    stop = start + width + (rank < extra)
    return range(start, stop)


def local_sum(n, rank, size, strategy):
    started = time.perf_counter()
    value = math.fsum(term(k) for k in indexes(n, rank, size, strategy))
    return value, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strategy", choices=["contiguous", "cyclic"], default="cyclic")
    parser.add_argument("--backend", choices=["local", "mpi"], default="local")
    args = parser.parse_args()
    if args.terms < 1 or args.workers < 1:
        parser.error("terms and workers must be positive")
    started = time.perf_counter()
    if args.backend == "mpi":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        result = local_sum(args.terms, rank, size, args.strategy)
        gathered = comm.gather(result, root=0)
        if rank:
            return
    else:
        gathered = [local_sum(args.terms, r, args.workers, args.strategy)
                    for r in range(args.workers)]
    estimate = math.sqrt(6 * math.fsum(x[0] for x in gathered))
    print(f"pi={estimate:.10f} absolute_error={abs(estimate-math.pi):.6g} "
          f"elapsed={time.perf_counter()-started:.6f}s")
    print("worker_seconds=", [round(x[1], 6) for x in gathered])


if __name__ == "__main__":
    main()
