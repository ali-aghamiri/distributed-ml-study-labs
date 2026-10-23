"""Run matrix-operation timing at explicit, memory-conscious sizes."""
import argparse
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    a = p.parse_args()
    print("numpy", np.__version__)
    np.show_config()
    for size in a.sizes:
        if size < 1:
            p.error("sizes must be positive")
        rng = np.random.default_rng(19)
        matrix = rng.standard_normal((size, size)) + size * np.eye(size)
        for name, operation in (("determinant", np.linalg.det),
                                ("inverse", np.linalg.inv)):
            start = time.perf_counter()
            result = operation(matrix)
            elapsed = time.perf_counter() - start
            print(f"n={size} operation={name} seconds={elapsed:.6f} "
                  f"finite={bool(np.isfinite(result).all())}")


if __name__ == "__main__":
    main()
