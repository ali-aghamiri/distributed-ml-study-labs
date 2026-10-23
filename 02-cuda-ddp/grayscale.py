"""RGB->gray reference loop and Numba CUDA implementation."""
import argparse
import time

import numpy as np


def grayscale_python(images):
    output = np.empty(images.shape[:3], dtype=np.float32)
    for n in range(images.shape[0]):
        for row in range(images.shape[1]):
            for col in range(images.shape[2]):
                r, g, b = images[n, row, col]
                output[n, row, col] = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return output


def grayscale_cuda(images):
    from numba import cuda

    @cuda.jit
    def kernel(rgb, gray):
        index = cuda.grid(1)
        total = gray.size
        if index < total:
            width = gray.shape[2]
            height = gray.shape[1]
            n = index // (height * width)
            row = (index // width) % height
            col = index % width
            gray[n, row, col] = (0.2989 * rgb[n, row, col, 0]
                                 + 0.5870 * rgb[n, row, col, 1]
                                 + 0.1140 * rgb[n, row, col, 2])

    if not cuda.is_available():
        raise RuntimeError("Numba CUDA backend needs an NVIDIA GPU and driver")
    device_input = cuda.to_device(images)
    device_output = cuda.device_array(images.shape[:3], dtype=np.float32)
    kernel[(device_output.size + 255) // 256, 256](device_input, device_output)
    cuda.synchronize()
    return device_output.copy_to_host()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--backend", choices=["python", "cuda"], default="python")
    p.add_argument("--side", type=int, default=32)
    p.add_argument("--input-npy", help="RGB images as an N,H,W,3 NumPy array")
    args = p.parse_args()
    if args.images < 1 or args.side < 1:
        p.error("images and side must be positive")
    if args.input_npy:
        images = np.asarray(np.load(args.input_npy)[:args.images], dtype=np.float32)
        if images.ndim != 4 or images.shape[-1] != 3:
            p.error("input array must have shape N,H,W,3")
        if len(images) == 0:
            p.error("input array is empty")
    else:
        images = np.random.default_rng(17).random(
            (args.images, args.side, args.side, 3), dtype=np.float32)
    start = time.perf_counter()
    reference = grayscale_python(images)
    print(f"python_seconds={time.perf_counter()-start:.6f}")
    if args.backend == "cuda":
        # Warm up JIT compilation before the timed call.
        grayscale_cuda(images[:1])
        start = time.perf_counter()
        actual = grayscale_cuda(images)
        print(f"cuda_seconds={time.perf_counter()-start:.6f} "
              f"max_abs_error={np.max(np.abs(reference-actual)):.7f}")
    else:
        expected = np.einsum("nhwc,c->nhw", images,
                             np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
        print(f"max_abs_error={np.max(np.abs(reference-expected)):.7f}")


if __name__ == "__main__":
    main()
