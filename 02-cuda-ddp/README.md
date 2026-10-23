# Lab 02 — CUDA image conversion and DDP classifier

`grayscale.py` implements the stated RGB weighting with a Python pixel loop and an optional Numba CUDA kernel. It checks maximum pixel error and times both paths. The local smoke run uses synthetic 32×32 RGB images; use STL-10 images for the actual assignment comparison.

`cnn_ddp.py` trains a small convolutional classifier using `DistributedDataParallel` under `torchrun`; it uses a deterministic synthetic dataset for smoke tests and can load STL-10 with `torchvision` when available.

```powershell
python 02-cuda-ddp/grayscale.py --images 20
python 02-cuda-ddp/cnn_ddp.py --epochs 1 --samples 64
torchrun --standalone --nproc-per-node=2 02-cuda-ddp/cnn_ddp.py --epochs 1 --samples 64
```

On an NVIDIA GPU, install Numba with CUDA support and run `grayscale.py --backend cuda --images 200 --input-npy PATH` after preparing an N×H×W×3 NumPy array from STL-10 images. Run the classifier with `--dataset stl10 --data-root PATH --batch-size 32` on the target GPU, and use `torchrun --nproc-per-node=2` for two GPUs. CPU/Gloo is a functional check, not a GPU speed or NCCL result. STL-10 training accuracy above the course threshold and GPU memory comparisons require an actual GPU run and tuning.
