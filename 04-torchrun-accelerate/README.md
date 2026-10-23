# Lab 04 — distributed training and CPU profiling

The dataset in the course archive has 512-dimensional `float16` features and integer labels. `common.py` converts features to `float32` and builds a feed-forward model with BatchNorm and ReLU. The same model is trained through `torchrun`/DDP or Hugging Face Accelerate. The profiler compares CPU operators and activation functions.

```powershell
python 04-torchrun-accelerate/train_torchrun.py --epochs 1 --samples 128
torchrun --standalone --nproc-per-node=2 04-torchrun-accelerate/train_torchrun.py --epochs 1 --samples 128
python 04-torchrun-accelerate/profile_cpu.py --batches 5
```

For the course data, extract the two supplied ZIP archives so the directory contains `train_x.npy`, `train_y.npy`, `test_x.npy`, and `test_y.npy`, then use `--data-dir PATH`. The training scripts accept `--checkpoint PATH`; rank zero alone writes it. Install `accelerate` to run `accelerate launch --cpu --num_processes 2 04-torchrun-accelerate/train_accelerate.py --data-dir PATH`. Multi-node SLURM configurations require a cluster and compatible launch parameters; no cluster result is claimed here. `fp16` mixed precision requires appropriate accelerator hardware; CPU runs use full precision.
