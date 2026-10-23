# Local validation

Tests were run on a Windows CPU machine with Python 3.11, NumPy 1.23.5 and PyTorch 2.5.1+cpu. These results establish that the local code paths execute; they are not distributed-system benchmarks.

| Check | Data | Observed result |
| --- | --- | --- |
| Basel series | 3,000 terms, four simulated workers | π ≈ 3.14127438; absolute error ≈ 0.00031827 |
| FedAvg | Course's three 2025 client arrays, 10 rounds × 1 epoch | Mean held-out client accuracy ≈ 0.6368 |
| BLAS | 32×32 and 64×64 matrices | Determinant and inverse returned finite values |
| Grayscale | Five synthetic RGB images | Pixel-wise and vectorized values differed by at most 1e-7 |
| CNN | Synthetic images, 64 training examples | Single-process CPU path completed |
| Text RDD logic | Course's Shahnameh text, local mode | 49,609 verse lines; 17,770 unique words |
| Skip-gram | Course's Shahnameh text, local mode | Embeddings produced for five requested characters; similarity/PCA plots saved locally |
| K-means | Course's 200 customer records, local mode | Five clusters produced |
| `torchrun` model | Course's 60,000 training and 10,000 test records; 5 epochs | Single-process CPU test accuracy 0.8476 |
| CPU profiler | Course data subset | Operator timing/memory output produced for four activations |
| Unit tests | Synthetic fixtures | Four tests passed |

The installed Windows PyTorch build fails at `torchrun` rendezvous because it lacks libuv support, before the training code starts. This is an environment limitation, so two-process CPU/Gloo behavior remains unverified here. MPI, Spark, CUDA/NCCL, Accelerate, HDFS and multi-node SLURM are likewise awaiting appropriate runtime access. The course's full cluster configuration and timing comparisons are not claimed.
