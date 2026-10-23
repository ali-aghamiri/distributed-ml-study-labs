# Distributed Machine Learning Systems: independent study labs

Four executable study labs based on the course assignment sheets for Distributed Machine Learning Systems. The implementation here was written for this repository with AI coding assistance. It is a study portfolio, not a record of completed graded submissions or measured cluster results.

| Lab | Topic | Locally checked | Additional environment |
| --- | --- | --- | --- |
| [01](01-mpi-fedavg/README.md) | MPI scheduling, FedAvg logistic regression, BLAS | serial/FedAvg simulation/BLAS | MPI runtime and `mpi4py` for distributed runs |
| [02](02-cuda-ddp/README.md) | RGB to grayscale, CNN, DDP | CPU conversion and single-process training | CUDA/Numba and STL-10 for GPU comparison |
| [03](03-spark/README.md) | RDD text analysis, skip-gram, Spark ML K-means | local text and skip-gram logic | PySpark/Java and HDFS for distributed runs |
| [04](04-torchrun-accelerate/README.md) | `torchrun`, Accelerate, CPU profiling | CPU local training/profiler | multi-node cluster and Accelerate for remaining configurations |

Start with `python -m pip install -r requirements.txt`; install the optional packages described inside each lab only when that backend is needed. Original course datasets are excluded from Git because of size. Every command defaults to a small deterministic smoke run or accepts a path to the original dataset.

## Reproducibility

This environment has a CPU-only PyTorch build. GPU, MPI, Spark/HDFS and multi-node SLURM measurements have not been run here. No performance values from another person's work are presented as results. Each README distinguishes locally executed behavior from commands to run on the target infrastructure. The assignment sheets and datasets supplied by the course are separate from this source repository.

See [local validation](RESULTS.md) for the runs performed on the available CPU machine.
