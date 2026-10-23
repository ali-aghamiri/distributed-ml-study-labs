# Lab 01 — MPI, FedAvg and BLAS

The Basel-series experiment compares contiguous and cyclic work assignment for a term whose artificial cost grows with its index. The logistic-regression experiment compares pooled gradient descent with sample-weighted FedAvg over three non-IID clients. The BLAS script times determinant and inverse operations with explicit matrix sizes.

```powershell
python 01-mpi-fedavg/basel.py --terms 3000 --workers 4
python 01-mpi-fedavg/fedavg.py --rounds 10 --local-epochs 1
python 01-mpi-fedavg/blas_benchmark.py --sizes 64 128
```

The first two commands run local simulation by default. For actual MPI, install an MPI implementation and `mpi4py`, then use `mpiexec -n 4 python 01-mpi-fedavg/basel.py --backend mpi --strategy cyclic --terms 500000` and `mpiexec -n 4 python 01-mpi-fedavg/fedavg.py --backend mpi --data-dir DATA --rounds 10 --local-epochs 1`. `DATA` must hold the course's `data1.npy`–`data3.npy` and `labels1.npy`–`labels3.npy`. The simulation uses synthetic non-IID data only when `--data-dir` is omitted. Run `--rounds 1 --local-epochs 10` for the other scenario. The BLAS benchmark reports the active NumPy build; compare libraries only in distinct environments. Very large inverses need substantial memory, so sizes are explicit and small by default.
