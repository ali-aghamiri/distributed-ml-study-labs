"""Correctness checks for partitioning, aggregation and data transforms."""
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def script(path):
    return runpy.run_path(str(ROOT / path))


def test_basel_partitions_cover_every_term_once():
    indexes = script("01-mpi-fedavg/basel.py")["indexes"]
    for strategy in ("contiguous", "cyclic"):
        for n, workers in ((1, 4), (19, 4), (20, 4), (23, 7)):
            assigned = [k for rank in range(workers)
                        for k in indexes(n, rank, workers, strategy)]
            assert sorted(assigned) == list(range(1, n + 1))


def test_one_epoch_weighted_fedavg_matches_pooled_gradient():
    module = script("01-mpi-fedavg/fedavg.py")
    clients = module["make_clients"](None, 42)
    aggregated = module["local_fedavg"](clients, rounds=1, epochs=1,
                                        lr=0.01, bad_client=None)
    x = np.concatenate([client[0][0] for client in clients])
    y = np.concatenate([client[0][1] for client in clients])
    pooled = module["train"](np.zeros(x.shape[1] + 1), x, y, 1, 0.01)
    np.testing.assert_allclose(aggregated, pooled, rtol=1e-12, atol=1e-12)


def test_grayscale_weights_and_shape():
    convert = script("02-cuda-ddp/grayscale.py")["grayscale_python"]
    images = np.array([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]]], dtype=np.float32)
    np.testing.assert_allclose(convert(images),
                               np.array([[[0.2989, 0.5870, 0.1140]]]),
                               atol=1e-6)


def test_shahname_verse_parser_rejects_headings():
    module = script("03-spark/text_analysis.py")
    assert module["parse_verse"]("## opening") is None
    tokens, rhyme = module["parse_verse"]("به نام خداوند جان و خرد\tکزین برتر اندیشه برنگذرد")
    assert rhyme == "برنگذرد"
    assert module["trigrams"](tokens)[0] == ("به", "نام", "خداوند")
