"""
Convert a NumPy ``.npy`` matrix into DiskANN's ``.bin`` binary format.

The microsoft/DiskANN Rust tools (``diskann-tools`` / ``diskann-benchmark``)
read a simple little-endian layout:

    [ npts : uint32 ][ dim : uint32 ][ npts * dim float32, row-major ]

The 3.38M base vectors and the 1,000 query vectors were converted with this
script before building the disk index. (The laptop harness in ``../`` uses
``.npy`` directly and does not need this.)

Usage:
    python npy_to_diskann_bin.py vectors.npy base.bin
    python npy_to_diskann_bin.py query_vectors.npy query.bin
"""
from __future__ import annotations

import sys

import numpy as np


def to_bin(npy_path: str, out_path: str) -> None:
    x = np.load(npy_path).astype(np.float32)
    if x.ndim != 2:
        raise SystemExit(f"expected a 2-D matrix, got shape {x.shape}")
    n, d = x.shape
    with open(out_path, "wb") as f:
        np.array([n, d], dtype=np.uint32).tofile(f)
        x.tofile(f)
    print(f"wrote {out_path}: {n} x {d} float32 (DiskANN .bin)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    to_bin(sys.argv[1], sys.argv[2])
