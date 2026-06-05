#!/usr/bin/env bash
#
# DiskANN (microsoft/DiskANN, Rust) build + search for the 3.38M x 1024
# production run, via the `diskann-benchmark` runner (graph-index-build-pq).
# Reproduces the two operating points in
# ../../data/benchmark_3.38M_summary.json:
#
#   R=32 / L=64   -> Recall@10 0.933, QPS 173.0, < 1 GB search RAM
#   R=64 / L=128  -> Recall@10 0.976, QPS 109.4, < 1 GB search RAM
#
# Prereqs:
#   * microsoft/DiskANN (Rust workspace) cloned and built: `cargo build --release`.
#   * Base/query vectors converted to .bin via npy_to_diskann_bin.py.
#   * Vectors are L2-normalized, so squared_l2 ranking == cosine == inner product.
#
# NOT runnable in CI: needs the proprietary corpus, a built DiskANN, and a fast
# NVMe SSD. The reproducible path is scripts/step2..5 in the parent directory.
set -euo pipefail

DISKANN="${DISKANN_REPO:-$HOME/DiskANN}"   # clone of microsoft/DiskANN (Rust)
DATA="${DATA_DIR:-$PWD/data}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 0) vectors -> DiskANN .bin -------------------------------------------------
python "$HERE/npy_to_diskann_bin.py" "$DATA/vectors.npy"       "$DATA/base.bin"
python "$HERE/npy_to_diskann_bin.py" "$DATA/query_vectors.npy" "$DATA/query.bin"

cd "$DISKANN"

# 1) exact top-100 ground truth ---------------------------------------------
cargo run --release --package diskann-tools --bin compute_groundtruth -- \
    --data_type float --dist_fn l2 \
    --base_file  "$DATA/base.bin" \
    --query_file "$DATA/query.bin" \
    --gt_file    "$DATA/gt" \
    --recall_at  100                      # -> $DATA/gt.bin

# 2) (one-time) inspect the PQ build/search schema for this DiskANN revision -
cargo run --release --package diskann-benchmark -- inputs graph-index-build-pq

# 3) build (R=32/L=64 and R=64/L=128) + search sweep -------------------------
#    diskann_job.json holds both jobs; max_degree=R, l_build=L, runs[].search_l
#    is the recall/latency knob, recall_k=10.
cargo run --release --package diskann-benchmark -- run \
    --input-file  "$HERE/diskann_job.json" \
    --output-file "$DATA/diskann_out.json"

# diskann_out.json carries per-(R, search_l) Recall@10 / latency / QPS and the
# resident PQ+cache footprint (the reported "< 1 GB"). Those rows were copied
# into ../../data/benchmark_3.38M_summary.json.
