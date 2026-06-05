"""
Step 3 -- FAISS IVF benchmark (cluster-based, disk-serializable index).

A controlled small-scale counterpart to the in-memory HNSW graph: an inverted
file (coarse k-means buckets) whose vectors live in a disk-resident index file
and whose recall/QPS is tuned by ``nprobe``. This is NOT DiskANN -- it is a
lightweight, dependency-light stand-in for the *class* of partition-based,
disk-friendly indexes, used here to exercise the harness end-to-end on a laptop.
For real DiskANN production numbers see ../data/benchmark_3.38M_summary.json
and ../docs/diskann_technical_analysis.md.

Usage:
    python step3_ivf_benchmark.py --data ../data --out ../data \\
        --nlist 50 --nprobe 2 4 8 12 16 24
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from bench_utils import (load_or_synthesize, measure_latencies,
                         percentile_stats, recall_at_k, rss_mb)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../data")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--nlist", type=int, default=50, help="#coarse centroids")
    p.add_argument("--nprobe", type=int, nargs="+",
                   default=[2, 4, 8, 12, 16, 24])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    import faiss

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "ivf.index")

    print("[1/4] Loading data ...")
    base, queries, gt = load_or_synthesize(
        args.data, args.n, args.dim, args.n_queries, 100, args.seed)
    dim = base.shape[1]
    print(f"  base={base.shape}  queries={queries.shape}")

    print(f"[2/4] Training + adding IVF (nlist={args.nlist}) ...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist,
                               faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.train(base)
    index.add(base)
    build_time = time.time() - t0
    faiss.write_index(index, index_path)          # persist to disk
    index_size_gb = os.path.getsize(index_path) / 1024 ** 3
    print(f"  build={build_time:.2f}s  index={index_size_gb * 1024:.1f} MB")
    del index, base

    print("[3/4] Reloading from disk and sweeping nprobe ...")
    mem_before_load = rss_mb()
    index = faiss.read_index(index_path)
    load_rss_delta = rss_mb() - mem_before_load

    by_param = {}
    for nprobe in args.nprobe:
        index.nprobe = nprobe

        results = []
        t0 = time.perf_counter()
        for q in queries:
            _, ids = index.search(q.reshape(1, -1), args.top_k)
            results.append(ids[0].tolist())
        qps = len(queries) / (time.perf_counter() - t0)

        lats = measure_latencies(
            lambda q: index.search(q.reshape(1, -1), args.top_k), queries)
        stats = percentile_stats(lats)
        recall = recall_at_k(results, gt, args.top_k)
        by_param[str(nprobe)] = {"qps": qps, "recall": recall, **stats}
        print(f"  nprobe={nprobe:>3}  recall@{args.top_k}={recall:.4f}  "
              f"qps={qps:8.1f}  p50={stats['p50']:.2f}ms  p99={stats['p99']:.2f}ms")
    search_rss = rss_mb()

    print("[4/4] Saving results ...")
    out = {
        "algo": "FAISS IVF (disk-resident, DiskANN-class stand-in)",
        "config": {"nlist": args.nlist, "metric": "inner_product",
                   "num_vectors": index.ntotal, "dim": dim},
        "build": {"time_s": build_time, "index_size_gb": index_size_gb},
        "memory": {"load_rss_delta_mb": load_rss_delta, "search_rss_mb": search_rss},
        "by_param": by_param,
    }
    path = os.path.join(args.out, "ivf_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
