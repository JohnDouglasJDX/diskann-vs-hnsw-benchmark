"""
Step 2 -- HNSW (hnswlib) benchmark.

Builds an in-memory HNSW graph index and sweeps ef_search, recording build
time, on-disk index size, load/search RSS, and per-ef Recall@10 / QPS / latency
percentiles. Falls back to a synthetic dataset if step1 output is absent.

Usage:
    python step2_hnsw_benchmark.py --data ../data --out ../data \\
        --n 10000 --dim 1024 --M 16 --ef-construction 200 \\
        --ef-search 16 32 64 100 150 200
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from bench_utils import (environment_info, load_or_synthesize, measure_latencies,
                         percentile_stats, recall_at_k, rss_mb)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="../data", help="dir with step1 .npy files")
    p.add_argument("--out", default="../data")
    p.add_argument("--n", type=int, default=10000, help="synthetic fallback size")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, nargs="+",
                   default=[16, 32, 64, 100, 150, 200])
    p.add_argument("--space", default="cosine", choices=["cosine", "ip", "l2"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    import hnswlib

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "hnsw_index.bin")

    print("[1/4] Loading data ...")
    data_source = ("files" if os.path.exists(os.path.join(args.data, "vectors.npy"))
                   else "synthetic-disjoint-queries")
    base, queries, gt = load_or_synthesize(
        args.data, args.n, args.dim, args.n_queries, 100, args.seed)
    dim = base.shape[1]
    print(f"  base={base.shape}  queries={queries.shape}")

    print(f"[2/4] Building HNSW (M={args.M}, "
          f"ef_construction={args.ef_construction}) ...")
    mem0 = rss_mb()
    t0 = time.time()
    index = hnswlib.Index(space=args.space, dim=dim)
    index.init_index(max_elements=len(base),
                     ef_construction=args.ef_construction, M=args.M)
    index.add_items(base)
    build_time = time.time() - t0
    build_rss_delta = rss_mb() - mem0
    index.save_index(index_path)
    index_size_gb = os.path.getsize(index_path) / 1024 ** 3
    print(f"  build={build_time:.2f}s  index={index_size_gb * 1024:.1f} MB  "
          f"build_rss_delta={build_rss_delta:.0f} MB")
    del index

    print("[3/4] Reloading index and sweeping ef_search ...")
    mem_before_load = rss_mb()
    index = hnswlib.Index(space=args.space, dim=dim)
    index.load_index(index_path)
    load_rss_delta = rss_mb() - mem_before_load

    by_param = {}
    for ef in args.ef_search:
        index.set_ef(ef)

        # throughput (single-query path, matches latency path)
        results = []
        t0 = time.perf_counter()
        for q in queries:
            labels, _ = index.knn_query(q.reshape(1, -1), k=args.top_k)
            results.append(labels[0].tolist())
        qps = len(queries) / (time.perf_counter() - t0)

        lats = measure_latencies(
            lambda q: index.knn_query(q.reshape(1, -1), k=args.top_k), queries)
        stats = percentile_stats(lats)
        recall = recall_at_k(results, gt, args.top_k)
        by_param[str(ef)] = {"qps": qps, "recall": recall, **stats}
        print(f"  ef={ef:>4}  recall@{args.top_k}={recall:.4f}  "
              f"qps={qps:8.1f}  p50={stats['p50']:.2f}ms  p99={stats['p99']:.2f}ms")
    search_rss = rss_mb()

    print("[4/4] Saving results ...")
    out = {
          "algo": "HNSW (hnswlib)",
          "environment": environment_info(),
          "config": {"M": args.M, "ef_construction": args.ef_construction,
                     "space": args.space, "num_vectors": len(base), "dim": dim,
                     "num_queries": len(queries), "data_source": data_source,
                     "seed": args.seed},
        "build": {"time_s": build_time, "index_size_gb": index_size_gb,
                  "build_rss_delta_mb": build_rss_delta},
        "memory": {"load_rss_delta_mb": load_rss_delta, "search_rss_mb": search_rss},
        "by_param": by_param,
    }
    path = os.path.join(args.out, "hnsw_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
