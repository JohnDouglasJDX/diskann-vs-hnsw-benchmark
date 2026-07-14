"""
HNSW baseline on Milvus 2.5.10 (standalone) for the 3.38M x 1024 production run.

Reproduces the HNSW row of ../../data/benchmark_3.38M_summary.json:
    M=16, efConstruction=200, metric=IP  ->  ef=32: Recall@10 0.927, QPS 52.4,
    11.7 GB resident in the Milvus container, 7.9 GB index on disk, 22.3 min build.

Vectors are L2-normalized, so inner product (IP) == cosine. Search memory is
read from the Milvus container's RSS (`docker stats`), not from this process.

Bring up Milvus 2.5.10 standalone first, e.g.:
    # https://milvus.io/docs/v2.5.x/install_standalone-docker.md
    docker compose up -d            # standalone + etcd + minio

This needs the proprietary corpus and is NOT part of CI. The reproducible
laptop path is ../step2_hnsw_benchmark.py.

Usage:
    python milvus_hnsw_3.38M.py --vectors ../../data/vectors.npy \\
        --queries ../../data/query_vectors.npy --gt ../../data/ground_truth.npy
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors", required=True, help="base vectors .npy (N x 1024)")
    p.add_argument("--queries", required=True, help="query vectors .npy")
    p.add_argument("--gt", required=True, help="exact top-k ground truth .npy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")
    p.add_argument("--collection", default="okw_papers")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, nargs="+",
                   default=[16, 32, 64, 100, 150, 200])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch", type=int, default=50000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--output", default="data/milvus_hnsw_raw.json",
                   help="machine-readable per-ef metrics and raw latencies")
    return p.parse_args()


def recall_at_k(results, gt, k: int) -> float:
    hits = sum(len(set(r[:k]) & set(g[:k])) for r, g in zip(results, gt))
    return hits / (len(gt) * k)


def main() -> None:
    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections, utility)

    args = parse_args()
    base = np.load(args.vectors).astype(np.float32)
    queries = np.load(args.queries).astype(np.float32)
    gt = np.load(args.gt)
    n, dim = base.shape
    print(f"base={base.shape}  queries={queries.shape}")

    connections.connect("default", host=args.host, port=args.port)
    if utility.has_collection(args.collection):
        utility.drop_collection(args.collection)

    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vec", DataType.FLOAT_VECTOR, dim=dim),
    ])
    col = Collection(args.collection, schema)

    print("[1/4] Inserting ...")
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        col.insert([list(range(s, e)), base[s:e].tolist()])
    col.flush()

    print(f"[2/4] Building HNSW (M={args.M}, efC={args.ef_construction}, IP) ...")
    t0 = time.time()
    col.create_index("vec", {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": args.M, "efConstruction": args.ef_construction},
    })
    utility.wait_for_index_building_complete(args.collection)
    print(f"  build={ (time.time() - t0) / 60:.1f} min")

    print("[3/4] Loading + sweeping ef_search ...")
    col.load()
    q = queries.tolist()
    by_ef = {}
    for ef in args.ef_search:
        params = {"metric_type": "IP", "params": {"ef": ef}}
        for query in q[:min(args.warmup, len(q))]:
            col.search([query], "vec", params, limit=args.top_k)

        labels, latencies_ms = [], []
        t0 = time.perf_counter()
        for query in q:
            query_t0 = time.perf_counter()
            res = col.search([query], "vec", params, limit=args.top_k)
            latencies_ms.append((time.perf_counter() - query_t0) * 1000.0)
            labels.append([hit.id for hit in res[0]])
        elapsed = time.perf_counter() - t0
        qps = len(q) / elapsed
        rec = recall_at_k(labels, gt, args.top_k)
        percentiles = np.percentile(latencies_ms, [50, 95, 99])
        row = {
            "recall_at_k": rec,
            "qps": qps,
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "p50_latency_ms": float(percentiles[0]),
            "p95_latency_ms": float(percentiles[1]),
            "p99_latency_ms": float(percentiles[2]),
            "raw_latency_ms": latencies_ms,
        }
        by_ef[str(ef)] = row
        print(f"  ef={ef:>4}  recall@{args.top_k}={rec:.4f}  "
              f"qps={qps:8.1f}  p50={percentiles[0]:.2f}ms  "
              f"p99={percentiles[2]:.2f}ms")

    output = {
        "engine": "Milvus 2.5.10 standalone",
        "protocol": "one query per client call, sequential, after warmup",
        "config": {"M": args.M, "efConstruction": args.ef_construction,
                   "top_k": args.top_k, "num_vectors": n, "dim": dim,
                   "num_queries": len(q), "warmup": args.warmup},
        "by_ef": by_ef,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  raw search results: {args.output}")

    print("[4/4] Read container RSS for search memory:  docker stats <milvus>")


if __name__ == "__main__":
    main()
