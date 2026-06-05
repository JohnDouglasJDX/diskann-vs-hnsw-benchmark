"""
Step 5 -- Concurrent throughput vs thread count.

The single-query QPS reported by step2/step3 is a *latency-bound* number (one
query at a time). A serving system instead cares about throughput under
concurrency. This script fixes one operating point per index and measures how
batched QPS scales as the engine is given more threads
(hnswlib ``num_threads`` / faiss OpenMP), turning the "single-threaded only"
caveat into a measured scaling curve.

Recall is independent of thread count, so it is computed once per index.

Usage:
    python step5_concurrency.py --data ../data --out ../data \\
        --ef 64 --nprobe 12 --threads 1 2 4 8
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from bench_utils import batch_qps, load_or_synthesize, recall_at_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../data")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, default=64, help="HNSW operating point")
    p.add_argument("--nlist", type=int, default=50)
    p.add_argument("--nprobe", type=int, default=12, help="IVF operating point")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def scaling(by_threads: dict[int, float]) -> dict[str, dict]:
    base = by_threads[min(by_threads)]
    t1 = min(by_threads)
    return {
        str(t): {
            "qps": q,
            "speedup": q / base,
            "efficiency": (q / base) / (t / t1),
        }
        for t, q in by_threads.items()
    }


def main() -> None:
    import faiss
    import hnswlib

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[1/4] Loading data ...")
    base, queries, gt = load_or_synthesize(
        args.data, args.n, args.dim, args.n_queries, 100, args.seed)
    dim = base.shape[1]
    print(f"  base={base.shape}  queries={queries.shape}")

    print(f"[2/4] Building HNSW (ef={args.ef}) + IVF (nprobe={args.nprobe}) ...")
    hnsw = hnswlib.Index(space="cosine", dim=dim)
    hnsw.init_index(max_elements=len(base),
                    ef_construction=args.ef_construction, M=args.M)
    hnsw.add_items(base)
    hnsw.set_ef(args.ef)

    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, args.nlist,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.train(base)
    ivf.add(base)
    ivf.nprobe = args.nprobe

    # recall once (thread-independent)
    h_labels, _ = hnsw.knn_query(queries, k=args.top_k, num_threads=1)
    hnsw_recall = recall_at_k(h_labels.tolist(), gt, args.top_k)
    faiss.omp_set_num_threads(1)
    _, i_ids = ivf.search(queries, args.top_k)
    ivf_recall = recall_at_k(i_ids.tolist(), gt, args.top_k)
    print(f"  HNSW recall@{args.top_k}={hnsw_recall:.4f}  "
          f"IVF recall@{args.top_k}={ivf_recall:.4f}")

    print("[3/4] Sweeping thread count ...")
    hnsw_by_t, ivf_by_t = {}, {}
    for t in args.threads:
        hnsw_by_t[t] = batch_qps(
            lambda q: hnsw.knn_query(q, k=args.top_k, num_threads=t), queries)

        faiss.omp_set_num_threads(t)
        ivf_by_t[t] = batch_qps(lambda q: ivf.search(q, args.top_k), queries)

        print(f"  threads={t:>2}  HNSW={hnsw_by_t[t]:9.0f} qps  "
              f"IVF={ivf_by_t[t]:9.0f} qps")

    print("[4/4] Saving results ...")
    out = {
        "config": {"num_vectors": len(base), "dim": dim,
                   "hnsw_ef": args.ef, "ivf_nprobe": args.nprobe,
                   "threads": args.threads},
        "recall": {"hnsw": hnsw_recall, "ivf": ivf_recall},
        "hnsw": scaling(hnsw_by_t),
        "ivf": scaling(ivf_by_t),
    }
    path = os.path.join(args.out, "concurrency_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
