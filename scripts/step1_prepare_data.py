"""
Step 1 -- Prepare the evaluation set.

Splits a text corpus into disjoint base/query documents, encodes both with a
SentenceTransformer model, and computes exact (brute-force) top-k ground truth.
Outputs: vectors.npy, query_vectors.npy, ground_truth.npy.

If you don't have the corpus / model, you can skip this step entirely: the
benchmark scripts (step2/step3) fall back to a reproducible synthetic set.

Usage:
    python step1_prepare_data.py --csv data.csv --text-cols title description \\
        --model BAAI/bge-large-zh-v1.5 --n 10000 --out ../data
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from bench_utils import exact_topk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="input CSV with text columns")
    p.add_argument("--text-cols", nargs="+", default=["title", "description"])
    p.add_argument("--model", default="BAAI/bge-large-zh-v1.5")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto)")
    p.add_argument("--n", type=int, default=10000,
                   help="number of indexed/base rows (queries are additional)")
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=100, help="ground-truth depth")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="../data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    required = args.n + args.n_queries
    print(f"[1/4] Reading {required} rows from {args.csv} ...")
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= required:
                break
            rows.append(row)
    if len(rows) < required:
        raise SystemExit(
            f"need at least {required} rows for {args.n} base + "
            f"{args.n_queries} disjoint queries; found {len(rows)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    query_rows = [rows[i] for i in order[:args.n_queries]]
    base_rows = [rows[i] for i in order[args.n_queries:required]]
    make_text = lambda r: " ".join((r.get(c) or "") for c in args.text_cols)
    base_texts = [make_text(r) for r in base_rows]
    query_texts = [make_text(r) for r in query_rows]
    texts = base_texts + query_texts
    print(f"  loaded {len(base_texts)} base + {len(query_texts)} disjoint queries")

    print(f"[2/4] Encoding with {args.model} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    t0 = time.time()
    encoded = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # => inner product == cosine
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    dt = time.time() - t0
    vectors = encoded[:args.n]
    queries = encoded[args.n:]
    print(f"  {encoded.shape} in {dt:.1f}s ({len(texts) / dt:.1f} texts/s)")
    np.save(os.path.join(args.out, "vectors.npy"), vectors)

    print(f"[3/4] Saving {args.n_queries} held-out queries ...")
    np.save(os.path.join(args.out, "query_vectors.npy"), queries)

    print(f"[4/4] Exact top-{args.top_k} ground truth ...")
    gt = exact_topk(vectors, queries, args.top_k)
    np.save(os.path.join(args.out, "ground_truth.npy"), gt)

    print(f"\nDone. Wrote vectors/queries/ground_truth to {args.out!r}")


if __name__ == "__main__":
    main()
