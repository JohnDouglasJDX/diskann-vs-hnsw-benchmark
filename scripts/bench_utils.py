"""
Shared utilities for the ANN benchmark harness.

Keeps the per-index scripts (HNSW, IVF) small and consistent: every index is
measured the same way -- same ground truth, same recall definition, same
latency percentiles, same RSS sampling -- so the numbers are comparable.
"""
from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Sequence

import numpy as np


def environment_info() -> dict[str, object]:
    """Minimal machine/runtime provenance stored with every benchmark result."""
    packages = {}
    for package in ("numpy", "hnswlib", "faiss-cpu", "psutil"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "packages": packages,
    }


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def rss_mb() -> float:
    """Resident set size of the current process, in MiB."""
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
def recall_at_k(results: Sequence[Sequence[int]],
                 ground_truth: np.ndarray,
                 k: int) -> float:
    """Mean Recall@k = |retrieved_k ∩ truth_k| / k, averaged over queries."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(results) != len(ground_truth):
        raise ValueError("results and ground_truth must contain the same queries")
    if len(ground_truth) == 0:
        raise ValueError("cannot compute recall for an empty query set")
    hits = sum(
        len(set(r[:k]) & set(gt[:k]))
        for r, gt in zip(results, ground_truth)
    )
    return hits / (len(ground_truth) * k)


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def measure_latencies(search_fn: Callable[[np.ndarray], object],
                      queries: np.ndarray,
                      warmup: int = 50) -> list[float]:
    """Per-query wall-clock latencies in milliseconds (single-query path)."""
    for q in queries[:warmup]:
        search_fn(q)
    lats = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def percentile_stats(latencies: Sequence[float]) -> dict[str, float]:
    """P50/P95/P99/mean using numpy's linear-interpolation percentiles."""
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


# --------------------------------------------------------------------------- #
# Concurrent throughput
# --------------------------------------------------------------------------- #
def batch_qps(search_batch_fn: Callable[[np.ndarray], object],
              queries: np.ndarray,
              repeats: int = 3) -> float:
    """
    Throughput (queries/sec) of one *batched* search call, where the engine
    parallelizes the batch internally (hnswlib ``num_threads`` / faiss OpenMP).

    Reports the median of ``repeats`` runs. Reporting the best run systematically
    inflates throughput and hides scheduler/cache variability.
    """
    elapsed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        search_batch_fn(queries)
        elapsed.append(time.perf_counter() - t0)
    return len(queries) / statistics.median(elapsed)


# --------------------------------------------------------------------------- #
# Data loading (real embeddings if present, else a reproducible synthetic set)
# --------------------------------------------------------------------------- #
def load_or_synthesize(data_dir: str,
                       n: int,
                       dim: int,
                       n_queries: int,
                       top_k: int,
                       seed: int = 42):
    """
    Return (base, queries, ground_truth).

    If ``data_dir`` already holds ``vectors.npy`` / ``query_vectors.npy`` /
    ``ground_truth.npy`` (produced by step1), load them. Otherwise synthesize a
    reproducible clustered base and a separate query set. Queries are never
    copied from the indexed base: self-matches would make Recall@k misleadingly
    easy and would not resemble a serving workload.
    """
    vp = os.path.join(data_dir, "vectors.npy")
    qp = os.path.join(data_dir, "query_vectors.npy")
    gp = os.path.join(data_dir, "ground_truth.npy")

    if os.path.exists(vp) and os.path.exists(qp) and os.path.exists(gp):
        base = np.load(vp).astype(np.float32)
        queries = np.load(qp).astype(np.float32)
        gt = np.load(gp)
        if base.ndim != 2 or queries.ndim != 2 or gt.ndim != 2:
            raise ValueError("vectors, queries, and ground truth must be 2-D arrays")
        if base.shape[1] != queries.shape[1]:
            raise ValueError("base and query dimensions do not match")
        if len(queries) != len(gt):
            raise ValueError("query and ground-truth counts do not match")
        if gt.shape[1] < top_k:
            raise ValueError(f"ground truth has depth {gt.shape[1]}, need {top_k}")
        return base, queries, gt

    print(f"  [synthetic] no real vectors in {data_dir!r}; "
          f"generating {n}x{dim} clustered vectors (seed={seed})")
    rng = np.random.default_rng(seed)
    n_centers = max(8, n // 500)
    centers = rng.standard_normal((n_centers, dim)).astype(np.float32)
    base_assign = rng.integers(0, n_centers, size=n)
    base = (centers[base_assign]
            + 0.35 * rng.standard_normal((n, dim)).astype(np.float32))
    base = _l2_normalize(base)

    query_assign = rng.integers(0, n_centers, size=n_queries)
    queries = (centers[query_assign]
               + 0.35 * rng.standard_normal((n_queries, dim)).astype(np.float32))
    queries = _l2_normalize(queries)
    gt = exact_topk(base, queries, top_k)
    return base, queries, gt


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute-force inner-product top-k ground truth (FAISS if available)."""
    if not 0 < k <= len(base):
        raise ValueError(f"k must be in [1, {len(base)}], got {k}")
    try:
        import faiss

        index = faiss.IndexFlatIP(base.shape[1])
        index.add(base)
        _, gt = index.search(queries, k)
        return gt
    except ImportError:
        sims = queries @ base.T
        candidates = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(sims, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        return np.take_along_axis(candidates, order, axis=1)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)
