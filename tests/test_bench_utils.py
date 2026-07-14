"""
Unit tests for the measurement core in ``scripts/bench_utils.py``.

These guard the parts every benchmark number depends on: the recall definition,
the latency percentiles, the exact ground-truth search, and the determinism of
the synthetic fallback set. If these are wrong, every reported number is wrong —
so they are the things worth pinning down.
"""
import numpy as np
import pytest

import bench_utils as bu


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
def test_recall_at_k_perfect():
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    results = [[1, 2, 3], [4, 5, 6]]
    assert bu.recall_at_k(results, gt, 3) == 1.0


def test_recall_at_k_partial():
    # row 0: {1,2} of {1,2,3} hit -> 2/3 ; row 1: {4} of {4,5,6} -> 1/3
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    results = [[1, 2, 9], [4, 8, 7]]
    assert bu.recall_at_k(results, gt, 3) == pytest.approx((2 + 1) / 6)


def test_recall_at_k_truncates_to_k():
    # only the first k entries of each list count
    gt = np.array([[1, 2, 3, 4]])
    results = [[1, 2, 99, 98]]
    assert bu.recall_at_k(results, gt, 2) == 1.0
    assert bu.recall_at_k(results, gt, 4) == 0.5


def test_recall_at_k_rejects_mismatched_query_counts():
    with pytest.raises(ValueError, match="same queries"):
        bu.recall_at_k([[1, 2]], np.array([[1, 2], [3, 4]]), 2)


# --------------------------------------------------------------------------- #
# percentile_stats
# --------------------------------------------------------------------------- #
def test_percentile_stats_ordering_and_values():
    stats = bu.percentile_stats(list(range(1, 101)))  # 1..100
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["p50"] <= stats["p95"] <= stats["p99"]


# --------------------------------------------------------------------------- #
# batch_qps
# --------------------------------------------------------------------------- #
def test_batch_qps_positive_and_runs_repeats():
    queries = np.zeros((100, 4), dtype=np.float32)
    calls = []
    qps = bu.batch_qps(lambda x: calls.append(len(x)), queries, repeats=3)
    assert np.isfinite(qps) and qps > 0
    assert calls == [100, 100, 100]  # full batch, once per repeat


# --------------------------------------------------------------------------- #
# _l2_normalize
# --------------------------------------------------------------------------- #
def test_l2_normalize_unit_norm():
    x = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    norms = np.linalg.norm(bu._l2_normalize(x), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_handles_zero_rows():
    x = np.zeros((3, 4), dtype=np.float32)
    out = bu._l2_normalize(x)
    assert np.isfinite(out).all()  # no divide-by-zero blow-up


# --------------------------------------------------------------------------- #
# exact_topk
# --------------------------------------------------------------------------- #
def test_exact_topk_matches_bruteforce():
    rng = np.random.default_rng(123)
    base = rng.standard_normal((80, 16)).astype(np.float32)
    queries = base[:5]
    k = 5

    got = exact = None
    got = bu.exact_topk(base, queries, k)
    sims = queries @ base.T
    exact = np.argsort(-sims, axis=1)[:, :k]

    for g, e in zip(got, exact):
        assert set(int(x) for x in g) == set(int(x) for x in e)


# --------------------------------------------------------------------------- #
# load_or_synthesize (fallback path)
# --------------------------------------------------------------------------- #
def test_synthesize_is_deterministic_and_well_shaped(tmp_path):
    kw = dict(data_dir=str(tmp_path), n=500, dim=16, n_queries=20, top_k=10)
    b1, q1, g1 = bu.load_or_synthesize(seed=7, **kw)
    b2, q2, g2 = bu.load_or_synthesize(seed=7, **kw)

    assert b1.shape == (500, 16)
    assert q1.shape == (20, 16)
    assert g1.shape == (20, 10)
    assert np.array_equal(b1, b2)  # same seed -> same data
    assert np.array_equal(g1, g2)


def test_synthesized_queries_are_disjoint_from_base(tmp_path):
    base, queries, gt = bu.load_or_synthesize(
        str(tmp_path), n=400, dim=16, n_queries=15, top_k=10, seed=1)
    assert gt.shape == (15, 10)
    assert not any(
        np.allclose(base_row, query, atol=1e-7)
        for query in queries
        for base_row in base
    )
