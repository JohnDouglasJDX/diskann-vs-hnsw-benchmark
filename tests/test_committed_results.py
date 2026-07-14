"""Contract tests for the canonical, user-visible benchmark artifacts."""
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(name: str) -> dict:
    with (DATA / name).open(encoding="utf-8") as f:
        return json.load(f)


def test_canonical_sweep_results_have_provenance_and_disjoint_queries():
    for name in ("hnsw_results.json", "ivf_results.json"):
        result = load(name)
        assert result["config"]["data_source"] == "synthetic-disjoint-queries"
        assert result["config"]["num_queries"] > 0
        assert result["environment"]["python"]
        assert result["environment"]["packages"]["numpy"]
        assert result["by_param"]
        for row in result["by_param"].values():
            assert 0.0 <= row["recall"] <= 1.0
            assert row["qps"] > 0.0
            assert row["p50"] <= row["p95"] <= row["p99"]


def test_canonical_concurrency_result_documents_aggregation():
    result = load("concurrency_results.json")
    assert result["config"]["data_source"] == "synthetic-disjoint-queries"
    assert result["config"]["aggregation"] == "median of 3 batched runs"
    assert set(result["hnsw"]) == set(result["ivf"])


def test_production_summary_is_explicitly_historical():
    result = load("benchmark_3.38M_summary.json")
    assert result["status"] == "historical_case_study_not_controlled_benchmark"
    assert len(result["limitations"]) >= 3


def test_figure_manifest_matches_canonical_result_data():
    manifest = load("../figures/result_data_manifest.json")
    for name in ("hnsw_results.json", "ivf_results.json",
                 "concurrency_results.json"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert manifest[name] == digest
