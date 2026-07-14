"""
Regenerate the small-scale Recall-QPS figure from the canonical result JSONs.

The production-scale (3.38M) figures were produced on the Windows benchmark
machine and are committed as PNGs; this script makes the laptop-scale figures
reproducible from data/*_results.json.

    python make_figures.py
"""
from __future__ import annotations

import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULT_FILES = ("hnsw_results.json", "ivf_results.json",
                "concurrency_results.json")


HNSW_C, IVF_C = "#534AB7", "#D9782A"


def curve(path):
    bp = json.load(open(path))["by_param"]
    items = sorted(bp.values(), key=lambda v: v["recall"])
    return [v["recall"] for v in items], [v["qps"] for v in items]


def recall_qps_figure() -> None:
    hr, hq = curve(os.path.join(DATA, "hnsw_results.json"))
    ir, iq = curve(os.path.join(DATA, "ivf_results.json"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hr, hq, "o-", color=HNSW_C, lw=2, ms=7, label="HNSW (ef sweep)")
    ax.plot(ir, iq, "s-", color=IVF_C, lw=2, ms=7, label="IVF (nprobe sweep)")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (single-query)")
    ax.set_title("Recall–QPS smoke test — 10K synthetic vectors, "
                 "1024-dim, disjoint queries")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "recall_qps_10k.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def concurrency_figure() -> None:
    path = os.path.join(DATA, "concurrency_results.json")
    if not os.path.exists(path):
        print("skip concurrency figure (run step5_concurrency.py first)")
        return
    d = json.load(open(path))
    threads = sorted(int(t) for t in d["hnsw"])
    hq = [d["hnsw"][str(t)]["qps"] for t in threads]
    iq = [d["ivf"][str(t)]["qps"] for t in threads]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threads, hq, "o-", color=HNSW_C, lw=2, ms=7,
            label=f"HNSW (ef={d['config']['hnsw_ef']})")
    ax.plot(threads, iq, "s-", color=IVF_C, lw=2, ms=7,
            label=f"IVF (nprobe={d['config']['ivf_nprobe']})")
    ideal = [hq[0] * t / threads[0] for t in threads]
    ax.plot(threads, ideal, "k--", lw=1, alpha=0.5, label="linear scaling (HNSW)")
    ax.set_xlabel("query threads")
    ax.set_ylabel("QPS (batched / concurrent)")
    ax.set_xticks(threads)
    ax.set_title("Concurrent throughput smoke test — 10K synthetic vectors, "
                 "1024-dim")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "concurrency_10k.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def write_data_manifest() -> None:
    """Record deterministic source hashes without comparing platform-specific PNGs."""
    manifest = {}
    for name in RESULT_FILES:
        path = os.path.join(DATA, name)
        with open(path, "rb") as f:
            manifest[name] = hashlib.sha256(f.read()).hexdigest()
    out = os.path.join(HERE, "result_data_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote", out)


def main() -> None:
    recall_qps_figure()
    concurrency_figure()
    write_data_manifest()


if __name__ == "__main__":
    main()
