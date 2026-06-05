"""
Regenerate the 10K small-scale Recall-QPS figure from the swept result JSONs.

The production-scale (3.38M) figures were produced on the Windows benchmark
machine and are committed as PNGs; this script makes the laptop-scale figure
reproducible from data/*_10k_results.json.

    python make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


HNSW_C, IVF_C = "#534AB7", "#D9782A"


def curve(path):
    bp = json.load(open(path))["by_param"]
    items = sorted(bp.values(), key=lambda v: v["recall"])
    return [v["recall"] for v in items], [v["qps"] for v in items]


def recall_qps_figure() -> None:
    hr, hq = curve(os.path.join(DATA, "hnsw_10k_results.json"))
    ir, iq = curve(os.path.join(DATA, "ivf_10k_results.json"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hr, hq, "o-", color=HNSW_C, lw=2, ms=7, label="HNSW (ef sweep)")
    ax.plot(ir, iq, "s-", color=IVF_C, lw=2, ms=7, label="IVF (nprobe sweep)")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (single-query)")
    ax.set_title("Recall–QPS trade-off — 10K vectors, 1024-dim "
                 "(BGE-large-zh-v1.5)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "recall_qps_10k.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def concurrency_figure() -> None:
    path = os.path.join(DATA, "concurrency_10k_results.json")
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
    ax.set_title("Concurrent throughput scaling — 10K vectors, 1024-dim")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "concurrency_10k.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main() -> None:
    recall_qps_figure()
    concurrency_figure()


if __name__ == "__main__":
    main()
