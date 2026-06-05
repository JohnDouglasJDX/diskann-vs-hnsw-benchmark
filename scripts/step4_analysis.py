"""
Step 4 -- Compare the swept results and print Markdown tables.

Reads hnsw_results.json + ivf_results.json (from step2/step3) and emits a
GitHub-flavoured Markdown comparison: build cost, memory, and the operating
point closest to a target recall. Pipe it straight into a report.

Usage:
    python step4_analysis.py --data ../data --target-recall 0.95
"""
from __future__ import annotations

import argparse
import json
import os


def load(path: str):
    return json.load(open(path)) if os.path.exists(path) else None


def closest(by_param: dict, target: float):
    valid = {k: v for k, v in by_param.items() if v.get("recall") is not None}
    key = min(valid, key=lambda k: abs(valid[k]["recall"] - target))
    return key, valid[key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="../data")
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    hnsw = load(os.path.join(args.data, "hnsw_results.json"))
    ivf = load(os.path.join(args.data, "ivf_results.json"))
    if not hnsw or not ivf:
        raise SystemExit("Run step2 and step3 first.")

    n = hnsw["config"]["num_vectors"]
    print(f"### Index build & footprint ({n:,} vectors, "
          f"{hnsw['config']['dim']}-dim)\n")
    print("| Metric | HNSW | IVF |")
    print("|---|--:|--:|")
    print(f"| Build time (s) | {hnsw['build']['time_s']:.2f} | "
          f"{ivf['build']['time_s']:.2f} |")
    print(f"| Index size (MB) | {hnsw['build']['index_size_gb']*1024:.1f} | "
          f"{ivf['build']['index_size_gb']*1024:.1f} |")
    print(f"| Search RSS (MB) | {hnsw['memory']['search_rss_mb']:.0f} | "
          f"{ivf['memory']['search_rss_mb']:.0f} |")

    hk, hv = closest(hnsw["by_param"], args.target_recall)
    ik, iv = closest(ivf["by_param"], args.target_recall)
    print(f"\n### Operating point nearest Recall@10 ≈ {args.target_recall}\n")
    print("| Metric | HNSW | IVF |")
    print("|---|--:|--:|")
    print(f"| Param | ef={hk} | nprobe={ik} |")
    print(f"| Recall@10 | {hv['recall']:.4f} | {iv['recall']:.4f} |")
    print(f"| QPS | {hv['qps']:.1f} | {iv['qps']:.1f} |")
    print(f"| P50 (ms) | {hv['p50']:.2f} | {iv['p50']:.2f} |")
    print(f"| P95 (ms) | {hv['p95']:.2f} | {iv['p95']:.2f} |")
    print(f"| P99 (ms) | {hv['p99']:.2f} | {iv['p99']:.2f} |")


if __name__ == "__main__":
    main()
