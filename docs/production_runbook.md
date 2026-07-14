# Production Runbook — the 3.38M-vector experiment

The historical numbers in the
[main report](../README.md#1-historical-338m-vector-case-study) came from an
earlier form of this pipeline. Unlike the laptop-scale harness
(`scripts/step2..4`, which runs anywhere on synthetic data), this run needs the
proprietary OKW corpus, a GPU for embedding, a built DiskANN, and a fast NVMe
SSD — so it is **not** part of CI. This document records how it was run, so the
result is partially auditable even though the inputs aren't redistributable.
The historical raw engine outputs and exact DiskANN revision were not preserved;
the limitations below are therefore part of the result, not optional footnotes.

**Hardware.** Intel i7-13700F · RTX 4070 SUPER 12 GB · 32 GB RAM · 2 TB NVMe ·
Windows 11 Pro. **Dates.** 2026-05-14 … 15.

**Common data.** 3,384,857 × 1024-dim BGE-large-zh-v1.5 embeddings, **L2-normalized**
so inner product == cosine, and `squared_l2` ranking is identical too. The
historical query-preparation script sampled 1,000 vectors from the indexed corpus,
so self-matches were present. Ground truth was exact brute-force top-100 on GPU
(~79 s). The corrected script now creates disjoint base/query documents, but the
private-corpus experiment has not yet been rerun with that protocol.

```
scripts/step1_prepare_data.py   # embed corpus -> vectors.npy / query_vectors.npy / ground_truth.npy
```

---

## Path A — HNSW served by Milvus 2.5.10 (standalone)

Milvus **2.5.10** standalone via Docker (standalone + etcd + minio). Index params
`M=16, efConstruction=200`, `metric_type=IP` (matches normalized = cosine).

```bash
# 1. bring up Milvus 2.5.10 standalone
#    https://milvus.io/docs/v2.5.x/install_standalone-docker.md
docker compose up -d

# 2. insert, build, sweep ef_search using one query per client call
python scripts/production/milvus_hnsw_3.38M.py \
    --vectors data/vectors.npy \
    --queries data/query_vectors.npy \
    --gt      data/ground_truth.npy \
    --output  data/milvus_hnsw_raw.json
```

The table below is the **historical** sweep. Its client submitted all queries in
one Milvus search request; it must not be compared as sequential single-query
throughput. The corrected script above records per-query raw latencies.

**Memory** was read from the Milvus container's resident set (`docker stats`) —
11.7 GB at serve time. **Index size on disk** (7.9 GB) and **build time**
(22.3 min) are from the same run.

| ef | 16 | 32 | 64 | 100 | 150 | 200 |
|---|--:|--:|--:|--:|--:|--:|
| Recall@10 | 0.820 | **0.927** | 0.955 | 0.968 | 0.975 | 0.980 |
| QPS | 65.0 | **52.4** | 38.0 | 28.0 | 20.0 | 14.0 |

---

## Path B — DiskANN (microsoft/DiskANN, Rust)

The current [microsoft/DiskANN](https://github.com/microsoft/DiskANN) is a Rust
workspace driven by its `diskann-benchmark` runner; the disk-resident,
PQ-compressed index used here is the **`graph-index-build-pq`** benchmark. Build
DiskANN with `cargo build --release` first.

```bash
# 1. vectors -> DiskANN .bin (npts:u32, dim:u32, then row-major float32)
python scripts/production/npy_to_diskann_bin.py data/vectors.npy       data/base.bin
python scripts/production/npy_to_diskann_bin.py data/query_vectors.npy data/query.bin

# 2. exact top-100 ground truth (diskann-tools)
cargo run --release --package diskann-tools --bin compute_groundtruth -- \
    --data_type float --dist_fn l2 \
    --base_file  data/base.bin \
    --query_file data/query.bin \
    --gt_file    data/gt \
    --recall_at  100              # writes data/gt.bin

# 3. inspect the PQ build/search schema for your DiskANN revision
cargo run --release --package diskann-benchmark -- inputs graph-index-build-pq

# 4. build (R=32/L=64 and R=64/L=128) + search sweep, machine-readable output
cargo run --release --package diskann-benchmark -- run \
    --input-file  scripts/production/diskann_job.json \
    --output-file data/diskann_out.json
```

`scripts/production/diskann_job.json` is the job file. The fields that set the
two operating points (and how they map to the report) are:

| Job field | Meaning | Value |
|---|---|---|
| `source.max_degree` | max graph out-degree (**R**) | 32 / 64 |
| `source.l_build` | build search-list size (**L**) | 64 / 128 |
| `source.alpha` | Vamana α (RobustPrune) | 1.2 |
| `source.distance` | metric on the normalized vectors | `squared_l2` |
| `search_phase.runs[].search_l` | search-time recall/latency knob | swept |
| `search_phase.runs[].recall_k` | report Recall@**K** | 10 |
| `search_phase.num_threads` | threads (single-query numbers use `[1]`) | `[1]` |

> The PQ chunk count — which sets the resident footprint behind the "< 1 GB" —
> lives in the `graph-index-build-pq` schema printed by step 3; set it for the
> 1024-dim vectors, then reconcile this job file with that schema for your exact
> DiskANN commit before running. Because the original commit and resolved PQ
> configuration were not preserved, this job file is not sufficient to reproduce
> the historical numbers exactly.

| Config | Recall@10 | QPS | Avg latency | Search RAM | Index on disk | Build |
|---|--:|--:|--:|--:|--:|--:|
| R=32 / L=64 | 0.933 | 173.0 | 5.7 ms | < 1 GB | 31.3 GB | 49.0 min |
| R=64 / L=128 | 0.976 | 109.4 | 9.1 ms | < 1 GB | 31.3 GB | ~210 min |

> **Distance metric.** Vectors are L2-normalized, so `squared_l2` ordering is
> identical to cosine / inner product — keeping DiskANN consistent with the
> Milvus `IP` baseline.

---

## How each metric was captured

| Metric | Source |
|---|---|
| Recall@10 | both engines scored against the same exact top-100 ground truth |
| HNSW QPS / latency | historical: one batched client request; corrected script: sequential per-query wall-clock + raw percentiles |
| DiskANN QPS / latency | runner output with `num_threads=[1]`; raw historical output was not committed |
| HNSW search memory | Milvus container RSS via `docker stats` |
| DiskANN search memory | resident PQ table + node cache reported by the runner |
| Index size on disk | `du` of the Milvus segment files / the DiskANN index dir |

The reported values are collected in
[`data/benchmark_3.38M_summary.json`](../data/benchmark_3.38M_summary.json),
which is a manually transcribed summary, not raw engine output. A controlled
rerun should commit the Milvus JSON, DiskANN runner JSON, dependency/image
digests, CPU/thread settings, SSD model, cache policy, and a script that derives
the summary tables from those artifacts.
