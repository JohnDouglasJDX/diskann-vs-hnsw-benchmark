# DiskANN vs HNSW — A Vector Index Benchmark

[![CI](https://github.com/JohnDouglasJDX/diskann-vs-hnsw-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnDouglasJDX/diskann-vs-hnsw-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Empirical comparison of two approximate-nearest-neighbor (ANN) index families on
real Chinese academic-paper embeddings, at two scales:

- **Production scale — 3,384,857 vectors × 1024-dim** (BGE-large-zh-v1.5):
  **Microsoft DiskANN** (disk-resident graph) vs **HNSW** (in-memory graph,
  served by Milvus).
- **Laptop scale — 10,006 vectors × 1024-dim**: a fully reproducible mini-harness
  comparing **HNSW** (`hnswlib`) vs **IVF** (`faiss`, a disk-serializable,
  partition-based index) with complete parameter sweeps.

> **Context.** This work was done as part of vector-database / ANN research at
> the Institute of Computing Technology, Chinese Academy of Sciences. The corpus
> — **OKW (Open Knowledge World)** — aggregates ~3.38M scholarly records from PKP-based
> platforms (OPS preprints, OJS journal articles, OMP monographs): title +
> abstract fields across mathematics, physics, computer science, and astronomy.
> The raw vectors are not redistributed, but every script runs out-of-the-box on
> a synthetic fallback set and all measured numbers are included as JSON.

---

## TL;DR

At **3.38M** vectors, DiskANN is not just the memory-frugal option — it is also
the **faster** one, overturning the common "HNSW is fast, disk indexes are slow"
intuition:

| Metric (Recall@10 ≈ 93%) | HNSW (Milvus) | DiskANN (R=32/L=64) | DiskANN advantage |
|---|--:|--:|:--|
| **Search memory** | 11.7 GB | **< 1 GB** | **> 12× less RAM** |
| **QPS** | 52.4 | **173.0** | **3.3× higher** |
| **Avg latency** | 19.0 ms | **5.7 ms** | **3.3× lower** |
| Recall@10 | 0.927 | 0.933 | comparable |
| Index size on disk | **7.9 GB** | 31.3 GB | 4.0× larger (HNSW wins) |
| Build time | **22.3 min** | 49.0 min | 2.2× slower (HNSW wins) |

DiskANN trades **disk space and build time** for **RAM and query speed**. For a
memory-bound, read-heavy service that is an excellent trade; for a small or
frequently-rebuilt index, HNSW's simplicity and fast builds win.

![Core comparison](figures/core_comparison_3.38M.png)

---

## 1. The headline result (3.38M vectors)

`data/benchmark_3.38M_summary.json` holds the full numbers. Two DiskANN
operating points were measured by sweeping the max out-degree `R` and build
search-list `L`:

| Configuration | Search RAM | Index disk | Build time | Recall@10 | QPS | Avg latency |
|---|--:|--:|--:|--:|--:|--:|
| HNSW (Milvus), ef=32 | 11.7 GB | 7.9 GB | 22.3 min | 0.927 | 52.4 | 19.0 ms |
| DiskANN R=32 / L=64 | < 1 GB | 31.3 GB | 49.0 min | 0.933 | 173.0 | 5.7 ms |
| DiskANN R=64 / L=128 | < 1 GB | 31.3 GB | ~210 min | 0.976 | 109.4 | 9.1 ms |

### Recall–QPS: DiskANN dominates the frontier

Sweeping HNSW's `ef_search` traces the usual recall/throughput curve; both
DiskANN points sit **above and to the right** of it — higher QPS at equal or
better recall.

![Recall-QPS trade-off](figures/recall_qps_3.38M.png)

### Memory: the structural difference

HNSW must hold the entire graph + vectors in RAM (11.7 GB). DiskANN keeps only
~32-byte PQ codes plus a hot-node cache resident (< 1 GB) and pages the graph and
full-precision vectors from SSD on demand — at the cost of a larger on-disk
footprint.

![Memory footprint breakdown](figures/memory_breakdown_3.38M.png)

### Everything at once

![Radar — normalized multi-dimensional comparison](figures/radar_3.38M.png)

*(Lower is better for memory, latency, build time, disk; higher is better for
QPS and recall. DiskANN R=64 maximizes quality; R=32 maximizes throughput; HNSW
minimizes disk and build cost.)*

---

## 2. Why — the mechanism in one paragraph

HNSW's graph implicitly prunes edges with **α = 1**, leaving a large-diameter
graph that it patches with a multi-layer hierarchy. DiskANN's **Vamana** graph
prunes with a tunable **α > 1**, directly retaining long-range "jump" edges in a
single flat graph, so greedy search converges in **2–3× fewer hops**. On SSD,
fewer hops means fewer random reads — which is where the ~3.3× latency/QPS win
comes from. Product-quantized codes in RAM plus full-precision re-ranking
(piggy-backed on the same disk sector) then cut memory ~12× without losing
recall.

A full, citation-backed derivation — including the FreshDiskANN (streaming
updates) and VeloANN (async I/O) follow-ups — is in
**[docs/diskann_technical_analysis.md](docs/diskann_technical_analysis.md)**.

---

## 3. Reproducible laptop-scale harness (10K vectors)

The 3.38M run needed a desktop GPU, Milvus, and a Rust DiskANN build. To make
the methodology runnable anywhere, `scripts/` contains a clean, parameterized
harness that benchmarks **HNSW (`hnswlib`)** against **IVF (`faiss`)** — IVF
standing in for the *class* of partition-based, disk-serializable indexes — with
full parameter sweeps and identical measurement code.

Measured at 10,006 vectors (`data/hnsw_10k_results.json`,
`data/ivf_10k_results.json`):

| Operating point (Recall@10 ≈ 0.95) | HNSW | IVF |
|---|--:|--:|
| Parameter | ef=32 | nprobe=12 |
| Recall@10 | 0.9701 | 0.9573 |
| QPS | 1,522 | 2,487 |
| P50 / P99 latency | 0.63 / 1.06 ms | 0.33 / 0.73 ms |
| Index size | 40.5 MB | 39.4 MB |

![Recall-QPS at 10K](figures/recall_qps_10k.png)

**Note the scale-dependence.** At 10K, IVF *dominates* the recall–QPS frontier:
with so few clusters to scan, a partition index is nearly brute force and beats
the graph. The graph index's asymptotic advantage only appears at large `N`.
This is precisely why the production conclusion (3.38M) cannot be extrapolated
from a toy benchmark — and why both scales are reported here.

### Concurrent throughput: single-query QPS is not the ceiling

The QPS above is *latency-bound* — one query at a time. A serving system cares
about throughput under concurrency, so `step5_concurrency.py` fixes one operating
point per index and sweeps the query-thread count (`data/concurrency_10k_results.json`):

| Query threads | 1 | 2 | 4 | 8 |
|---|--:|--:|--:|--:|
| HNSW (ef=64) QPS | 3,615 | 7,182 | 12,933 | **20,089** |
| IVF (nprobe=12) QPS | 7,138 | 10,749 | 12,217 | 16,412 |

HNSW scales **5.56× at 8 threads** (parallel efficiency ≈ 0.69) and overtakes
IVF beyond two threads: greedy graph walks are embarrassingly parallel across
queries, while IVF — already near-brute-force at this scale — saturates memory
bandwidth sooner. *(Laptop/synthetic run; absolute QPS is hardware-specific, the
**scaling shape** is the point.)*

![Concurrent throughput scaling](figures/concurrency_10k.png)

### Run it

```bash
pip install -r requirements.txt

cd scripts
# (optional) build a real evaluation set from your own CSV + an embedding model:
python step1_prepare_data.py --csv your.csv --text-cols title abstract \
    --model BAAI/bge-large-zh-v1.5 --n 10000 --out ../data

# benchmark — if step1 wasn't run, a reproducible synthetic set is generated:
python step2_hnsw_benchmark.py --data ../data --out ../data
python step3_ivf_benchmark.py  --data ../data --out ../data
python step4_analysis.py       --data ../data --target-recall 0.95
python step5_concurrency.py    --data ../data --out ../data  # multi-thread QPS

cd ../figures && python make_figures.py
```

The full 3.38M production pipeline (Milvus + microsoft/DiskANN) is documented,
command-for-command, in **[docs/production_runbook.md](docs/production_runbook.md)**.

---

## 4. When to use which

| Choose **DiskANN** when… | Choose **HNSW** when… |
|---|---|
| Index ≫ RAM (10M–1B+ vectors) | Index comfortably fits in RAM |
| RAM is the cost/scaling bottleneck | RAM is plentiful; you want minimal moving parts |
| Read-heavy, latency-sensitive serving | Index is small or rebuilt frequently |
| A fast NVMe SSD is available | Disk space or build time is constrained |
| You can amortize a longer build | You need the simplest possible deployment |

At small scale (≲ 100K) the absolute differences are minor and an in-memory
index is usually the pragmatic default. DiskANN's advantages compound with
scale.

---

## 5. Methodology & honest caveats

**Common setup.** 1024-dim BGE-large-zh-v1.5 embeddings, L2-normalized so inner
product equals cosine similarity. Ground truth is exact brute-force top-k.
Metric is Recall@10. Latency is single-query wall-clock; QPS is sequential
single-query throughput (not batched/concurrent).

**Two experiments, different engines — read this before comparing across
sections:**

- The **3.38M** results compare **HNSW served by Milvus 2.5.10 (standalone)**
  against **Microsoft's Rust DiskANN** (the `diskann-benchmark` runner's
  `graph-index-build-pq` path), on a Windows desktop (i7-13700F, RTX 4070 SUPER,
  32 GB RAM, 2 TB NVMe). The exact build/search invocations are in
  **[docs/production_runbook.md](docs/production_runbook.md)**. Search memory for
  DiskANN is reported as "< 1 GB"; HNSW memory is the Milvus container's resident
  set.
- The **10K** results compare **`hnswlib`** against **`faiss` IVF** on a laptop.
  **IVF is not DiskANN** — it is a lightweight stand-in for partition-based
  disk indexes, used to keep the reproducible harness dependency-light. Do not
  read the 10K IVF numbers as DiskANN numbers.

**Limitations.**
- DiskANN's α was left at its default; sweeping α (≈1.2 is the usual sweet spot)
  and directly counting search hops are natural next steps.
- The headline QPS numbers are single-query (latency-bound). Multi-threaded
  scaling *is* measured at 10K (§3, `step5_concurrency.py`), but concurrent
  throughput at 3.38M — and async-I/O gains à la VeloANN — are not.
- The "< 1 GB" DiskANN memory is a working-set figure, not an exact RSS trace.
- Only two scales are measured; the curve between them is not characterized.

Anything not directly measured is labeled as such; no projected or illustrative
numbers are presented as measurements.

---

## 6. Repository layout

```
diskann-vs-hnsw-benchmark/
├── README.md                         # this report
├── requirements.txt  ·  requirements-dev.txt
├── .github/workflows/ci.yml          # tests + synthetic harness on every push
├── data/
│   ├── benchmark_3.38M_summary.json  # measured production numbers
│   ├── hnsw_10k_results.json         # full ef sweep
│   ├── ivf_10k_results.json          # full nprobe sweep
│   └── concurrency_10k_results.json  # thread-scaling sweep
├── scripts/
│   ├── bench_utils.py                # shared recall / latency / RSS / data
│   ├── step1_prepare_data.py         # embed corpus + ground truth
│   ├── step2_hnsw_benchmark.py       # HNSW sweep
│   ├── step3_ivf_benchmark.py        # IVF sweep
│   ├── step4_analysis.py             # Markdown comparison tables
│   ├── step5_concurrency.py          # multi-thread QPS scaling
│   └── production/                   # the real 3.38M pipeline
│       ├── milvus_hnsw_3.38M.py      #   Milvus 2.5.10 HNSW
│       ├── diskann_3.38M.sh          #   microsoft/DiskANN (Rust) build + search
│       ├── diskann_job.json          #   diskann-benchmark job (R=32 & R=64)
│       └── npy_to_diskann_bin.py     #   .npy -> DiskANN .bin
├── tests/
│   └── test_bench_utils.py           # recall / percentile / ground-truth units
├── figures/
│   ├── make_figures.py               # regenerates the 10K charts from data
│   └── *.png
└── docs/
    ├── diskann_technical_analysis.md # α-RNG / BeamSearch / PQ re-ranking
    └── production_runbook.md         # command-for-command 3.38M reproduction
```

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search on a Single Node*, NeurIPS 2019.
- Singh et al., *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for
  Streaming Similarity Search*, arXiv:2105.09613.
- *VeloANN: Optimizing SSD-Resident Graph Indexing for High-Throughput Vector
  Search*, PVLDB 2026, arXiv:2602.22805.
- [microsoft/DiskANN](https://github.com/microsoft/DiskANN) ·
  [nmslib/hnswlib](https://github.com/nmslib/hnswlib) ·
  [facebookresearch/faiss](https://github.com/facebookresearch/faiss) ·
  [Milvus](https://github.com/milvus-io/milvus)

## License

MIT — see [LICENSE](LICENSE). Benchmark numbers are from a specific
hardware/software setup; treat them as directional, and re-run on your own data
before making production decisions.
