# CPU-CAST Stage-1 Infra

CPU-Aware Active-Set Scheduling for Cross-Encoder Reranking (CPU-CAST) —
Stage-1 infrastructure: compute-resource probe, environment, datasets,
baseline repositories, and model weights.

## Layout

| Path | Purpose | Tracked |
|---|---|---|
| `scripts/probe_hardware.py` | Static platform info + torch thread-scaling / memory-bandwidth microbenchmarks | yes |
| `scripts/download_datasets.py` | Download official BeIR HF repos → classic BEIR layout under `data/` | yes |
| `scripts/download_models.py` | Download Stage-1 model weights into `models/` | yes |
| `scripts/clone_baselines.sh` | Shallow-clone baseline repos into `baselines/`, pin SHAs | yes |
| `scripts/smoke_test.py` | Load every model locally, score pairs, verify BEIR structure | yes |
| `infra/results/` | Probe and smoke-test outputs (JSON + Markdown) | yes |
| `manifests/` | Dataset / model / baseline / pip manifests for reproducibility | yes |
| `data/`, `models/`, `baselines/` | Heavy artifacts (~1.1 GiB), reproducible from scripts | no |

## Reproduce

```sh
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python3 scripts/probe_hardware.py
python3 scripts/download_datasets.py
python3 scripts/download_models.py
sh scripts/clone_baselines.sh
python3 scripts/smoke_test.py
```

Environment note: install torch first from the CPU wheel index, then the rest
from PyPI. `pyserini` additionally requires a JDK at runtime (not available in
this sandbox; repository cloned for reference).

## Stage-1 result summary (2026-11-05)

- Platform: Intel Xeon 6985P-C, 5 vCPU (KVM guest), AVX-512 + AVX512-VNNI + BF16, 5.92 GiB RAM
- Matmul thread scaling (1024² fp32): 1T 180.6 → 2T 368.2 → 4T 638.6 → 5T 736.8 GFLOP/s
- Transformer layer (batch 32, seq 256, hidden 384): 1T 253.7 ms → 5T 85.5 ms (3.0× on 5 threads — sublinear scaling, relevant to CPU-CAST Gate A)
- Memory bandwidth: 40.0 GiB/s
- Datasets verified: scifact 5,183 docs / nfcorpus 3,633 / arguana 8,674 / fiqa 57,638
- Models verified locally: TinyBERT-L2, MiniLM-L2/L4/L6 cross-encoders, BGE-small, all-MiniLM-L6-v2
