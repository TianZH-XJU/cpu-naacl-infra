# CPU-CAST Stage-1: 计算资源测试与准备报告

生成时间（UTC）：2026-11-05

## 1. 计算资源测试

### 平台

| 项目 | 值 |
|---|---|
| CPU | Intel(R) Xeon(R) 6985P-C @ 2.30GHz |
| vCPU | 5（1 socket × 5 core，无超线程） |
| 虚拟化 | KVM guest（full virtualization） |
| ISA | AVX2、AVX-512 全家（F/DQ/CD/BW/VL/VNNI/BF16/FP16）、FMA、SHA |
| AMX | 不支持 |
| 内存 | 5.92 GiB 总量 / 5.51 GiB 可用，无 swap |
| 磁盘 | 约 5.4 TiB 可用 |
| 缓存（每核） | L1d 48 KiB / L1i 64 KiB / L2 2 MiB；L3 共享 504 MiB（KVM 呈现值） |

### Microbenchmark（torch 2.15.0+cpu，实际推理栈）

| 测试 | 1 线程 | 2 线程 | 4 线程 | 5 线程 |
|---|---|---|---|---|
| Matmul 1024² fp32（GFLOP/s） | 180.6 | 368.2 | 638.6 | 736.8 |
| Transformer layer（batch 32, seq 256, hidden 384, ms/层） | 253.7 | 140.1 | — | 85.5 |

- 内存带宽（256 MiB 拷贝）：40.0 GiB/s。
- **关键信号**：Transformer 层 5 线程仅获得 3.0× 加速（1→5 线程），低于 matmul 的 4.1×。
  小 batch Transformer 受同步/内存约束更强，与 CPU-CAST 关心的"减少 FLOPs ≠ 减少 wall-clock"一致。
  注意：本机是 5 vCPU 云主机，与目标 2-vCPU 设备不同，仅用于验证探测方法可用。

## 2. 环境准备

Python 3.12.15，pip 安装（用户级）：

| 包 | 版本 |
|---|---|
| torch | 2.15.0+cpu |
| transformers | 5.22.1 |
| sentence-transformers | 6.2.0 |
| onnxruntime | 1.31.1 |
| beir | 2.2.0 |
| fastembed / pytrec-eval-terrier / datasets / pandas / psutil / huggingface_hub | 已装（见 manifests/python_env.txt） |

未安装：`pyserini` 运行时依赖 JDK（本机无 Java），仓库已克隆备用；`openvino`、`torchao` 按计划留到量化/runtime 对比阶段再装。

## 3. 数据集（官方 BeIR HuggingFace 仓库 → 经典 BEIR 目录）

| 数据集 | docs | queries 总数 | qrels (train/dev/test) | corpus parquet 大小 |
|---|---|---|---|---|
| scifact | 5,183 | 1,109 | 919 / — / 339 | 4.26 MiB |
| nfcorpus | 3,633 | 3,237 | 110,575 / 11,385 / 12,334 | 3.01 MiB |
| arguana | 8,674 | 1,406 | — / — / 1,406 | 4.76 MiB |
| fiqa | 57,638 | 6,648 | 14,166 / 1,238 / 1,706 | 26.42 MiB |

- 目录：`data/<name>/{corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv}`（原始 parquet 保留在 `data/<name>/raw/`）。
- 每个 parquet 的 SHA-256 记录在 `manifests/datasets_manifest.json`。
- 注意：queries 总数包含 train/dev；test query 数以 qrels test.tsv 为准（如 scifact test=339）。
- MS MARCO Passage 未在本轮下载（大规模候选实验阶段再取）。

## 4. 基线与工具仓库（shallow clone，SHA 已固定）

| 仓库 | 用途 | commit |
|---|---|---|
| ibrahimjamadi98/GAINER | Gain-aware depth allocation（预印本，直接竞争基线） | 074ac4b |
| alexlimh/CEC-Ranking | 候选剪枝的 certified error control | 6bec43b |
| castorini/earlyexiting-monobert | Learned early exit | 29b7ac9 |
| veneres/SEE-SIGIR25 | Similarity-based early exit（SIGIR 2025） | 06431a7 |
| xpmir/mice | Minimal Interaction Cross-Encoders（EMNLP 2026） | 9b8e1fe |
| lightonai/fast-plaid | 多向量索引/文档表示压缩 | df2be46 |
| castorini/pyserini | BM25 / candidate runs（需 JDK） | c846a61 |

Prism（EuroSys 2026）未找到可确认的官方实现，按计划以论文描述独立复现并标注 reimplementation。
完整 URL 与 commit 见 `manifests/baselines_manifest.txt`。

## 5. 模型权重（HuggingFace 本地化，共约 906 MiB）

| 模型 | 用途 | 大小 |
|---|---|---|
| BAAI/bge-small-en-v1.5 | 第一阶段 embedding | 255.5 MiB |
| sentence-transformers/all-MiniLM-L6-v2 | embedding 对照 | 174.0 MiB |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 极小浅层对照 | 34.4 MiB |
| cross-encoder/ms-marco-MiniLM-L-2-v2 | 2 层浅模型 | 120.1 MiB |
| cross-encoder/ms-marco-MiniLM-L-4-v2 | 4 层中间基线 | 147.2 MiB |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | **主模型** | 174.3 MiB |

Ettin family（17M/32M/68M）与 MICE-P 按计划留到后续阶段。

## 6. 冒烟验证（全部通过）

- 4 个交叉编码器从 `models/` 本地加载，query-doc 打分方向正确（相关 ≈ +7~8，无关 ≈ -6~-12）。
  MiniLM-L6 每对约 0.9 ms（batch=3，4 线程）。
- BGE-small 嵌入维度 384，语义相似度方向正确（相关对 0.776）。
- 4 个 BEIR 数据集 corpus/queries/qrels 结构与列名校验通过。
- 明细：`infra/results/smoke_test.json`、`infra/results/hardware_probe.{json,md}`。

## 7. 与 Gate A 的衔接

下一阶段（Gate A）应在本机直接做 compaction vs keep microbenchmark：
用 MiniLM-L6 真实权重，测 C_keep 与 C_compact 随 active-set 比例（100%→5%）变化的曲线，
并区分 gather/拷贝/内存分配开销与纯计算开销。
