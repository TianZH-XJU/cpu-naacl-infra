# CPU-CAST Stage 0/1 + Gate A 执行报告

执行时间（UTC）：2026-11-05 ~ 2026-11-06。对应 CPU-CAST v2 计划的 Stage 0（第 1 周）、Stage 1（第 2–3 周）与 Gate A 前置验证。

## 0. 环境探针（artifacts/env/）

| 项 | 值 | 与计划目标环境对比 |
|---|---|---|
| CPU | Xeon 6985P-C @ 2.30GHz，5 vCPU 在线 | 计划目标 2 vCPU |
| **cgroup cpu.max** | **400000/100000 = 4 vCPU 有效配额** | 接近目标；基准以 1/2 线程为主 |
| **cgroup memory.max** | **4,236,247,040 B ≈ 3.95 GiB** | 与计划 3.8 GB 几乎一致 |
| ISA | AVX2/AVX512 全套 + VNNI + BF16/FP16，无 AMX | torch 实际使用 AVX512 内核 |
| 亲和性 | 0-4（未绑定单核） | 正式实验需 taskset 固定 |
| perf | 未安装，硬件计数器未测 | 记录为限制 |
| 缓存 | L1d 48K / L2 2M / L3 504M（共享），行 64B | — |

## 1. Stage 0 交付物

- `artifacts/env/{system_probe,cache_probe,perf_probe}.txt`（含 cgroup、affinity、ORT/torch 能力）
- `configs/{models,datasets,runtime,workloads}.yaml`
- `cpu_cast/` 包：`logging_utils.py`（统一 JSONL logger）、`data.py`（BEIR loader）、`metrics.py`（pytrec_eval）
- 固定候选 run（test split，top-100，BGE-small 离线编码 + rank_bm25 + RRF）：

| 数据集 | 方法 | Recall@100 |
|---|---|---|
| scifact (300 test q) | BM25 / Dense / RRF | 0.7876 / 0.9587 / 0.9469 |
| nfcorpus (323 test q) | BM25 / Dense / RRF | 0.1340 / 0.2029 / 0.2064 |

  注：nfcorpus 平均每 query 相关文档多（~38），Recall@100 天然受限；后续如需更高召回可加大 candidate budget。
- 端到端 smoke test（`artifacts/runs/e2e_smoke.jsonl`）：query 5（scifact test），RRF top-100 → MiniLM-L6 重排，相关文档排第 1，candidate nDCG@10=1.0，MRR=1.0；tokenize 34ms，rerank 2.51s（100 对，seq≈256，2 线程，≈25ms/对）。
- 修复记录：pytrec_eval measure 名必须用下划线（`ndcg_cut_10`），点号形式会静默返回 0。

## 2. Stage 1：reranker 基线（scifact，RRF top-100，seq≤256，2 线程）

质量：前 100 个 test query 的候选级 nDCG@10；延迟：20 个 query，batch-1（K=10 逐对）与 batch-K（K=100 单批）。

| 模型 | 后端 | nDCG@10 | batch-1 p50 (ms/q) | batch-K p50 (ms/q) |
|---|---|---|---|---|
| MiniLM-L6 | torch FP32 | 0.7038 | 2697.9 | 2962.0 |
| MiniLM-L6 | ORT FP32 | 0.7038 | 2965.0 | 4027.1 |
| MiniLM-L6 | ORT INT8-dyn | 0.6996 | 2071.9 | 2887.6 |
| MiniLM-L4 | torch FP32 | **0.7072** | 1840.9 | 1917.7 |
| MiniLM-L4 | ORT INT8-dyn | 0.7010 | 1432.6 | 1872.7 |
| MiniLM-L2 | torch FP32 | 0.6701 | 982.9 | 983.1 |
| TinyBERT-L2 | torch FP32 | 0.6960 | 324.8 | **196.4** |
| TinyBERT-L2 | ORT INT8-dyn | 0.6877 | 245.2 | 198.9 |

初步结论（对应计划 §16 决策表）：
- **ORT INT8 在此形状上收益有限**（batch-K 仅比 torch FP32 快约 2.5%）；FP32 路线暂可保留。
- **静态浅层模型非常强**：TinyBERT-L2 比 L6 快约 15 倍，nDCG 只降 0.008；L4 甚至略优于 L6。这压缩了"动态深度分配"的收益空间——Stage 2 必须把静态浅层作为强基线。
- ORT FP32 在大 batch 动态形状上慢于 PyTorch（4.0s vs 3.0s），后续需尝试 bucket 后静态 shape。
- 完整数据：`artifacts/runs/bench_reranker.jsonl`（含 p95、RSS；1 线程矩阵待补）。

## 3. Gate A：active-set compaction break-even（核心结果）

方法：MiniLM-L6 真实权重逐层执行（0-indexed layer 2），真实 tokenized (query, doc) 对；同一层分别测量
C_keep（全 batch 含无效行前向）与 C_compact（gather + 活跃行前向 + scatter 回写）；warmup 5、30 次测量取均值。

**增益（gain% = 1 − C_compact/C_keep；\* 表示 compact 严格更快）**：

| T | seq | f \ batch | 8 | 16 | 32 | 64 | 100 |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 0.75 | +24.7\* | +19.6\* | +24.9\* | +20.1\* | +26.0\* |
| 1 | 128 | 0.50 | +43.1\* | +49.6\* | +49.0\* | +54.3\* | +57.1\* |
| 1 | 128 | 0.10 | +85.0\* | +86.2\* | +89.6\* | +92.0\* | +91.9\* |
| 2 | 256 | 0.75 | +22.0\* | +23.3\* | +29.8\* | +27.3\* | +26.3\* |
| 2 | 256 | 0.50 | +48.9\* | +49.4\* | +57.9\* | +53.9\* | +54.0\* |
| 2 | 256 | 0.10 | +84.9\* | +86.1\* | +90.8\* | +91.9\* | +92.3\* |

- f=1.0 时两者持平（−3% ~ +3%）；gather/scatter 开销 0.03–8.6 ms（多数 <1.5 ms），相对层耗时 10–100 ms 可忽略。
- 结论在 batch 8–100、seq 128/256、1–2 线程全部成立。
- **Gate A 判定：通过**。低核数 CPU 上存在真实的 active-set execution gap，compaction 有稳定净收益区间（活跃比例 ≲0.75 即开始获益）。
- 边界条件（写入 Stage 3 待办）：单层隔离测量（PyTorch eager、layer 2）；需在真实退出模式、多层累积、early-exit 判定开销共存下复核；compaction interval m=1/2/3/4 对比未做。

## 4. 下一步（按计划）

1. Stage 1 补充：1 线程矩阵、长度分桶、ORT 静态 shape、p99。
2. Stage 2：SEE / Xin early-exit / GAINER / Prism-reimpl 复现（仓库已克隆并固定 commit）。
3. Stage 3：完整 cost grid（compaction interval、INT8 break-even、FLOP-proxy 对比）。
4. Gate A 复核：真实 early-exit 轨迹下的端到端 compaction 收益。

## 5. 已知限制

- 本机 4 vCPU 配额 / 5 vCPU 在线，非目标 2 vCPU 设备；所有结论需在目标硬件复测。
- perf 未安装，无硬件计数器；无能耗测量。
- 质量 nDCG 基于 scifact test 前 100 个 query，非完整 300。
- 候选 run 为本机生成的 BM25/dense/RRF，非官方预生成 run；已固定并记录 Recall 供追溯。
