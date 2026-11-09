# CPU-CAST Stage 2：强基线矩阵报告

日期：2026-11-06　环境：KVM 虚拟机，4 vCPU 有效配额（cgroup cpu.max=400000/100000），~3.95 GiB RAM，2 线程推理（OMP_NUM_THREADS=2, torch.set_num_threads(2)）

## 0. 范围与设置

- 数据集：scifact（100 条 test query 子集），固定候选 run `data/candidate_runs/scifact/rrf_top100.jsonl`（BM25+dense RRF top-100，Recall@100=0.9469；dense 单路为 0.9587——早期报告误将 dense 召回记在 RRF run 上，已按 manifests/candidate_runs/scifact_manifest.json 更正）
- 主模型：`cross-encoder/ms-marco-MiniLM-L6-v2`（6 层），检查点层 {1, 3}
- 评分方式：rank-based（分数→排名→N−rank），保证跨方法可比
- 指标：nDCG@10、MRR、p50/p95/p99 延迟、layers_eq（层当量）、top-10 agreement vs full-depth
- 前置正确性验证：`cpu_cast/active_set.py` 的逐层执行器与 HuggingFace 完整前向**逐位一致**（max_abs_diff = 0.00e+00，见 `scripts/test_active_set.py`）

## 1. 结果总表

| 方法 | keep | nDCG@10 | MRR | p50 ms | p95 ms | p99 ms | layers_eq | agree@10 | 加速比 |
|---|---|---|---|---|---|---|---|---|---|
| full_depth | — | 0.7038 | 0.6827 | 2872.5 | 2970.5 | 3004.5 | 6.00 | 1.000 | 1.00× |
| tinybert_only | — | 0.6960 | 0.6812 | 179.7 | 203.3 | 206.6 | — | 0.656 | 15.98× |
| cascade_shallow_p50 | 0.50 | **0.7038** | 0.6824 | 1447.7 | 1510.3 | 1539.0 | 3.25 | **0.992** | 1.98× |
| cascade_shallow_p25 | 0.25 | 0.7002 | 0.6808 | 789.9 | 837.2 | 853.0 | 1.88 | 0.911 | 3.64× |
| see_token_p50 | 0.50 | 0.6985 | 0.6810 | 1314.1 | 1366.6 | 1390.1 | 3.25 | 0.894 | 2.19× |
| see_token_p25 | 0.25 | 0.6881 | 0.6690 | 642.7 | 686.9 | 876.0 | 1.88 | 0.673 | 4.47× |
| score_cascade_p50 | 0.50 | 0.3996 | 0.4105 | 1574.0 | 1648.4 | 1692.3 | 3.50 | 0.261 | 1.82× |
| score_cascade_p25 | 0.25 | 0.2394 | 0.2663 | 1206.6 | 1264.2 | 1288.6 | 2.62 | 0.175 | 2.38× |
| budget_alloc_p50 | 0.50 | 0.4031 | 0.4130 | 1831.4 | 1891.8 | 1905.4 | 4.00 | 0.271 | 1.57× |
| budget_alloc_p25 | 0.25 | 0.2493 | 0.2662 | 1354.5 | 1398.8 | 1408.0 | 3.00 | 0.186 | 2.12× |
| prism_cluster_cv0.1 | — | 0.7038 | 0.6827 | 2732.6 | 2841.2 | 2900.0 | 6.00 | 1.000 | 1.05× |

方法说明（忠实性标注）：
- `cascade_shallow`：TinyBERT-L2 对全部候选打分 → 保留 top-p → L6 精算幸存者。**真实可部署基线，无代理成分。**
- `see_token`：embedding 层 query–doc token max-sim 过滤（SEE 的简化推理级复现）。
- `score_cascade` / `budget_alloc`：中间检查点用**未训练**的 pooler+classifier 打分（代理，代表下界，不是 GAINER 的 trained gain ranker）。
- `prism_cluster`：CV 触发的一维 KMeans（k=3）丢弃最低簇——**简化 reimplementation，非官方代码**。

## 2. 关键发现

**F1 — 浅层优先级联是最强基线，占据大部分收益空间。**
cascade_shallow_p50 达到与 full_depth 完全相同的 nDCG@10（0.7038）、99.2% top-10 一致性，p50 加速 1.98×；p25 档 3.64× 加速仅损失 0.0036 nDCG。任何新方法（包括 CPU-CAST）必须先超过这条线。

**F2 — 未训练的中间打分代理灾难性失败，验证了 GAINER 的核心动机。**
score_cascade / budget_alloc 的 nDCG 只有 0.24–0.40（对比 0.70），说明"中间层隐藏状态 + 未训练头"完全无法预测最终相关性。这正面支持 GAINER 论文"必须有训练过的 gain ranker"的论点。**含义：我们的 GAINER 对比必须使用其官方发布的训练好的 ranker / registered runs，不能用代理代替，否则审稿人会直接否决。**

**F3 — SEE 式 token 过滤有效但被同 keep 率的浅层级联支配。**
see_token_p50（0.6985 / 2.19×）对比 cascade_shallow_p50（0.7038 / 1.98×）：质量更低、速度未更快；p25 档差距更大（agree 0.673 vs 0.911）。SEE 作为独立预过滤基线在我们的设置下不构成最强对照。

**F4 — Prism 简化版在 CV=0.1 下从未触发。**
prism_cluster_cv0.1 与 full_depth 完全一致（100/100 query 无一触发聚类剪枝），仅有的 1.05× 差异来自 KMeans 开销。CV 阈值未经过校准，**不能据此下任何结论**；需要先在 dev query 上扫描 CV 阈值分布，再重测。

**F5 — 2 线程 CPU 上逐检查点开销是真实的。**
score_cascade（layers_eq 3.50）比 cascade_shallow（3.25）**更慢**（1574 vs 1448 ms）；score_cascade_p25（2.62 层当量）也比 cascade_shallow_p25（1.88）慢。原因：每个检查点的 score_hidden + keep_fn + 索引操作在低核数 CPU 上是固定开销，小 batch 无法摊薄。**对 CPU-CAST 设计的直接约束：检查点必须少（1–2 个）且便宜；频繁的逐层决策在 CPU 上是负收益。这与 Gate A 的 compaction 结论（f≤0.75 时压缩有 20–96% 收益）共同构成 CPU-CAST 的设计空间。**

## 3. 对研究方向的含义 / Gate B 风险

- **Gate B 当前状态：未通过，也未失败——最强基线（浅层级联）尚未被任何动态方法超过。** 在 MiniLM-L6 + scifact + top-100 设定下，"动态候选淘汰"类方法（SEE、未训练代理、未校准 Prism）全部不优于静态浅层级联。这与计划第 16 节的预案一致：如果 GAINER/Prism 简化实现已给出同等效果且无可解释差异，就停止把"动态候选淘汰"本身作为主创新。
- CPU-CAST 的差异化空间收窄为计划中的核心命题：**不规则 active set 在低核数 CPU 上的执行组织**（compaction 时机、跨查询 wavefront、measured-cost 调度），而不是新的淘汰算法。Gate A 已证明执行层 gap 存在（f≤0.75 时 C_compact < C_keep，+20–96%）；现在的任务是在 F1 的强基线之上证明执行层收益能转化为端到端 p50/p95 改善。
- 必须把 cascade_shallow 加入所有后续实验的必跑基线集合。

## 4. 数据与复现

```bash
# 正确性验证（逐位一致）
OMP_NUM_THREADS=2 python scripts/test_active_set.py

# Stage 2 基线矩阵（~30 分钟，2 线程）
OMP_NUM_THREADS=2 python scripts/bench_stage2_baselines.py \
  --dataset scifact --run data/candidate_runs/scifact/rrf_top100.jsonl \
  --num-queries 100 --out artifacts/runs/bench_stage2.jsonl
```

原始记录：`artifacts/runs/bench_stage2.jsonl`（11 条记录，含 per-query 明细）。

## 5. 已知限制

- 4 vCPU 配额 ≠ 计划目标 2 vCPU；所有结论需在目标硬件复验。
- 100 query 子集、单一数据集（scifact）；nfcorpus 管线已就绪未跑。
- 无 perf 计数器、无能耗测量。
- GAINER 对比使用未训练代理（下界），Prism 为简化 reimplementation 且阈值未校准——两者都不能作为最终对比结论。
