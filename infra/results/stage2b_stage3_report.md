# CPU-CAST Stage 2b + Stage 3：GAINER 真实对照、Prism 校准、compaction 间隔网格

日期：2026-11-06　环境：KVM，4 vCPU 配额，2 线程（OMP_NUM_THREADS=2）；scifact 100 test query，RRF top-100，MiniLM-L6，max_length 256

## 1. GAINER 真实对照（Gate B 要求）

按官方 `GAINER_v3.ipynb` 配方实现（`scripts/build_gainer_train_data.py` → `scripts/train_gainer_heads.py` → `scripts/bench_stage2b_gainer.py`）：

- **Leave-one-dataset-out**：exit heads / gain rankers 在 nfcorpus train（747 query，BM25 top-50 pool，27,624 对）训练，在 scifact 测试。GAINER 原文用 6 个 BEIR 数据集训练；我们只有 nfcorpus 本地可用（规模缩减，见限制）。
- **Exit heads**：MLP(384→160→1)，GELU+dropout0.1，AdamW lr 2e-3 wd 1e-4，smooth-L1 回归最终分数，batch 512，≤25 epoch，patience 5——与原配方一致。验证损失：checkpoint-1 为 2.12（第 2 层 CLS 信号弱），checkpoint-2 为 0.598。
- **Gain rankers**：每转移一个成对 ranker，训练目标为反事实 nDCG@10 效用（该候选前进到最终分数时的 nDCG 增量），从 current-rank rollout（5 档 survival）收集状态（125k/69k）。特征为简化集 [当前分, 秩, z, 前一分] + CLS（原论文用更长标量特征表）。
- **检查点**：第 2、4 层后（6 层模型的 1/3、2/3，对应 GAINER 12 层模型的 4、8）。
- 配额语义：每个检查点保留 top-r 比例的**活跃**候选（仿真与实测一致）。

### 结果（质量为 GAINER 式仿真 rollout；延迟为真实条件执行）

| 方法 | nDCG@10 | p50 ms | layers_eq | agree@10 |
|---|---|---|---|---|
| full_depth | 0.7038 | 2760 | 6.00 | 1.000 |
| current_rank_r50（GAINER 基线） | 0.5656 | 1608 | 3.50 | 0.631 |
| current_rank_r25 | 0.4142 | 1218 | 2.62 | 0.341 |
| gainer_gain_r50（我们的 gain ranker） | 0.5232 | 1607 | 3.50 | 0.508 |
| gainer_gain_r25 | 0.3568 | 1221 | 2.62 | 0.304 |
| oracle_r50（仿真上界） | 0.8107 | — | 3.50 | 0.431 |
| oracle_r25（仿真上界） | 0.8556 | — | 2.62 | 0.279 |

### 发现

**F6 — current_rank 基线复现成功。** 训练过的 exit head + 配额分配在 r50 得 0.5656，与 GAINER 注册结果中 CurrentRank 在 minilm12/scifact 的 0.539 同量级（设置不同：我们的 pool 是 RRF top-100、max_len 256、6 层模型）。相比 Stage 2 的未训练代理（0.40），训练 head 带来大幅改善——再次确认 F2。

**F7 — 我们的降配版 gain ranker 未能复现 GAINER 的优势。** gainer_gain（0.523）不优于 current_rank（0.566），而 GAINER 论文报告 0.677 vs 0.539。可能原因：(a) 训练数据只有 1 个数据集（原文 6 个）；(b) 特征集缩减；(c) 单种子；(d) 域偏移。**结论：GAINER 对比必须引用其注册结果并注明我们的复现为下界；不能声称 GAINER 方法无效。**

**F8 — Oracle 显示巨大策略空间（本项目最重要的正面信号）。** 同样 3.5 层当量下，oracle 选择达 0.811，比 current_rank 高 0.245；甚至高于 full-depth（0.704，混合评分效应：被弃候选用 exit-head 分数替代后排序噪声反而降低）。这说明在相同计算预算下，**选择策略的质量决定性远大于执行开销**——正是 CPU-CAST 调度器可以争取的空间。注意 oracle 用了未来信息且 agree@10 仅 0.43，其"超过 teacher"来自混合评分伪影，不能作为可达结果引用。

## 2. Prism CV 阈值校准（解决 F4）

`scripts/calibrate_prism_cv.py`（缓存 checkpoint-1 分数）+ `scripts/sweep_prism_stats.py`（离线精确仿真 + 实测复验）。Prism 基线仍为我们的 reimplementation（CV 触发 1-D KMeans k=3，丢弃最低簇）。

**F9 — 原始 logit 的 CV 触发器结构性失活。** 150 个 train query 上，checkpoint-1 分数的 CV 中位 0.013、p90 0.0147，且分布极窄（p10–p90 仅 0.012–0.015）——任何固定阈值要么永不触发、要么几乎总是触发。这解释了 Stage 2 中 cv0.1 从不触发的原因：CV 统计量未对分数尺度归一化。

**F10 — 校准后的触发器仍然强烈不划算。** 换用尺度稳健统计量（minmax_cv、gap_ratio）后触发器可以正常工作，但收益/代价严重失衡：

| 配置（train 仿真） | trigger | nDCG@10 | Δ vs full | layers_eq |
|---|---|---|---|---|
| full（train） | — | 0.7526 | — | 6.0 |
| raw_cv=0.014 | 20% | 0.7016 | −0.051 | 5.76 |
| gap_ratio=0.3 | 17% | 0.6975 | −0.055 | 5.76 |
| minmax_cv=0.4 | 43% | 0.6562 | −0.096 | 5.47 |

测试集实测（前两档）：raw_cv=0.014 → 0.6717（−0.032），gap_ratio=0.3 → 0.6659（−0.038），均只省 ~5% 层当量。**丢弃簇的候选多为真实相关文档**（未训练的中间层分数无法区分），质量损失是计算节省的 6–10 倍（按每层当量换算）。结论：在我们设置下，Prism 式簇剪枝不是有竞争力的基线；若引用 Prism，必须说明其原始设计针对不同模型/特征，且我们的 reimplementation 为下界。

## 3. Stage 3：compaction 间隔网格（Gate A 的全运行级验证）

`scripts/bench_stage3_interval.py`。固定退出模式（检查点在第 2/4 层后，r=每档保留比例，exit-head 打分），比较每 m 层尝试一次压缩（m=∞ 为从不压缩，始终跑满 batch）。质量在所有 m 下完全一致（压缩不改变结果），只比较墙钟：

| 配置 | m=1（每层） | m=2 | m=3 | 从不 |
|---|---|---|---|---|
| r50 p50 ms | **1634** | 2261 | 2575 | 2829 |
| r50 vs 从不 | −42% | −20% | −9% | — |
| r25 p50 ms | **1224** | 2023 | 2432 | 2770 |
| r25 vs 从不 | −56% | −27% | −12% | — |

**F11 — 逐层压缩在真实退出模式下强烈占优。** 与计划中"每层压缩可能过于频繁"的假设相反：在 2 线程 CPU、batch 100、seq 256 的形状上，gather/scatter 开销相对死行浪费的计算可忽略，m=1 比 m=2 再快 28–40%。Gate A 的微观基准结论（f≤0.75 时压缩有净收益）在全运行层面成立，且**拐点非常尖锐：压缩频率只要低于每层一次，收益就大部分丢失**。设计含义：CPU-CAST runtime 应默认逐层压缩，无需保守的间隔参数；节省的调优空间转向检查点数量与位置（F5：检查点要少而便宜）。

## 4. 对 Gate B 的更新结论

- 最强基线仍是**浅层级联**（TinyBERT→L6：0.7038 @ 1448ms，Stage 2 F1）。
- GAINER 降配复现（current_rank 0.566 / gainer_gain 0.523 @ ~1600ms）与 Prism 校准后复现（≥0.666 @ ~2600ms）都**不敌浅层级联**。
- 但 oracle 上界（同预算 0.81+）表明选择策略空间巨大。Gate B 的路径：不是发明新的淘汰算法，而是**用 CPU 测得成本驱动的执行组织和更好的选择信号去逼近 oracle**——这正是 CPU-CAST 的核心命题。
- 下一步（按计划 Stage 3 剩余 + Stage 4）：完整 cost grid（active rows × seq × batch × threads）、INT8 break-even 复核，然后进入 CPU-CAST runtime 与跨查询 wavefront 调度器原型。

## 5. 复现命令

```bash
# 1. GAINER 训练数据（nfcorpus train，BM25 top-50，~11 min）
OMP_NUM_THREADS=2 python scripts/build_gainer_train_data.py
# 2. 训练 exit heads + gain rankers（~2 min）
OMP_NUM_THREADS=2 python scripts/train_gainer_heads.py
# 3. GAINER 评测（~18 min）
OMP_NUM_THREADS=2 python scripts/bench_stage2b_gainer.py
# 4. Prism 校准（缓存分数 ~8 min；离线扫参 + 实测 ~10 min）
OMP_NUM_THREADS=2 python scripts/calibrate_prism_cv.py --cache-only
OMP_NUM_THREADS=2 python scripts/sweep_prism_stats.py
# 5. compaction 间隔网格（~35 min）
OMP_NUM_THREADS=2 python scripts/bench_stage3_interval.py
```

原始记录：`artifacts/runs/bench_stage2b_gainer.jsonl`、`artifacts/runs/prism_cv_calibration.jsonl`、`artifacts/runs/bench_stage3_interval.jsonl`。

## 6. 已知限制

- GAINER 复现为降配版（1 个训练数据集 vs 原文 6 个、简化特征、单种子）；其注册结果（ettin68m/minilm12，GPU）不可与我们的 CPU 数字直接比较。
- Prism 为 reimplementation，触发统计量的原始形式在论文中可能不同；我们的结论只适用于本实现。
- oracle 使用未来信息，仅作上界；其"超过 full-depth"来自混合评分伪影。
- 4 vCPU 配额 ≠ 目标 2 vCPU；100 query 子集；单数据集；无 perf/能耗。
- 训练好的 exit head/gain ranker 仅在 nfcorpus train 上训练，未在 scifact 上调参（保持 leave-one-dataset-out）。
