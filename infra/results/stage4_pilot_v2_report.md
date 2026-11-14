# Stage 4 Pilot v2 — 交接说明（handoff）

**分支**: `vibe/fix-serial-scheduler-bdc178` (PR #2) · **HEAD**: `e0a4fac` + 本次提交
**日期**: 2026-11-14 · **审查来源**: 用户审查（问题 A/B + 协议问题）
**状态**: 测量修复已完成并通过奇偶校验；pilot v2 已完成两轮（500ms/1000ms），在线跨 Query parity 通过

## 1. 为什么之前"运行总是完不成"

- 沙箱在会话之间被重置：`/tmp` 日志、本地仓库（回退到 `8d45d28`）、Python 环境与模型全部丢失。
- 恢复方式（可复现）：
  1. `git fetch origin && git checkout vibe/fix-serial-scheduler-bdc178`（远端分支完好，HEAD=`e0a4fac`）。
  2. 依赖按 `manifests/python_env.txt` 钉住版本重装：`torch==2.15.0+cpu`（`--index-url https://download.pytorch.org/whl/cpu`）、`transformers==5.22.1`、`sentence-transformers==6.2.0`、`numpy==2.5.4`、`pyarrow==26.0.1`、`rank_bm25`、`pytrec-eval-terrier`、`beir==2.2.0`。
  3. `python3 scripts/download_models.py`（钉住 HF revision，重建后恢复 `manifests/models_manifest.json` 以消除时间戳漂移）。
  4. 重建 scifact（只需此数据集）：BEIR 官方 parquet → `data/scifact/`，然后 `python3 scripts/build_candidate_runs.py --datasets scifact`。重建后 RRF Recall@100=0.9469，与冻结值一致；pilot 的 SHA-256 校验通过，证明字节级可复现。
- 另一个中断原因（已修复）：启动第二个负载点时 `--out` 文件名被换行截断，1000ms 记录写入了 `artifacts/runs/stage4_pilot`；已合并回 `stage4_pilot_v2.jsonl` 并删除杂散文件。

## 2. 本轮已完成的审查修复（commit `e0a4fac`）

- **问题 B（测量定义）**：`cpu_cast/scheduler.py` 统一 4 点时间轴 `arrival → submit → activation(embedding 完成) → first_exec(首个层事件) → finish`；Wavefront 的 `submit_time` 移到 embed 之前，`first_exec_time` 由首个层事件的绝对时钟设置；Serial 的 `first_exec_time = activation_time`。所有层事件（含最终评分事件）记录绝对 `start_ms`/`end_ms`。
- **问题 A（真正跨 Query parity）**：`scripts/pilot_stage4_paired.py` 新增 `online_cross_query_rank()` —— 用真实 MiniLM、冻结 qid、交错到达跑真正的在线 `WavefrontScheduler.run(reqs, online=True)`，逐 Query 逐候选对比 per-query reference 的 `exit_layer`/分数，报告决策幅度（checkpoint 处距阈值最小距离）与翻转行 margin。
- **协议修复**：pilot 用独立种子（协议种子+1=20261110）从 100 个冻结 qid 随机抽 20 个，**到达顺序=抽样顺序**（不排序）；`load_frozen()` 同时校验 candidate run 与 qrels 的 SHA-256；p99 按冻结门槛不报告（`p99_reported: false` + 原因字段）。
- 冻结文件 `manifests/stage4_protocol.json` **未改动**（遵守"不修改冻结文件"指示）。

## 3. Pilot v2 结果（`artifacts/runs/stage4_pilot_v2.jsonl`，16 条记录）

环境：4 vCPU / 2 torch 线程 / MiniLM-L-6-v2 / 20 qid（seed 20261110，随机到达顺序）/ `git_dirty=false` @ `e0a4fac`。

### 在线跨 Query parity（核心新证据，两个负载点一致）

- `exit_mismatches = 0`（在线 Wavefront 的退出决策与 per-query reference 完全一致）
- `max_score_diff = 9.5e-07`（远小于 1e-4 容差）
- 决策幅度：checkpoint 1 最小 margin = 3e-05，checkpoint 3 = 4.8e-05；`flip_margins = []`
- 结论：**跨 Query 大 batch 合并不改变真实 MiniLM 的退出决策**。延迟差异可解释为"相同计算决策下的 scheduler 差异"。

### 4 点时间轴分解（每值 2 reps 平均，单位 ms）

| stagger | scheduler | p50 | p95 | T_activation(mean) | T_ready_queue(mean) | rows_executed | nDCG@10 |
|---|---|---|---|---|---|---|---|
| 500 | serial | 11048 | 23637 | 10137 | 0 | 7499 | 0.4141 |
| 500 | wavefront | 31311 | 34990 | 600 | 2697 | 7038 | 0.4141 |
| 1000 | serial | 6600 | 15004 | 5620 | 0 | 7499 | 0.4141 |
| 1000 | wavefront | 23458 | 29414 | 544 | 1864 | 7038 | 0.4141 |

- Serial 的瓶颈在 **T_activation**（单 worker 逐个 embed，后面的 query 排队等 embedding）。
- Wavefront 的 embedding 很快（T_activation≈0.5–0.7s），但 **T_ready_queue≈1.9–2.7s**：候选激活后要等跨 Query 的层事件轮次。
- 两者 nDCG@10 / MRR 完全一致（0.4141 / 0.4087），rows_executed 差异（7038 vs 7499）来自跨 Query 池化后的退出时点差异，质量无损失。
- **Wavefront 在两个高负载点 p50 慢 2.8–3.5×**，与 pilot v1 结论一致，现在有了正确测量定义和 parity 背书。

### 已知缺口（本次提交修复）

- pilot 此前只持久化了 layer event **数量**（39），未写出完整 per-event trace。本次提交在 `online_cross_query_parity` 记录中新增 `layer_event_trace`（event_id/layer/batch_rows/scored_rows/compacted/n_queries/gather/forward/scatter/start_ms/end_ms）。**需重跑一次 pilot 生成含 trace 的证据**（结果预期与本次一致，种子确定性）。

## 4. 下一步（按用户指示，勿越界）

1. ✅ 测量修复 + 在线跨 Query parity（本提交）
2. ⏳ 用修复后的脚本重跑 pilot v2（500ms/1000ms，同种子），获得含层事件 trace 的完整证据 → 提交
3. ⏸ **不启动**正式全量复跑；若 parity 保持且 Wavefront 仍慢，记录 H2 负面结果
4. ⏸ D003 (GAINER) / H1 线**暂不开始**
5. 待办记录：正式协议 v2 应保留 `load_qids` 随机排列顺序（现文件为 sorted()，pilot 用独立种子抽样绕过）；p99 门槛保持 ≥10,000 完成数（当前规模只报 p50/p90/p95）

## 5. 复现命令

```bash
python3 scripts/test_scheduler_toy.py          # toy parity（CI 同款）
python3 scripts/test_serial_scheduler.py       # 真实模型 parity
python3 scripts/pilot_stage4_paired.py --reps 2 --stagger-ms 500  --out artifacts/runs/stage4_pilot_v2.jsonl
python3 scripts/pilot_stage4_paired.py --reps 2 --stagger-ms 1000 --out artifacts/runs/stage4_pilot_v2.jsonl
```
