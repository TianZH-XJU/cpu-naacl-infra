# Stage 4 Pilot v2 — 交接说明（handoff）

**分支**: `vibe/fix-serial-scheduler-bdc178` (PR #2) · **HEAD**: `c8651b6` + 证据提交
**日期**: 2026-11-14 · **审查来源**: 用户审查（问题 A/B + 协议问题）
**状态**: ✅ 测量修复完成、✅ 在线跨 Query parity 通过、✅ pilot v2 两轮完成（含层事件 trace）——**正式全量复跑仍按指示未启动**

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

## 3. Pilot v2 最终结果（`artifacts/runs/stage4_pilot_v2.jsonl`，16 条记录，含 `layer_event_trace`）

环境：4 vCPU / 2 torch 线程 / MiniLM-L-6-v2 / 20 qid（seed 20261110，随机到达顺序）/ `git_dirty=false` @ `c8651b6`。两轮运行（修复 trace 前/后）parity 与质量结果逐位一致，延迟在噪声范围内。

### 在线跨 Query parity（核心新证据，两个负载点一致）

- `exit_mismatches = 0`（在线 Wavefront 的退出决策与 per-query reference 完全一致）
- `max_score_diff = 9.5e-07`（远小于 1e-4 容差）
- 决策幅度：checkpoint 1 最小 margin = 3e-05，checkpoint 3 = 4.8e-05；`flip_margins = []`
- 结论：**跨 Query 大 batch 合并不改变真实 MiniLM 的退出决策**。延迟差异可解释为"相同计算决策下的 scheduler 差异"。

### 4 点时间轴分解（最终轮，单位 ms）

| stagger | scheduler | p50 | p95 | T_activation(mean) | T_ready_queue(mean) | rows_executed | nDCG@10 |
|---|---|---|---|---|---|---|---|
| 500 | serial | 9917–10756 | 21950–22418 | 9233–9781 | 0 | 7499 | 0.4141 |
| 500 | wavefront | 29932–30125 | 33710–33905 | 634–689 | 2513–2538 | 7038 | 0.4141 |
| 1000 | serial | 5666–5837 | 13341–13544 | 4812–4926 | 0 | 7499 | 0.4141 |
| 1000 | wavefront | 20759–21764 | 28744–30314 | 522–545 | 1233–1239 | 7038 | 0.4141 |

- Serial 的瓶颈在 **T_activation**（单 worker 逐个 embed，后面的 query 排队等 embedding）。
- Wavefront 的 embedding 很快（T_activation≈0.5–0.7s），但 **T_ready_queue≈1.2–2.5s**：候选激活后要等跨 Query 的层事件轮次。
- 两者 nDCG@10 / MRR 完全一致（0.4141 / 0.4087），质量无损失；rows_executed 7038 vs 7499 来自跨 Query 池化后的退出时点差异。
- **Wavefront 在两个高负载点 p50 慢 2.8–3.7×**，与 pilot v1 结论一致，现在有正确测量定义 + parity 背书。

### 层事件 trace（Wavefront 慢的机制定位）

39 个层事件，总 forward 时间 33.0s（makespan ≈40s 的 83%）：

| layer | 事件数 | 平均参与 query 数 | 平均 batch rows | forward 均值/最大 (ms) |
|---|---|---|---|---|
| 0 | 10 | 2.7 | 200 | 940 / 1240 |
| 1 | 9 | 3.0 | 222 | 1034 / 1208 |
| 2 | 6 | 4.2 | 176 | 827 / 1239 |
| 3 | 6 | 4.2 | 176 | 816 / 1211 |
| 4 | 4 | 5.8 | 115 | 545 / 1208 |
| 5 | 4 | 5.8 | 115 | 554 / 1271 |

- 30/39 事件触发了 active-set compaction，但 batch 仍达 115–222 行（max_batch=256 封顶）。
- **在 2 线程 CPU 上，~200 行的大 batch 每 event forward 要 0.8–1.2s**；一个候选要走到 layer 3 退出就要串行等待多个全局轮次 → 队头阻塞。Serial 用查询内 compaction 的小 batch（立即执行、不等全局层对齐），反而每行更便宜。
- 这直接支持用户假设：**低核 CPU 上跨 Query 规则化大 batch 不如查询内 active-set compaction**。H2 的可研究方向应从"跨 Query wavefront"转向 CPU-aware compaction / 批大小控制 / 层内细粒度调度。

## 4. 下一步（按用户指示，勿越界）

1. ✅ 测量修复 + 在线跨 Query parity（`e0a4fac`）
2. ✅ 层事件 trace 持久化（`c8651b6`）+ pilot v2 重跑（本证据提交，`git_dirty=false`）
3. ⏸ **不启动**正式全量复跑——parity 通过且 Wavefront 仍显著更慢；等待用户决定：记录 H2 负面结果，或先探索 CPU-aware compaction / batch 控制变体
4. ⏸ D003 (GAINER) / H1 线**暂不开始**
5. 待办记录：正式协议 v2 应保留 `load_qids` 随机排列顺序（现文件为 sorted()，pilot 用独立种子抽样绕过）；p99 门槛保持 ≥10,000 完成数（当前规模只报 p50/p90/p95）

## 5. 复现命令

```bash
python3 scripts/test_scheduler_toy.py          # toy parity（CI 同款）
python3 scripts/test_serial_scheduler.py       # 真实模型 parity
python3 scripts/pilot_stage4_paired.py --reps 2 --stagger-ms 500  --out artifacts/runs/stage4_pilot_v2.jsonl
python3 scripts/pilot_stage4_paired.py --reps 2 --stagger-ms 1000 --out artifacts/runs/stage4_pilot_v2.jsonl
```
