# CPU-CAST Stage 4 报告（v5 权威版）— 第二轮审稿（899e580d）响应

日期：2026-11-09 ｜ 分支：`vibe/cpu-cast-stage1-infra-ef8e86`
权威数据：`artifacts/runs/bench_stage4_load_v5.jsonl`（run_id `fe7d8aae`，git_sha `549fa7d8e782`，git_dirty=false）
环境：KVM VM，4 vCPU quota，~3.95 GiB cgroup；全程 `OMP_NUM_THREADS=2`，串行执行。

## 0. 本轮相对 v3 的关键变化

第二轮审稿（899e580d）要求的前五项修复全部完成，并且**在权威运行前发现了两个会使 v4
数据全部作废的回归**——通过既有测试套件在 launch 前捕获：

| # | 问题 | 发现方式 | 修复 |
|---|---|---|---|
| 1 | `WavefrontScheduler` scatter 重构时丢失 `r.hidden.index_copy(...)` 回写——每层都会在原始 embedding 上重复执行 | Test A 失败（max_abs_diff=10.6） | 恢复回写 + 注释护栏（commit bedad5b） |
| 2 | 事件 trace 的 `request_ids` 对 dict 直接解包（测试用 2 字符 qid 侥幸通过，真实 qid 崩溃） | v4 首次运行崩溃 | `list(by_req.keys())`（369e403） |
| 3 | 时间约定不一致：`WavefrontScheduler` 记录相对时间、`SerialScheduler` 记录绝对时间，bench 对部分字段重复减 `t0` → 串行在线延迟为负（-24825s 量级） | v4 数据审计 | 统一为绝对 `perf_counter`，bench 统一减 `t0`（dfcbc55） |
| 4 | 多阶段运行 OOM（anon-rss 4.1GB > 3.95GiB cgroup；glibc 不及时归还内存） | dmesg oom-kill 记录 | 阶段间 `gc.collect()+malloc_trim` + `MALLOC_ARENA_MAX=2`（dfcbc55） |

废弃运行留档（不删除，含失败原因）：`bench_stage4_load_v4_partial_crashed.jsonl`（bug #2）、
`bench_stage4_load_v4_partial_oom.jsonl`（bug #3/#4，串行在线延迟无效）。

v5 记录的 `version` 字段仍为 `v4`（脚本内常量未随版本递增，已在 dfcbc55 后修正为 `v5`）；
**以 `git_sha` 字段为权威溯源依据**。

## 1. P0-1/P0-2：统一成本账本

- **gather 成本曲面**：`cost_grid.jsonl` 新增 30 条 `gather2` 记录，测量
  C_gather(N_src, K)（N_src∈{16,50,100,200}, K/N∈{0.05..1.0}），
  `CostTable.gather_ms(n_from, n_keep)` 改为双变量插值。
- **embedding 计入 wavefront 仿真**：`simulate_wavefront` 在请求到达时计入
  `cost.embed_ms(n)`（v3 遗漏，约 30×31.1ms≈933ms）。
- **串行 gather 参数修正**：`service_time_serial` 现以该层执行前的源行数
  （而非执行后活跃数）调用 gather。
- **总量校准（v5 实测）**：隔离执行 46.4/47.2/47.3s vs 仿真 47.8s（偏差 1.5–3.0%）。

已知剩余近似（诚实声明）：exit_head 代理仍用 GAINER MLP 延迟代表 pooler 路径
（量级相近但非同一计算图）；gather 表未覆盖多请求 `torch.cat`/`index_copy` 的
组合成本曲面。见 §6 开放问题。

## 2. P0-3：真实串行在线调度器（核心新结果）

v3 的 "serial online" 是 measured_service_replay（离线测得服务时间后 Python 重放）。
v5 新增 `SerialScheduler`（`cpu_cast/scheduler.py`）：真实单_worker FIFO 循环，
sleep-until-arrival、逐请求完成驱动，与 wavefront 使用**完全相同的确定性到达轨迹**
（30 请求，间隔 2000ms，span 58s）和相同的退出策略（固定阈值，checkpoint 1/3）。

**同一到达轨迹、同为真实执行、各 3 次重复（ms）：**

| 指标 | Serial online（真实） | Wavefront online（真实） | 变化 |
|---|---|---|---|
| p50 | 15215 / 15689 / 15484 | 1669 / 1745 / 1765 | **−89%** |
| p95 | 26471 / 26455 / 25759 | 5397 / 4796 / 5371 | **−80%** |
| p99 | 27476 / 27386 / 26632 | 5501 / 5555 / 5665 | **−80%** |
| makespan | 85.7 / 85.6 / 84.8 s | 59.1 / 59.1 / 59.2 s | −31% |

**这推翻了 v3 的核心结论。** v3 报告的"wavefront p99 恶化 +105~120%"是
measured_service_replay 模型的伪影：该模型用隔离测量的服务时间重放，无法再现
真实 FIFO 队列在到达率超过服务率时的排队积累（服务 ~2.4s > 到达间隔 2s，
积压线性增长，第 k 个请求等待 ≈ (k−1)×0.4s）。真实串行在线调度器的尾延迟
由排队主导，wavefront 通过跨查询组批同时改善中位与尾部延迟。

**H3 修正后的表述**：在真实在线服务条件下（本负载剖面），跨查询 wavefront 调度
同时改善 p50/p95/p99；"尾延迟恶化"仅在离线 burst 仿真中出现
（sim burst: wavefront p50 47661 vs serial 25928——离线一次性提交时组批等待
拖慢所有请求），不适用于到达驱动的在线场景。

**仿真仍不能替代实测**：sim serial staggered_light p50=1741ms vs 真实 15215ms——
成本模型按隔离执行校准（隔离实测 46.4s ≈ 仿真 47.8s），但真实串行在线在持续
负载下服务时间退化约 50%（85.7s vs 理论 61.5s；疑似缓存/内存带宽效应）。
成本模型能校准隔离服务时间，不能预测排队系统行为——这正是第二轮审稿
"成本模型与负载实验未形成可信闭环"的正面回应：现在闭环由真实调度器闭合，
仿真仅作趋势参考。

## 3. 请求级 trace 与会计校验（P0-5/P1-4）

每条 `real_*_online` 记录含逐请求 `{qid, arrival_ms, submit_ms, first_exec/first_layer_ms,
finish_ms, latency_ms, embed_ms, layer_log[]}` 与事件级 `layer_events[]`
（event_id、参与请求、逐请求/整批行数、gather/forward/scatter 分项计时）。

会计恒等式验证（rep0，全部通过）：
- serial：`finish − arrival = (submit − arrival) + (finish − submit)`，逐请求误差 < 1ms
- wavefront：`latency_ms == finish_ms − arrival_ms`（误差 < 2ms）
- `submit_ms ≥ arrival_ms` 全部成立（完成驱动提交，无提前提交）

## 4. P1-3/P1-5/P1-6：调度器健壮性

- 长度桶检查提前到 `run()` 入口（`_check_lengths`），online 模式在**任何请求到达前**
  拒绝不兼容混合（Test F 新增；Test E 覆盖 offline）。
- 7 项测试全过：A 全深度（diff 0.00e+00）、B exit-head、C pooler 路径
  （Stage 4 实际路径，diff 4.77e-07）、D max_batch 溢出、E/F 长度桶。
- `keep_frac` 语义澄清：无 `keep_fn` 时为**全局 wavefront 预算**（跨查询 top-k，
  有意的全局资源分配）；Stage 4 实验使用逐层固定阈值（`keep_fn`），与查询组成无关。

## 5. 强基线与质量（P1-7，诚实结论）

同 30 查询、同候选集、同计时边界：

| 方法 | nDCG@10 | 延迟 |
|---|---|---|
| cascade_shallow（TinyBERT→L6, keep 50） | **0.6805** | p50 1481ms，总 44.7s |
| active-set 固定阈值（当前 CPU-CAST 策略） | 0.3702 | 见 §2 |
| full-depth L6（参考，100-query 评测） | 0.7038 | — |

**当前 active-set 策略落后 cascade_shallow 0.31 nDCG，这一差距是论文层面
的核心未决问题。** v5 证明的是：在给定策略下，wavefront 调度相对真实串行
调度有大幅延迟优势；策略本身的质量缺口需在 Stage 5 通过更好的退出信号
（而非调度器）解决。在质量缺口闭合之前，不得声称 CPU-CAST 优于最强基线。

## 6. 开放问题（不因本轮关闭）

1. **质量缺口**（Q9）：exit 信号质量是瓶颈；GAINER query-level split 后
   gain ranker 0.4454 vs current-rank 0.5775，未超过平凡基线。
2. **规模与普遍性**（Q16）：仍为单数据集（SciFact）、30 查询、单一长度桶、
   单一到达率（2s）；无重复种子 CI；`configs/workloads.yaml` 中的多长度桶/
   Poisson/CPU pressure 未执行。
3. **ORT 后端**（Q10）：wavefront 仍为 Torch 专用；ORT 结果限于整模型前向。
4. **成本账本残差**：exit_head 代理、多请求 gather 组合成本、负载下服务时间
   退化机制（§2 观察，未解释）。
5. **负载下服务时间退化**：真实串行在线服务 ~2.4s/请求 vs 隔离 1.55s（+50%），
   机制未查明（疑似缓存/频率），已如实报告。

## 7. 结论状态

- ✅ 可声明：真实在线条件下（本剖面），wavefront 相对真实串行 FIFO 调度
  p50 −89% / p99 −80%，且会计恒等式可从原始 trace 独立复算。
- ✅ 可声明：v3 的"尾延迟恶化"结论为仿真伪影，已撤回。
- ❌ 不可声明：CPU-CAST 优于 cascade_shallow（质量缺口 0.31 nDCG 未闭合）。
- ❌ 不可声明：H3 已被普遍验证（单数据集、单到达率、30 请求）。
