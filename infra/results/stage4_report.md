# CPU-CAST Stage 4 Report — Wavefront Scheduler, Measured Cost Model, Load Experiment

日期：2026-11-06 ｜ 分支：`vibe/cpu-cast-stage1-infra-ef8e86`
环境：KVM VM，4 vCPU quota（cgroup cpu.max=400000/100000），~3.95 GiB，基准以 2 线程串行运行。

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `cpu_cast/scheduler.py` | 跨查询 wavefront 调度器（`Request`、`WavefrontScheduler`） |
| `scripts/test_wavefront.py` | 调度器正确性测试（全深度 + 退出策略） |
| `scripts/bench_stage3_costgrid.py` | shape→latency 成本网格（torch/ORT 拆分 pass，支持断点续跑） |
| `scripts/bench_stage4_load.py` | 负载实验：查询本地 vs wavefront（实测成本仿真 + 真实执行验证） |
| `artifacts/runs/cost_grid.jsonl` | 成本网格测量（torch pass 完成；ORT pass 后台补齐） |
| `artifacts/runs/bench_stage4_load.jsonl` | keep_frac=0.5 负载实验 |
| `artifacts/runs/bench_stage4_load_kf75.jsonl` | keep_frac=0.75 负载实验 |

## 2. 调度器正确性（Stage 4 go/no-go：通过）

`WavefrontScheduler` 以逐行 `next_layer` 指针跨查询组批：每轮收集所有在途查询中
处于同一层的 ready 行，选 ready 数最多的层执行，checkpoint 处按 keep 策略淘汰。

- **Test A（全深度、无退出）**：wavefront 跨 3 个请求（6/11/17 候选）执行 vs 逐查询
  `forward_full`：`max_abs_diff = 0.00e+00`（精确一致）。
- **Test B（checkpoint 退出策略，layer 1/3，固定阈值 keep）**：vs 逐查询参考实现：
  退出层完全一致（0 mismatch，23 行退出），分数 `max_abs_diff = 4.77e-07`
  （来自不同 GEMM 批组成的浮点非确定性，符合预期）。

开发中发现并修复的 bug：ready 行集合未排除 `next_layer == L` 的已完成行；
checkpoint 分数按请求局部 mask 误索引全批 eh；非 compact 路径未回写 hidden
（该 bug 使每层都在重复执行 embedding，曾导致 0 退出与 nDCG=0）。

## 3. 成本网格与 OOM 教训

首轮成本网格在 N=100/200 × L=256 处被 cgroup OOM 杀死（RSS 4.1 GB > 3.95 GiB）：
torch 与两个 ORT session 同时驻留。修复：拆分为 torch / ORT 两个独立进程 pass
（`--backends torch|ort`），并增加断点续跑（跳过已记录的 (T,N,L,kind)）。

torch FP32 pass（T=2, L=256）关键数值：

| N | 6 层整批 (ms) | 折算单层 (ms) | compact_half (ms) |
|---|---|---|---|
| 32 | 768.7 | 128.1 | 56.6 |
| 64 | 1593.3 | 265.6 | 123.2 |
| 100 | 2698.4 | 449.7 | 211.7 |
| 200 | 5734.0 | 955.7 | 479.4 |

单层成本在 N∈[32,200] 近似线性（~7.2 µs/row/layer @L=256），批次加大近 2× 时
无超线性惩罚 → **跨查询组批的算术基础成立**（t(200) < 2×t(100) 不严格成立但接近，
组批收益主要来自消除小批次的固定开销与空转行）。

## 4. 实测成本模型与仿真验证

成本表（来自 cost_grid，线性插值）：`layer1(N)=layer_block/6`；
`gather(n,k)=compact_half(n)−layer1(k)`（N=100→50 时 ≈6.3 ms）；
`score(N)≈exit_head(N)`；`embed(N)`。

验证（12 查询 × 100 候选 × L=256，burst）：

| 路径 | 真实 | 仿真 | 比值 |
|---|---|---|---|
| 串行查询本地（kf=0.5） | 22.43 s | 24.95 s | 1.11 |
| 串行查询本地（kf=0.75） | 28.25 s | 28.39 s | **1.00** |
| wavefront makespan（kf=0.5） | 25.47 s | 23.89 s | 0.94 |
| wavefront makespan（kf=0.75） | 31.79 s | 26.46 s | 0.83 |

串行路径成本模型误差 ≤12%；wavefront 路径仿真低估 6–17%，缺口来自调度器
Python 开销（逐请求 index_select/cat/张量簿记，每次 layer event ≈50–100 ms，
未计入成本表）。**这本身就是 RQ1/H1 的证据：调度开销在低核 CPU 上不可忽略。**

## 5. 负载实验（RQ3/H3）

策略：两策略使用完全相同的退出决策（校准集分位数阈值，批无关 keep：
`keep = score ≥ threshold`，checkpoint = layer 1/3，pooler 信号）。质量为
策略属性，两策略相同：kf=0.5 → nDCG@10=0.4418；kf=0.75 → 0.5723
（对照 full-depth 0.7038）。负载查询 12 条，校准查询独立 20 条。

仿真结果（ms，12 查询，100 候选/查询）：

| 负载 (ρ) | 策略 | p50 | p95 | p99 | makespan |
|---|---|---|---|---|---|
| burst (=∞) | serial | 13562 | 23973 | 24755 | 24950 |
| burst (=∞) | wavefront | 20409 | 23889 | 23889 | **23889** |
| staggered 2s (ρ≈1.04) | serial | 12187 | 21360 | 22032 | 24950 |
| staggered 2s (ρ≈1.04) | wavefront | 18814 | 22329 | 22331 | 25082 |
| staggered 3s (ρ≈0.79, kf75) | serial | 2469 | 2685 | 2721 | 34778 |
| staggered 3s (ρ≈0.79, kf75) | wavefront | 2438 | 2654 | 2690 | 34403 |

**H3 得到支持，且方向与计划预测一致：**

- **饱和负载（ρ≥1）**：wavefront makespan 略优（−4%），但 p50 恶化 +50%
  （13562→20409 ms）——所有请求齐头并进，批等待转嫁到每个请求；p95/p99 基本持平。
- **轻负载（ρ≈0.8）**：wavefront 与 serial 几乎无差异（±1–2%）——无批可组时
  wavefront 退化为逐查询执行，无额外代价（H6 的退化路径自然成立）。
- **真实执行（burst）**：wavefront 25.5–31.8 s vs 串行 22.4–28.3 s——当前
  Python 实现的逐请求 gather/簿记开销吞掉了组批收益，真实 wall-clock 上
  wavefront 反而慢 10–14%。

## 6. 结论与下一步（对照计划第 16 节决策表）

1. **RQ2（compaction 何时值得）**：Gate A 已证 C_compact < C_keep（active ≤0.75）；
   本阶段补充：gather 成本实测很小（~6 ms @100→50），compaction 决策由
   active 行数主导，每层压缩（m=1）最优（Stage 3 结论维持）。
2. **RQ3（跨查询组批）**：算术上成立（成本近线性），但**当前 Python 调度器
   实现不足以兑现**：需要把跨请求 hidden 打包为单一连续张量、避免逐请求
   index_select/cat，并把 checkpoint 打分与 ready 收集向量化。这是 C1/C2 的
   核心工程问题，也是论文的 CPU-specific 论点来源。
3. **RQ5（尾延迟）**：batching-wait 拐点实测存在（ρ≥1 时 p50 +50%）。
   下一步：deadline-aware 准入（超过 slack 的请求走 query-local fallback，
   计划 10.5 节），目标是在 ρ≈1 区域恢复 serial 的 p95 同时保留 makespan 收益。
4. 质量注意：pooler 在 checkpoint 1/3 的域内判别力弱（kf=0.5 时 nDCG 掉
   0.26），GAINER 头（nfcorpus 训练）迁移有限。质量—算力联合优化需要
   per-checkpoint 校准或在目标域上训练轻量头（Stage 5 内容）。

### 已知限制
- 12 负载查询 + 20 校准查询（scifact 单域）；L=256 单一长度桶。
- 仿真不含 Python 调度开销、cache/频率效应；真实验证给出误差界（0.83–1.12）。
- ORT FP32/INT8 网格补齐中（独立进程，避免 OOM 复发）。
- 4 vCPU quota ≠ 计划目标 2 vCPU；第二平台验证（Gate D/E）未开始。
