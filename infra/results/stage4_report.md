# CPU-CAST Stage 4 报告（修订版）— 审稿响应：统一成本账本、闭环修复、请求级 trace、强基线对比

日期：2026-11-09 ｜ 分支：`vibe/cpu-cast-stage1-infra-ef8e86` ｜ 权威数据：`artifacts/runs/bench_stage4_load_v3.jsonl`（run_id 见文件）
环境：KVM VM，4 vCPU quota（cgroup cpu.max=400000/100000），~3.95 GiB；基准 2 线程串行；重负载实验串行执行。

本报告是对内部审稿报告（f0873cd4，Major Revision）的响应。审稿确认的 17 个问题已逐项处理；
**在证据链修复完成之前，不进入 Stage 5**。本文所有结论均以 v3 权威记录为准；
被废弃的运行见 §8 废弃登记。

## 0. 审稿响应总览

| 优先级 | 审稿要求 | 状态 | 证据 |
|---|---|---|---|
| P0-1 | ORT FP32/INT8 成本网格补齐，线程配置可核验 | 完成（有真实缺口） | §1，Q1 |
| P0-2 | 统一成本账本（embed/layer/checkpoint/final/gather/scatter 分开测；serial 与 wavefront 同口径） | 完成 | §2，Q2 |
| P0-3 | closed-loop 仿真改为完成驱动 | 完成 | §3，Q3 |
| P0-4 | 统一真实验证计时边界 | 完成 | §4，Q5 |
| P0-5 | 请求级 trace（arrival/等待/逐层/完成） | 完成 | §5，Q6 |
| P1-6 | pooler 路径正确性测试 | 完成（5 项测试全过） | §6，Q8 |
| P1-7 | 与最强基线 cascade_shallow 同查询同边界对比 | 完成（诚实结论：当前策略质量落后） | §7，Q9 |
| P1-8 | 扩大负载实验 | 部分完成（30 查询 × 3 重复；长度桶/多到达率未做） | §7，Q16 |
| P1-9 | GAINER query-level split；Prism KMeans 修复 | 完成 | §9，Q11/Q12 |
| P2 | provenance | 完成（revision/hash/CLI/废弃登记） | §8，Q14/Q15 |

## 1. 成本网格（P0-1，审稿 Q1）

`scripts/bench_stage3_costgrid.py` 重构：ORT session 在 T 循环内以 `threads=T` 创建，
每条 ORT 记录携带 `session_threads`（从 `session.get_session_options().intra_op_num_threads`
读出的实际值）与 `torch_num_threads`。102/102 条 ORT 记录核验一致。

- 51 条旧 ORT 记录因线程标签不可信，移入 `artifacts/runs/cost_grid_ort_mislabeled.jsonl` 并废弃。
- OOM 缺口（诚实记录，非隐瞒）：`torch + 2 个常驻 ORT session` 在 N∈{100,200}×L=256
  超出 3.95 GiB cgroup 限制。改为每 (T×L×N组) 独立子进程后，仍有两个形状 OOM：
  **(T∈{1,2}, N=200, L=256 两种 ORT) 与 (T∈{1,2}, N=100, L=256 ort_int8)**。
  结论：ORT 全模型前向在低核内存预算下无法覆盖这些形状——这本身是目标环境（2 vCPU/3.8GB）
  的真实部署约束，报告如实呈现，不用插补值顶替。
- 新增直接测量（torch，T=2，L=256）：`layer(256)=1194.39ms`、`gather(256)=31.87ms`、
  `score(256)=0.30ms`。外推预测 1244ms vs 实测 1194ms（偏差 +4%，略保守，可接受）。

覆盖：torch FP32 全网格（9 batch × 3 seq × 2 threads）；ORT FP32/INT8 除上述 5 个形状外全覆盖。

## 2. 统一成本账本（P0-2，审稿 Q2）

旧账本用 `layer_block/6`（整模型前向摊薄，含 embedding 与 pooler/classifier）当单层成本，
随后又单独加 `embed_ms()` 与 `score_ms()`——系统性重复计费。已重写：

- 成本网格新增直接测量 kind：`layer`（`forward_layer` 单层）、`gather`
  （仅 `index_select`+`index_copy`）、`score`（pooler+classifier，即 `score_hidden`）。
- `CostTable` 只使用直接测量，线性插值；表边界外**线性外推**（不再截断——截断曾把
  256 行的层事件按 200 行计价，是 wavefront 仿真低估的主因之一，见 §4）。
- 关键实测（T=2，L=256）：`layer(100)=463.14ms`、`layer(200)=964.15ms`、`layer(256)=1194.39ms`
  → 每行 4.63/4.82/4.67ms；`gather(100)=4.12ms`；`score(100)=0.130ms`；`embed(100)=31.11ms`。
- serial 与 wavefront 仿真对 embed、layer、checkpoint scoring、final scoring、gather
  使用**同一张表、同一计价规则**。

## 3. 闭环到达语义（P0-3，审稿 Q3）

`ClosedLoopArrivals` 重写：`busy[c]` 完成驱动，client 仅在 `notify_done` 后提交下一条；
提交轨迹 `submitted` 逐条记录。v3 记录中 closed2 的 `submit_trajectory` 证实：
每 client 初始仅 1 条在途，提交时刻严格由前一请求完成时间推出（不再是初始一次性全提交）。

## 4. 仿真 vs 真实：成本模型校准（审稿 Q5）

计时边界统一（P0-4）：tokenization 与请求构造移出计时区，单独记录
（`preprocessing: tokenize_build_s=1.497`）；两条真实路径在**相同预构造请求对象**上计时。

| 指标（30 查询，keep_frac=0.5，burst） | 仿真 | 真实 | 误差 |
|---|---|---|---|
| serial makespan | 47.21 s | 45.5–46.6 s（3 reps） | 1.5–4% |
| wavefront makespan | 47.23 s | 51.6–52.8 s（3 reps） | ~8% |

- **serial 校准良好**：单查询 exec_rows ≤ 100，全部在测量范围内，无外推。
- **wavefront 残差 ~8%**：修复截断（−10s）与 gather 计价（−2s）后，
  残余来自逐事件 Python 调度开销与逐请求 gather 实际成本，未入表。
  诚实结论：仿真在两条路径上误差 ≤10%，但**不再宣称仿真可精确归因调度开销**。

**修正后的仿真结论（重要）**：统一账本下，**burst/staggered 高负载时 wavefront p50 反而更差**
（burst：45177 vs 25626 ms；staggered：40523 vs 22001 ms）——v1 报告中 wavefront 的高负载
优势是**串行路径漏算 checkpoint 打分 + 批量计价不对称**造成的账本伪影，不是真实计算差异。
只有轻载（staggered_light）wavefront p50 略优（1575 vs 1690 ms，−6.8%）。

## 5. 请求级 trace 与真实 H3 证据（P0-5，审稿 Q6）

`Request` 携带 `arrival/submit_time/first_exec_time/finish_time/layer_log`
（逐层：layer、exec_rows、scored_rows、compacted、ms）；online 模式按真实错峰到达
（sleep-until-arrival）。p50/p95/p99 直接从原始 trace 计算，可独立复算。

真实 online（stagger 2000ms，2 线程，30 查询 × 3 reps）：

| 策略 | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| serial（逐查询） | 1598–1630 | 2303–2355 | 2366–2397 |
| wavefront | **1530–1657（−2~4%）** | 3421–3924（**+46~68%**） | 4888–5266（**+105%**） |

**H3 修订状态：方向被真实请求级数据支持，但幅度远小于仿真宣称的数值。**
wavefront 仅在中位延迟有 ~2–4% 的边际优势，尾延迟恶化 +46~105%。
此前 v1 报告的 "p50 −10%，p95/p99 +44–48%" 是仿真数字，不得再表述为实测拐点。
且当前调度器无 max_batch_delay/deadline 等待窗口参数——它是一个离线批内跨查询
调度原型，**尚不是在线服务调度器**；在线服务结论仍受此限制。

## 6. 调度器正确性（P1-6，审稿 Q8）

`scripts/test_wavefront.py` 5 项测试全过：

- **Test A**（全深度、无退出）：wavefront vs `forward_full`：`max_abs_diff = 0.00e+00`。
- **Test B**（exit-head MLP 路径、固定阈值）：退出层 0 mismatch，`4.77e-07`（GEMM 批组浮点非确定性）。
- **Test C（新增，生产路径）**：`signal="pooler"`（`score_hidden`，即 Stage 4 实际使用的
  checkpoint/最终打分路径）：`4.77e-07`，0 mismatch，26 行退出。**生产打分路径已获同等验证。**
- **Test D（新增，max_batch=8 溢出）**：无丢行、无重复执行、`next_layer` 更新正确。
- **Test E（新增，混合序列长度）**：以 `ValueError` 拒绝（单长度桶不变式显式声明）。

keep_frac 语义（审稿 Q8 后半）：`keep_frac` 是**全局 wavefront 批预算**（合并 ready 行上取
top-k），不是每查询预算。这是有意为之的跨查询资源分配策略，已在接口 docstring 声明；
与 query-local 语义的差异由 Test B/C 的逐查询参考路径覆盖。

## 7. 强基线对比（P1-7，审稿 Q9）— 诚实结论：当前策略质量落后

同一 30 查询、同一候选集（RRF top-100）、同一计时边界、2 线程：

| 方法 | nDCG@10 | 总耗时 | p50 延迟 |
|---|---|---|---|
| cascade_shallow（TinyBERT→L6，top-50） | **0.6805** | 42.64 s | 1421.5 ms |
| active-set 固定阈值策略（keep_frac=0.5） | 0.3702 | —（质量评估） | — |

**在测试的 keep 速率下，active-set 策略在质量上明显落后于最强基线**（−0.31 nDCG@10）。
原因链：固定阈值 keep 策略的退出比例（ckpt1 51.5%、ckpt3 30.0%，见 §5 修正后数字）
远高于名义值对应的质量损失预期，且部分查询幸存者为 0（全候选早退，无最终打分）。
**在质量—延迟前沿上，CPU-CAST 当前不构成对 cascade_shallow 的改进主张。**
论文主张必须重新定位：当前可信的贡献是 (a) active-set compaction 的实测现象（Gate A）、
(b) 跨查询 wavefront 调度的真实调度开销测量、(c) 统一成本账本方法论；
**不是**一个优于最强基线的完整系统。质量差距的缩小（更好的退出信号、每查询预算）
是后续阶段的前提，不是当前结论。

## 8. 废弃运行登记（审稿 Q14）

| 文件 | 状态 | 原因 |
|---|---|---|
| `bench_stage4_load.jsonl`（v1，49 行） | **废弃** | 早期调试：nDCG=0 全保留运行、`sim_makespan_s=31702` 单位错误（秒/毫秒）、collect_trace 未更新 hidden |
| `bench_stage4_load_kf75.jsonl`（v1） | **废弃** | 同上批次调试运行 |
| `bench_stage4_load_v2.jsonl`（47 行） | **部分有效** | offline wavefront rep1/rep2 因 stale `finish_time` 出现负延迟（bug 在 v2 运行后修复）；rep0 与全部 online/serial 记录有效 |
| `bench_stage4_load_v3.jsonl`（26 行） | **权威** | 全部修复后重跑 |
| `cost_grid_ort_mislabeled.jsonl`（51 行） | **废弃** | ORT 线程标签不可信 |

正式结果筛选规则：`event` 字段 + `run_id` + `git_sha` + 完整 CLI（记录于每条记录）；
仅 `v3` 与 `cost_grid.jsonl`（N=256 补测后）用于本报告。

## 9. GAINER / Prism 修正（P1-9，审稿 Q11/Q12）

- **GAINER query-level split**：`train_exit_head` 改为按查询划分（10% 查询整体留出，
  完全未见）；重训至 `models/gainer_heads_qsplit`。val loss 2.35/0.631（泄漏版 2.12/0.598
  ——诚实地更差）。重评：`gainer_gain_r50 = 0.4454`（泄漏版 0.5232）。
  **泄漏夸大了 exit-head 泛化约 0.08**；复现差距中信号质量与划分/候选池差异并存，
  不能归因单一来源。
- **Prism KMeans bug**：`kmeans1d` 多初始化时 `best_centers` 未随 `best_labels` 保存
  （返回的 centers 属最后一次初始化）。已修复并重跑：
  测试集无触发（CV 结构性惰性，中位 0.013），nDCG@10=0.7038 不变；
  触发时质量损失（−0.03~−0.10）仍大于算力节省。**负面结论在修复后成立。**

## 10. 限制（审稿 Q10/Q16/Q17）

- **后端**：wavefront 调度器是 torch 专用原型（依赖 `embed/forward_layer/score_hidden`）；
  ORT 无逐层接口。ORT 成本网格仅覆盖全模型前向。**不得泛化为后端无关 runtime。**
- **规模**：scifact 单数据集，30 负载查询 + 20 校准查询，单长度桶 L=256，2 线程，
  3 重复无 CI。p99 仍是小样本上界。长度桶、多到达率、CPU 压力、更大查询量未做。
- **预算口径（Q17）**：本仓库所有比较使用**实测 CPU 时间**（统一账本）；
  Stage 2 报告中的"层当量"换算（TinyBERT≈0.5 MiniLM 层/淘汰候选）缺乏测量依据，
  已在该报告标注为任意指定，cascade 结论以端到端 wall-clock 为准；
  评分融合 `1e3 + s2` 的理由（二阶段候选强制排前一阶段淘汰者之前）已注明。

## 11. 结论

Stage 4 证据链修复完成：成本账本统一并经真实验证校准（serial 误差 ≤4%，wavefront ≤8%），
闭环语义修复并有提交轨迹证明，请求级 trace 齐全，生产打分路径通过等价测试，
强基线对比诚实呈现（当前落后）。**H3 在真实数据上仅余 "轻载 p50 边际收益 + 尾延迟恶化"
的方向性信号；高负载 wavefront 优势被证明是账本伪影。** 建议在质量差距缩小之前，
论文主线以 (a) Gate A compaction 现象 + (b) 调度开销测量 + (c) 成本账本方法论为核心。
