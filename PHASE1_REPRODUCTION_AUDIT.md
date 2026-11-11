# CPU-CAST 代码、环境与实验结果严格审计

> 审计基线：默认分支 vibe/cpu-cast-stage1-infra-ef8e86，提交 8d45d280a461eb37fcc560c5e7d68254e43a2e5b  
> v5 实际运行代码：549fa7d8e782aaf6d5ce90fa5f53da1d89b22b48  
> 目的：判定现有结果是否可信、哪些必须作废或重跑，以及全部基线复现前必须解决的阻断项。

---

## 0. 总结裁决

**当前仓库具备可运行的研究原型，但尚未形成可复现、可审计的 baseline reproduction pipeline。** 当前已有结果主要是 CPU-CAST 自己的代理实现、缩减数据实验和小规模单次 benchmark；不能把这些结果表述为 Early Exiting MonoBERT、Shallow Cross-Encoders、SEE、CEC-Ranking、GAINER、PRISM、MICE 的官方复现。

本审计确认：

- **P0：实现/结果正确性问题**：v5 SerialScheduler 没有更新逐层 hidden state；GAINER 训练与实测的特征定义不一致，且报告质量来自仿真、延迟来自另一条实测路径。
- **P0：数据与环境缺少不可变 pin 和完整 provenance**：数据下载使用可变 HF main，缓存命中不校验 hash；依赖锁不完整；Java/GPU/旧 Python 环境缺失；实验日志缺少精确 code/data/model/environment 来源。
- **P1：benchmark 协议存在偏差**：query 子集按文件顺序截取；p99 样本量不足；cascade/active-set 计时边界不一致；candidate manifest 缺 runfile hash；macro/micro Recall 定义混用。
- **P1：baseline 覆盖率远低于计划承诺**：官方仓库有部分 shallow clone，但没有逐项按论文运行官方脚本、重建论文表格、核对输出或完成独立复跑。
- **最终结论**：修复 P0、重新执行受影响实验之前，Stage 4 v5 在线串行对照、现有 GAINER 比较和若干旧报告只能作为历史调试记录，不能作为论文主证据。

本审计不把“脚本存在”“仓库已 clone”“模型能加载”“在 CPU 上跑通一个代理”视为论文复现完成。

## 1. 严重性分级

| 级别 | 定义 | 对论文结果的处理 |
|---|---|---|
| P0 — Blocking correctness/provenance | 算法语义、比较对称性、数据完整性或来源无法保证 | 修复前禁止使用受影响数值支持论文结论；必须重跑 |
| P1 — Major reproduction gap | 官方协议未复现、统计/候选/计时不足、环境差异未量化 | 可作探索性结果，不能称完整复现或最终公平比较 |
| P2 — Reproducibility hardening | 文档、脚本可维护性、自动检查和归档不完善 | 纳入完整复现工程 |

## 2. P0：必须先修复的代码与结果问题

### P0-1. SerialScheduler 的逐层 hidden state 未回写，v5 串行基线计算的不是正常的逐层 Transformer

文件：**cpu_cast/scheduler.py**，类 **SerialScheduler.run**。此问题在 v5 实际执行代码 549fa7d8e782... 与当前提交 8d45d28... 中都存在。

当前路径逐层执行：
1. 从 r.hidden 取出全量或 active rows；
2. 计算 h = forward_layer(h, mask, layer)；
3. 进行 checkpoint 评分；
4. 进入下一层。

但是没有将本层输出写回 r.hidden。与之相反：
- **ActiveSetRanker.run_active_set** 在压缩路径中执行 hidden.index_copy(...)，在非压缩路径中替换 hidden；
- **WavefrontScheduler.run** 在 scatter 阶段执行 r.hidden = r.hidden.index_copy(...)。

结果是 SerialScheduler 在下一层仍从原始 embedding hidden state 出发，而不是上一层输出。最终 survivor 的 score_hidden(r.hidden[idx]) 也读取未更新的 hidden state。checkpoint scores、exit decisions、候选执行层数和 latency 都可能因此改变。

**裁决：v5 的 real_serial_online 数值整体失去有效比较资格。** 不能只修一个 score 或改报告措辞；必须修复执行逻辑、添加 parity tests，再用同一 trace 对 Serial/Wavefront 重新测量。v5 中 wavefront 的独立结果也应重新跑成一组新的配对实验，不应继续引用旧串行结果计算相对收益。

必须新增测试：
- 无 early exit 时，SerialScheduler 输出与 Hugging Face 标准完整前向一致；
- 固定阈值时，SerialScheduler 与独立 per-query reference 的 scores 和 exit_layer 一致；
- 压缩与不压缩路径都覆盖；
- active row index 非连续时覆盖；
- 多个 checkpoint、无 survivor、全 survivor、单候选、候选数小于 batch 等边界覆盖；
- 同一 query/策略下 Serial 与 Wavefront 的每候选输出在容差内一致。

### P0-2. SerialScheduler 的 checkpoint score scatter 另有索引错误

同一文件、同一方法：

当前逻辑在 non-compacted 分支中先通过 cur_active = cur[idx] 选出 active candidates，再计算 finished；但保存退出分数时使用 cur[finished]。当 idx 是非连续索引时，finished 是 active-local mask，而 cur 是全量候选维度，两个索引空间不一致。应该从与 cur_active 对应的张量提取退出分数，再映射回原 candidate indices。

该错误与 P0-1 独立，需有专门的非连续 active-row 测试。只测试全部 rows 都 active 的情况无法发现它。

### P0-3. GAINER measured policy 与训练期特征定义不一致

文件：
- **scripts/train_gainer_heads.py**
- **scripts/bench_stage2b_gainer.py**

训练侧 **features_at** 的 rank fraction 使用总候选槽位数作为分母（cur.shape[1]）；实际 measured path 的 **features_flat** 使用当前 active candidates 数量作为分母。随着 active set 缩小，rank feature 的数值范围改变。训练的 gain ranker 与部署时输入的 feature distribution 不一致。

此外，**bench_stage2b_gainer.py** 的结果组织方式是：
- nDCG / MRR / top-10 agreement 存于 sim_rollout 的仿真结果；
- 实际 conditional execution 只把 measured latency / layer count 写入结果；
- 实际测量返回的 scores_m 没有被用于计算和保存实际质量指标。

因此现有表格不是“同一个被测实现”的质量与延迟配对结果。它把仿真质量和实测开销拼在一起；如果 measured route 的排序结果与仿真存在差异，表格不会发现。

**裁决：当前本地 GAINER-style 结果不构成有效的 measured GAINER comparison。** 修复时需要统一 feature transform，加载训练时保存的 feature schema/version，实测输出直接计算质量指标，并逐 query 对比 simulated 与 measured scores/exit decisions。任何不一致都必须阻断写表。

更根本的是，当前项目实现明确缩减了官方 GAINER 实验：仅用 NFCorpus train、MiniLM-L6、top-50、本地 BM25、max length 256、简化特征及单种子。它不是官方 GAINER 复现，不能根据这份数值声称 GAINER 方法无效。

### P0-4. 数据下载使用可变 revision，且缓存命中不校验内容

文件：**scripts/download_datasets.py**

发现：
- corpus/queries/qrels URL 使用 HF resolve/main/，未固定数据仓库的 immutable commit revision；
- fetch() 只要目标文件存在且大小大于 0 就复用，不核验预期 SHA-256；
- qrels 下载使用宽泛的 except Exception 并报告为 “no split qrels”，网络失败、服务端错误、权限错误也会被误报成 split 不存在；
- 当前 checked-in datasets_manifest.json 已包含部分 qrels_sha256（四个 BEIR 数据集）以及少量 SciFact candidate-run SHA；但这些字段并非由 download_datasets.py 生成/校验，下载脚本本身只写 corpus/queries parquet SHA 与 qrels 行数。manifest 也没有 pin qrels/dataset 的源 HF commit revision，没有覆盖所有 candidate runs（例如 NFCorpus）的完整 runfile hash，也没有转换后 corpus.jsonl / queries.jsonl / qrels TSV 的独立 hash。换言之，当前有部分事后补录的 hash，但还没有可 clean rebuild 的、端到端自洽的 immutable data manifest。

这意味着相同命令在不同时间可能得到不同数据；已有缓存即使损坏、半旧或被人工替换，也会被当成正确文件。

修复要求：
1. 每个 HF dataset 和 qrels repository 都 pin 到完整 immutable commit SHA；
2. manifest 保存源 URL、revision、原始文件 SHA-256、大小、转化输出 SHA-256、行数、schema；
3. 缓存只有在 SHA 匹配时才可复用，否则报错并重新下载；
4. 将 NOT_FOUND_SPLIT 与网络、权限、校验失败分开；
5. 对 corpus/queries/qrels 做唯一 ID、重复行、缺失 doc/query、rel 值范围和 split overlap 校验。

### P0-5. 当前 BM25 candidate run 不是 Pyserini/Anserini 官方 BM25

文件：**scripts/build_candidate_runs.py**

当前 BM25 使用 rank_bm25.BM25Okapi，文本经 lower().split() 直接 whitespace tokenize，未固定 Anserini/Lucene analyzer、stemming、stopwords、k1、b，也没有与官方 index/runfile 的等价校验。并且只保留 BM25 score > 0 的候选，导致每个 query 可能不足 top-100。相同分数时使用默认 np.argsort，没有显式 docid tie-break，跨环境不保证候选排序完全确定。

因此：
- 当前 BM25 必须命名为 **rank_bm25 whitespace proxy**；
- 不得标注为 Pyserini BM25；
- R3 如使用它，须明确 candidate source 为项目自建 proxy；
- 官方 Pyserini / Anserini candidate generation 必须单独执行并保存原始 runfile。

candidate manifest 目前记录 path、query count、Recall@K，没有 candidate run SHA-256、qid set SHA、每个 query 候选数分布、检索模型 revision 和 retrieval 参数，无法锁定同一候选池。

### P0-6. 实验 provenance 不足，无法证明日志对应哪个精确代码/环境/输入

文件：
- **cpu_cast/logging_utils.py**
- **scripts/bench_stage2_baselines.py**
- **scripts/bench_stage2b_gainer.py**
- **scripts/bench_stage3_interval.py**
- **scripts/bench_stage4_load.py**
- **manifests/**

问题：
- 通用 JSONL logger 默认只记 timestamp/hostname/PID；
- Stage 2/2b/3 的 run metadata 没有完整 code SHA、seed、完整 CLI、model revision、candidate run hash、qrels/data hash、dependency lock hash；
- Stage 4 git_state() 只保存 12 位 HEAD，并仅用 git diff 计算 diff hash；staged changes 不在普通 git diff 中，untracked 文件也不会被纳入 diff 内容哈希；
- scripts/download_models.py 下载时按 revision SHA，但加载时没有统一校验本地权重是否匹配 manifest；
- 旧日志无法可靠恢复这些字段。

修复要求：每次 run 写入 40 位 git HEAD、dirty 状态、staged/unstaged diff 哈希、untracked 文件清单哈希、依赖 lock 哈希、完整环境摘要、模型 revision/文件 hashes、数据/qrels/candidate run hashes、完整 CLI、随机种子、线程/亲和性/cgroup 和输出文件 hashes。缺字段的历史结果标记 LEGACY_NON_AUTHORITATIVE，不得自动并入最终表。

## 3. P1：benchmark 协议与统计上的主要问题

### P1-1. Query 子集按文件/字典顺序切片，不是预注册的随机样本

Stage 4 使用 all_qids[:num_queries] 作为 load queries，随后一段作为 calibration queries；Stage 2/2b 同样使用 dict(list(runs.items())[:num_queries])。

这种方式能保证同一次输入文件中的 train/calibration/load qids 不重叠，但它不是随机抽样，也没有固定 query manifest、seed 或分层策略。数据存储顺序可能与 query/domain/topic 属性相关，导致小子集有偏。

要求：预先生成并 hash 固定的 train/calibration/dev/test qid lists；按照官方 split 使用 test；如需子样本，以明确 seed 抽样并记录 qid hash；按 query length、候选数、相关文档数等做分层统计。

### P1-2. 当前 p99 样本量不支持稳定尾延迟主张

- Stage 2 主要对 100 个 query 计算 per-query latency；
- Stage 4 v5 每次只有 30 个请求，重复 3 次；
- 多次重复同一批 30 个 query 不等价于大量独立 query 或请求样本。

30 个请求时 p99 几乎就是最大值附近的 order statistic。当前 v5 的 30 请求 trace 只能证明该特定 trace 下观察到的延迟，不支持一般化 p99/SLO 结论。

要求：load test 每个 workload/load point 至少 10,000 个真实完成请求、至少 5 次独立运行；分开报告 warmup 与 steady state；记录到达过程、并发、CPU steal、queue wait、失败/超时。若无法达到样本量，报告 p50/p90 或明确不报告 p99。

### P1-3. cascade_shallow 与 active-set 的计时边界不完全对称

文件：**scripts/bench_stage4_load.py** 的 run_cascade_shallow。

当前实现先在计时区域之外构建全候选 token tensors；cascade 内又对保留候选重新 tokenizer 一次，而 active-set 路径继续使用预构建 tensor。Stage 4 的 total timer 还包含 cascade 内模型加载，而每 query latency 不包含同一模型加载。这样不同数字的边界不一致。

要求：分别定义：
- cold-start latency（包含模型加载、tokenizer 初始化）；
- warm request latency（模型已常驻、包含线上请求所需 tokenize/筛选/重排）；
- model-only inference latency（输入 tensors 已准备）。

同一张表只能比较相同边界。最好复用原始 tokenization 并 slice tensors；若重新 tokenize 是实际 pipeline 的必要步骤，则两个方法都要在计时边界内计入各自实际预处理成本。

### P1-4. 报告中的“逐 query 结果”与原始产物不一致

infra/results/stage2_report.md 声称 artifacts/runs/bench_stage2.jsonl 含 per-query 明细；实际该文件是 11 条 aggregate JSONL 记录，不含逐 query 输出。当前日志不足以独立 bootstrap，也无法从原始文件重算 per-query 置信区间。

修复：每个 run 保存 query-level results（qid、候选数、逐候选分数/名次、指标、exit trace、latency）；汇总 JSONL 只作为 derivative summary，并保存从 raw results 生成汇总的脚本和 hash。

### P1-5. Macro Recall 与 micro Recall 定义混用

文件：**cpu_cast/metrics.py**。

evaluate() 中 pytrec_eval 的 recall_100 是 per-query measure 的 macro aggregation；recall_at_k() 手工统计所有 query 的相关文档命中总数再除总相关文档数，是 micro Recall。candidate manifest 使用后一种，但指标命名没有区分。两种计算回答不同问题，不能直接混用或在报告中不说明。

要求：明确命名 macro_recall@K 和 micro_recall@K，固定最终主指标，写单元测试对已知 toy qrels/run 验证；Candidate Recall、ranking recall、nDCG/MRR 分开报告。

### P1-6. Stage 2b GAINER 报告已经被后续 qsplit 记录修正，但旧报告仍然保留旧数值

infra/results/stage2b_stage3_report.md 中的 GAINER 数字（例如 current_rank_r50 0.5656、gainer_gain_r50 0.5232）与后续 artifacts/runs/bench_stage2b_gainer_qsplit.jsonl 的结果（0.5775、0.4454）不同。报告本身说明训练流程和数据规模有限，但没有将旧报告显著标成 superseded。

要求：保留历史报告但加 SUPERSEDED / NON-AUTHORITATIVE 标记，列出 replacement run id、代码 SHA、差异来源。不能静默覆盖或把两个结果放进同一比较表。

### P1-8. 数据 loader 静默覆盖重复 ID / qrels 项，输入完整性错误无法被发现

文件：**cpu_cast/data.py**。

- load_corpus() 与 load_queries() 直接按 ID 写入 dict；重复 _id 会覆盖先前记录，不计数、不报错；
- load_qrels() 对重复的 (qid, docid) 直接覆盖旧 relevance；
- load_qrels() 无条件丢弃第一行作为 header；输入格式不符合预期时，可能静默丢掉第一条真实 qrel；
- loader 没有检查 qrels qid 是否存在 queries、qrels docid 是否存在 corpus，也没有输出 duplicate/missing counts。

修复要求：读取时统计并拒绝重复 ID/key；按明确 schema 判断 header，而不是无条件跳过；对 qrels-corpus-query join 做完整性检查；生成 machine-readable validation report。任何缺失/重复不能仅打印 warning 后继续跑正式实验。

### P1-7. 当前硬件探针不能完整反映有效配额

文件：**scripts/probe_hardware.py**。

问题：
- Path("/proc/xen") is False 恒不成立，JSON 中 virtualization 判定错误；
- psutil.virtual_memory() 是宿主可见内存，不等于 cgroup memory.max；
- 未把 cpu.max、cpuset.cpus.effective、memory.max、memory.events、taskset affinity、cpu.stat/steal 纳入 canonical hardware JSON；
- 线程测试包含 5 logical CPUs，但当前 cgroup CPU quota 只有 4 vCPU；
- memory bandwidth probe 使用全部可见 logical CPUs，可能超配额；
- perf stat -- sleep 2 只测 sleep 进程，不能当作模型运行硬件 counter。

要求：合并 env_probe 与 hardware_probe，记录 host-visible 和 cgroup-effective limits 两套值；对每个 benchmark 线程设置做配额敏感解释；硬件计数器必须包围目标 benchmark 进程本身，权限不足明确标记 unavailable。

## 4. 环境审计：当前沙箱能做什么，不能做什么

| 资源 | 当前证据 | 对复现的影响 |
|---|---|---|
| CPU | Xeon 6985P-C KVM guest；5 logical vCPU 可见，cgroup quota 4 vCPU | 可做 CPU port，但不是论文原始设备；必须 pin affinity/threads |
| Memory | cgroup memory.max 约 3.95 GiB；无 swap；曾有 anon-RSS 约 4.1 GB OOM | 不能并行跑重模型；单独进程、分阶段退出、记录 memory.events |
| Disk | 约 5.5 TiB 可用 | 不是当前主限制；注意缓存一致性与 model/data manifest |
| Python | 3.12.15 | 不可原样运行 Python 3.7.7 的 MonoBERT 原始环境 |
| PyTorch | 2.15.0+cpu | 无 CUDA；不能复现 GPU 论文效率/训练结果 |
| Transformers | 5.22.1 | 与多个旧 repo 的旧版内部 API、依赖版本需隔离 |
| Java | 未安装；没有 sudo | Pyserini/PyTerrier 的 Lucene/Terrier 工作流阻塞，需 user-space JDK |
| GPU | 当前沙箱无 NVIDIA/CUDA runtime | SEE、GAINER 官方 GPU 路径和需要 GPU 的训练不可在本环境原样完成 |
| OS privileges | 无 sudo / no-new-privileges（用户提供的环境检查） | 依赖 user-space 安装、容器权限需验证 |

### 当前 Python 依赖并不是可重建 lock

文件：**requirements.txt**。

Torch、Transformers、Sentence Transformers、ORT、BEIR 有部分 pin，但 numpy、scikit-learn、onnx、fastembed、pytrec-eval-terrier、huggingface_hub、datasets、pandas、psutil、rank_bm25 未 pin。没有 Python patch version lock、wheel hash、完整 transitive lock 或安装平台声明。manifests/python_env.txt 更接近整个沙箱的 pip freeze，不是项目最小依赖规范。

configs/runtime.yaml 设定各线程数为 1，但当前关键实验脚本使用 2 线程。配置、环境变量、实际 torch/ORT thread counts 不一致。

必须建立：
- CPU-CAST 共用环境的 lock（python patch、所有依赖精确版本与 hash、CPU wheel source）；
- 每个原始 baseline 的独立 environment/lock/container；
- 统一的环境报告与每次 run 的环境 hash；
- 不允许为了让旧 baseline 安装而覆盖/升级共用主环境。

### Java / Pyserini 约束

Pyserini 官方安装说明当前要求 Python 3.12 和 Java 21；底层 Anserini/Lucene 使用 JVM，预建 MS MARCO index 可以避免自己构建索引，但不能让 Lucene 检索本身不再需要 Java。预建 MS MARCO index/runfile 也不能代替 SciFact/NFCorpus 的候选检索。

优先用户目录安装 Temurin/OpenJDK 21，记录 tarball SHA-256、Java version、JAVA_HOME；如需要从源码 build Anserini，再单独准备 Maven。若环境无法安装 Java，应冻结现有 candidate run 或使用清楚标注的 Python proxy，不能标成 Pyserini R1。

## 5. 全部基线当前真实复现状态

| 基线 | 仓库/Artifact 状态 | 当前实现/运行状态 | R1 | R2 | R3 |
|---|---|---|---|---|---|
| Full-depth MiniLM-L6 | HF 模型已 pin，CPU 可加载 | 当前内部参考模型，主要在 SciFact 子集评估 | 非特定论文 R1 | 部分完成 | 部分完成 |
| Shallow Cross-Encoders | 官方 repo 未加入 clone 脚本/基线 manifest | TinyBERT/MiniLM shallow weights 有下载；未完成官方 gBCE 训练与官方评估闭环 | 未完成 | 仅内部 proxy | 未完成 |
| Early Exiting MonoBERT/Xin20 | 官方 repo SHA 有记录 | 没有原始 Python3.7.7/PyTorch1.3.1/CUDA10.1 环境运行记录 | 环境阻塞/未完成 | 未完成；现代模型试验不能替代 | 未完成 |
| SEE | 官方 repo SHA 有记录 | see_token 是 embedding-layer token max-sim proxy，不是官方 SEE；没有 NVIDIA Docker/RQ scripts 全跑和数据哈希 | 未完成/资源阻塞 | proxy only | 未完成 |
| CEC-Ranking | 官方 repo SHA 有记录 | 未见 MS MARCO + Quora 完整校准与 test run；未见 RCPS 原始对照结果 | 未完成 | 未完成 | 未完成 |
| GAINER | 官方 repo clone + notebook/预注册/registered outputs 可访问 | 项目内是缩减版实现，feature mismatch，质量/延迟分离 | 未完成 | 缩减 port，且当前实测结果无效 | 未完成 |
| PRISM | 官方 artifact 页面/Zenodo 地址已识别 | 尚未证明官方 artifact 运行成功；项目实现明确是 reimplementation | 未完成/原始设备受限 | reimplementation only | 未完成 |
| MICE | 官方 Git repo SHA 有记录 | 未见 recursive submodules、uv sync、官方 experiment config/checkpoint inference 的完整执行记录 | 未完成 | 未完成 | 未完成 |
| MICE-P | MICE/Fast-PLAID 仓库可取得 | 未见独立端到端 indexing/retrieval pipeline 复现 | 未完成 | 未完成 | 未完成 |
| Pyserini/BEIR | Pyserini clone SHA / BEIR package 有记录 | 当前 candidate run 使用 rank_bm25 proxy；Java 缺失 | 未完成 | proxy only | 未完成 |

R0（拿到仓库或模型）不是 R1（按原文复现）。同样，R2 port 和 R3 harmonized comparison 也必须分开验收。

## 6. 对现有结果的使用裁决

| 结果/报告 | 裁决 | 原因/后续 |
|---|---|---|
| Stage 0/1 smoke tests | exploratory / legacy | 可以证明模型能加载、基本数据格式能读；不能作为 baseline 主结果 |
| Stage 2 baseline matrix | exploratory only | 100 query SciFact、内部候选池；SEE/Gainer/PRISM 为 proxy；单次 query latency，缺 CI/provenance |
| Stage 2b GAINER | invalid for official comparison | 训练与推理特征不一致；质量来自 sim、延迟来自 measured path；缩减数据/模型 |
| Stage 2b/3 report old values | superseded | 与后续 qsplit result 不一致，需要历史标注 |
| PRISM CV reports | reimplementation-only | 不等于官方 artifact，统计量/模型和论文须独立核对 |
| Stage 4 v3 tail-latency conclusion | retracted | measured_service_replay 不能代表真实 FIFO online queue |
| Stage 4 v5 serial-vs-wavefront | invalid comparator pending rerun | SerialScheduler hidden state 不回写，serial forward semantics 错误；30 请求也不满足 p99 证据标准 |
| Stage 4 v5 cascade_shallow quality | exploratory evidence | 同 30-query 小样本质量差距有指示意义，但 timing boundary 需统一并在更多 query/dataset 验证 |

不要删除历史结果。保留原始文件和 SHA，在报告里标为 RETRACTED、SUPERSEDED 或 EXPLORATORY_ONLY，并链接修正后的替代 run。

## 7. 修复优先级

### P0-A：correctness blockers
- 修复 SerialScheduler hidden-state writeback；
- 修复 active-local mask 与原始 candidate index 的分数映射；
- 增加 SerialScheduler 全深度与 early-exit parity tests；
- 修复 GAINER feature scaling mismatch；
- measured GAINER results 必须从 measured returned scores 计算质量指标；
- simulated/measured ranking 与 exit decisions 做逐 query parity；
- cascade_shallow / active-set timing boundaries 拆分 cold/warm/model-only。

### P0-B：数据与 provenance
- 固定所有 dataset/qrels HF commit revisions；
- 缓存 SHA 校验，输出转换后文件 SHA；
- candidate run file SHA、qid SHA、candidate counts、retriever config 全部入 manifest；
- duplicate/missing IDs、qrels joins、rel value、split overlap 自动审查；
- 所有实验日志加入完整 provenance；
- 历史旧 run 标记 legacy/non-authoritative。

### P0-C：环境
- 建立可重建 CPU common env lock；
- 建立 baseline-specific original environment lock/container；
- 安装并验证 user-space Java 21，或明确 Pyserini R1 blocked；
- 获取外部 GPU/原始设备运行资源；当前 CPU 沙箱不能替代 CUDA/M2/RTX 实验；
- 将 cgroup effective resources 纳入统一环境探测。

以上 P0 完成并通过独立复跑前，不应开展新的算法/调度器结论性比较。

## 8. 复现完成的定义

一条 baseline 只有在以下项目齐全时才可标为 R1 complete：
1. 官方论文、代码、模型、数据版本和许可证确认；
2. 官方依赖环境能够创建，或 exact blocker 被证明；
3. 官方主实验/脚本运行成功；
4. 论文关键表格/曲线由原始输出重建；
5. 原始输出、逐 query 结果、latency traces、模型和数据 SHA 已归档；
6. 与论文报告差异逐项解释；
7. 另一份 clean checkout / 独立 runner 能重跑关键配置。

R2 complete 要有算法语义测试与 reference parity；R3 complete 要固定同一候选池、qid/qrels、输入文本、计时边界和统计协议。任何一类资源受限都应如实标为 partial/blocked，不允许把其他等级代替。

## 9. 官方参考入口

- [Early Exiting MonoBERT](https://github.com/castorini/earlyexiting-monobert)
- [Shallow Cross-Encoders](https://github.com/asash/shallow-cross-encoders)
- [SEE SIGIR'25](https://github.com/veneres/SEE-SIGIR25)
- [CEC-Ranking](https://github.com/alexlimh/CEC-Ranking)
- [GAINER](https://github.com/ibrahimjamadi98/GAINER)
- [PRISM artifact / Zenodo](https://zenodo.org/records/18809731)
- [MICE](https://github.com/xpmir/mice)
- [Fast-PLAID](https://github.com/lightonai/fast-plaid)
- [Pyserini installation](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
- [BEIR](https://github.com/beir-cellar/beir)
