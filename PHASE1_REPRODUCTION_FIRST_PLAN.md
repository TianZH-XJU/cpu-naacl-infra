# CPU-CAST Phase I：基线复现、Benchmark 运行与严格证据审计计划

> **状态：强制前置阶段（Blocking Gate）**  
> **版本日期：2026-11-09（rev.2：纳入 Stage 4 v5 结果与资源矩阵修订）**  
> **研究纪律：Phase I 未通过验收，不进入新算法/控制器研发，不对外声称 CPU-CAST 已优于现有方法。**
>
>
> **本计划的配套审计与执行文件：**
>
> - [代码、环境与实验结果严格审计](PHASE1_REPRODUCTION_AUDIT.md)：P0/P1 问题、结果有效性裁决与资源限制。
> - [全部基线官方复现与公平比较执行计划](PHASE1_ALL_BASELINES_EXECUTION_PLAN.md)：E00–E09 环境矩阵、B01–B09 baseline 执行步骤、R1/R2/R3 和 Gate 0–5。
>
> **执行顺序：先关闭审计中的 P0，再开始全部基线 R1；随后做 R2 与 R3。文档中的计划/验收条件不是已完成实验的声明。**
>
> **代码基线快照（三个 SHA 各有用途，禁止混用）：**
>
> | 用途 | SHA |
> |---|---|
> | Phase I 当前代码基线快照（含 v5 报告与运行数据） | `8d45d280a461eb37fcc560c5e7d68254e43a2e5b` |
> | Stage 4 v5 实验实际执行代码（v5 JSONL `git_sha` 字段） | `549fa7d8e782aaf6d5ce90fa5f53da1d89b22b48` |
> | 本 PR 创建时的旧 base（历史参照，不再作为起点） | `f2ab2fe84ac79ebeea45c4d8099d24f6d637795c` |
>
> 后续所有正式运行记录必须使用完整 40 位 commit SHA；`version` 标签与短 SHA 仅作辅助。

---

## 0. 本计划的最高优先级决定

CPU-CAST 当前最缺少的不是更多模型、更多 scheduler 功能或新的算法名称，而是对已有 strongest baselines 和 benchmark 的**忠实复现（paper-faithful reproduction）**、**同条件复评（harmonized re-evaluation）**和**独立严格分析（independent audit）**。

因此，研究分为三个阶段：

1. **Phase I — 复现与严格分析（必须首先完整完成）**：复现官方模型、官方代码、官方 benchmark、论文主结果；再将算法移植到共同 CPU 环境，以固定 query、qrels 和 candidate pool 做公平比较；审计所有差异、失败和测量偏差。
2. **Phase II — 新方法设计与验证**：只有 Phase I 的证据完整后，才研究 query-level ranking-risk control、adaptive depth、Top-K 区间决策等新方法。
3. **Phase III — 系统化在线服务与论文级评测**：只有新方法确实优于强基线后，才投资 deadline-aware scheduling、wavefront、tail-latency/SLO 等系统优化。

**不允许倒置顺序。** Phase I 不允许通过更换数据集、减少候选数、降低 baseline 预算、使用不同 qrels、或只选择对 CPU-CAST 有利的 query 子集来获得更好的比较结果。

### 0.1 为什么 Phase I 必须先做

Stage 4 v5（`infra/results/stage4_v5_report.md`，执行代码 `549fa7d8e782`）撤回了 v3 基于离线服务时间重放（measured_service_replay）得到的 wavefront 尾延迟恶化结论：那是重放模型的伪影，不是真实系统的行为。在 SciFact 的 30 请求、2 秒间隔负载剖面下，真实 wavefront 调度器相对真实 FIFO `SerialScheduler` 的 p50、p95、p99 三个分位点均更低（p50 约 −89%，p95/p99 约 −80%）。

但该结论必须连同其限制一起引用：

1. 实验只报告了 p50/p95/p99 三个分位点，不得写成"全分位胜出"；
2. 每次仅 30 个请求，单次 p99 接近样本最大值，三次重复未构成有说服力的置信区间——不满足本计划第 5.4 节"每负载点至少 10,000 次真实请求完成"的尾延迟证据标准；
3. 单数据集、单到达率、30 请求，不能据此宣称 H3（或任何调度优势）已被普遍验证。

当前更明确、且有同条件证据的未决问题是**策略质量缺口**：在同一批 30 个查询、同一候选集、同一计时边界上，active-set 固定阈值策略的 nDCG@10 为 0.3702，低于 `cascade_shallow` 的 0.6805。这仍是小规模 SciFact 样本上的结论，不应写成跨数据集的总体定论。

因此 Phase I 的首要任务是：验证并解释策略质量缺口及其成因，同时保留对调度收益的独立验证（扩大负载、多到达率、多长度桶后重新确认）。这个结论不能靠改名、加一个风险校准层或扩大 scheduler 功能来掩盖。

同时，相关方法已经有公开实现或可取得的 artifact：

- Xin et al. 的 Early Exiting MonoBERT；
- Petrov et al. 的 Shallow Cross-Encoders；
- Busolin et al. 的 SEE；
- Li et al. 的 CEC-Ranking；
- GAINER；
- PRISM；
- MICE / MICE-P；
- Pyserini、BEIR 等 retrieval / benchmark 工具。

项目现有 `scripts/clone_baselines.sh` 已包含 GAINER、CEC-Ranking、EarlyExiting MonoBERT、SEE、MICE、Fast-PLAID 和 Pyserini，但**缺少 Shallow Cross-Encoders 的官方仓库**；PRISM 的 artifact 不在 GitHub；当前 clone 使用 shallow clone 后取当时 HEAD，仍须建立正式 pin manifest。已有项目的 GAINER-style 训练脚本也不能直接标成官方 GAINER 复现。Phase I 首先解决这些问题。

---

## 1. Phase I 的范围、术语与复现等级

### 1.1 三种结果必须分开标记

不能把“读过代码”“运行过一份脚本”和“复现了论文”混在一起。所有 baseline 必须使用以下状态：

| 状态 | 含义 | 可否写成“已复现” |
|---|---|---|
| R0 — Artifact acquired | 取得官方代码/权重/数据，记录许可证、commit、hash；环境能够安装或失败原因明确 | 否 |
| R1 — Paper-faithful reproduction | 按原文数据、模型、训练/推理脚本、阈值、候选规模和指标运行；论文主要表格/曲线可重建，差异可解释 | 是，但必须注明硬件及偏差 |
| R2 — CPU-CAST common-port reproduction | 原算法语义保持不变，在 CPU-CAST 锁定的 CPU 环境和统一输入上实现/移植，并与官方输出或参考实现核对 | 可写“CPU port reproduced”，不得替代 R1 |
| R3 — Harmonized comparison | 所有方法使用相同 query IDs、qrels、候选池、计时边界、硬件和统计协议比较 | 这是 CPU-CAST 论文比较所必需的状态 |

**R1 与 R2 是两项不同任务。** 旧代码可以在原始环境复现成功，但无法在现代 CPU/PyTorch 环境直接运行；此时可以 R1 完成、R2 部分完成，不能用 R2 改写 R1 的复现结论。反过来，现代重写代码在 CPU 上跑通，也不等于原论文已复现。

**子状态标签（防止"部分证据被写成完整复现"）：** 每个 baseline 在 manifest 中必须使用以下精确状态之一，禁止把 checkpoint-only 或环境阻塞写成 R1-complete：

| 子状态 | 含义 |
|---|---|
| `R1-complete` | 原文数据/模型/脚本/指标下论文主结果已重建 |
| `R1-partial-checkpoint-only` | 仅用官方 checkpoint 完成 inference 复现；训练/完整协议未复现（如无 GPU 时的 Shallow CE gBCE） |
| `R1-blocked-environment` | 原文环境无法构建（如 MonoBERT 的 Python 3.7.7/PyTorch 1.3.1/CUDA 10.1）；失败原因已记录 |
| `R1-blocked-resource` | 原文硬件不可得（如 PRISM 的 Apple M2 / RTX 5070） |
| `R2-complete` / `R3-complete` | 对应端口/同条件比较已完成 |

注意：CPU 上"能否训练"取决于训练量与时间预算，不是数学上的不可能——资源受限必须按实测 dry-run 工时判定，而不是预先断言。

### 1.2 Baseline 优先级

- **P0：必须完成 R1 + R2 + R3**：Full-depth cross-encoder、shallow cross-encoder、Early Exiting MonoBERT、SEE、CEC-Ranking、GAINER、PRISM、CPU-CAST 现有 `cascade_shallow`。
- **P1：必须完成可执行的 R1 子集/已发布 checkpoint inference + R2/R3**：MICE（EMNLP 2026）；如资源允许，MICE-P 作为端到端 retrieval 扩展。
- **支持组件：必须固定版本并验证**：Pyserini/Anserini、BEIR、Sentence-Transformers、Fast-PLAID（仅为 MICE-P/late-interaction pipeline 服务时）。
- **消融/下界/上界**：Random、Current-Rank、Boundary、Fixed-Depth、Fixed-Residual、Patience、Oracle 等。GAINER 官方 notebook 本身包含这些控制组，应优先直接复现，而不是自行发明不同定义。

---

## 2. 官方代码仓库与 Artifact 清单（逐项复现）

以下地址均为本轮核对过的官方/作者仓库或 artifact 入口。Phase I 开始时必须再次 fetch 原始页面，核对仓库是否迁移、release 是否更新，再把**仓库 commit SHA**写入 manifest。README blob SHA 不是 git commit SHA，严禁混用。

### 2.1 核心算法基线

| ID | 工作 / 角色 | 官方代码或 artifact | Phase I 要求 | 重要注意事项 |
|---|---|---|---|---|
| B01 | Early Exiting BERT for Efficient Document Ranking (2020) | [castorini/earlyexiting-monobert](https://github.com/castorini/earlyexiting-monobert) | R1：MS MARCO Passage、ASNQ、TREC-DL 2019 的原始训练/评测；R2：迁移其正负非对称 exit 语义到 CPU-CAST | 官方 README 说明环境为 Python 3.7.7、PyTorch 1.3.1、CUDA 10.1；不能直接把现代 MiniLM pooler 试验称作原文复现 |
| B02 | Shallow Cross-Encoders for Low-Latency Retrieval (ECIR 2024) | [asash/shallow-cross-encoders](https://github.com/asash/shallow-cross-encoders) | R1：复现 TinyBERT/MiniBERT/SmallBERT + gBCE；R2/R3：在固定 CPU 环境上测 CPU latency-quality frontier | 这个仓库目前不在 CPU-CAST 的 `clone_baselines.sh` 中，必须补入；训练代码默认 negatives 数与论文主配置需核对 |
| B03 | SEE (SIGIR 2025) | [veneres/SEE-SIGIR25](https://github.com/veneres/SEE-SIGIR25) | R1：执行论文 RQ scripts、重建 figure/table；R2：将官方 SEE 规则应用于 CPU-CAST 固定候选池 | 官方 Docker 依赖 NVIDIA runtime；模型与额外 CSV 有 Google Drive 外部下载，必须另行记录 SHA-256 和来源 |
| B04 | Certified Error Control (EMNLP 2022) | [alexlimh/CEC-Ranking](https://github.com/alexlimh/CEC-Ranking) | R1：MS MARCO + Quora、RCPS、官方 `cec_prediction.sh`；R2/R3：同一候选池上的风险/质量/候选量对照 | 它控制两阶段候选剪枝风险，不是 layer-wise cross-encoder early-exit；不能把其 risk metric 改写成 Top-K ordering guarantee |
| B05 | GAINER (2026 preprint) | [ibrahimjamadi98/GAINER](https://github.com/ibrahimjamadi98/GAINER) | R1：完整运行 `GAINER_v3.ipynb` 和 `GAINER_v3_1.ipynb`；R2/R3：相同候选池、相同模型/预算口径做 CPU-CAST 对照 | 官方 README 标记 preprint；包含 preregistration、registered runs 和多种基线，必须保留原始配置，不可只摘取有利表格 |
| B06 | PRISM (EuroSys 2026) | [官方 artifact 页面](https://ipads.se.sjtu.edu.cn:1312/opensource/monolithic_forwarding_ae)；[Zenodo artifact record](https://zenodo.org/records/18809731) | R1：取得官方 artifact、执行可运行的原始 benchmark 并记录原设备限制；R2：按原文实现迁移到 CPU-CAST；R3：相同 query/candidate pool 比较 Top-K precision / ranking quality | **未找到可核验的 GitHub 官方仓库**。Zenodo 页面列出约 257.7 MB 的 tarball。CPU-CAST 现有 Prism 代码属于独立 reimplementation，不能标记为官方实现 |
| B07 | MICE (EMNLP 2026) | [xpmir/mice](https://github.com/xpmir/mice)；[HF model collection](https://huggingface.co/collections/xpmir/mice) | R1：优先执行官方公开 checkpoint inference 和官方 dry-run；有可用训练资源时复现核心 experiment config；R2/R3：MiniLM 兼容模型或发布 checkpoint CPU 推理 | MICE 改变 query/document interaction 架构，并非普通 early-exit；必须单列为结构性效率基线。官方仓库基于 `uv` + Experimaestro，完整训练可能需要 GPU/SLURM |
| B08 | MICE-P / end-to-end retrieval extension | [xpmir/mice](https://github.com/xpmir/mice)；[Fast-PLAID](https://github.com/lightonai/fast-plaid) | P1 扩展：单独复现 retrieval/indexing pipeline，不与 MICE reranking-only 结果混表 | MICE-P 是独立 pipeline / 预印本线；不要把 MICE 主论文和 MICE-P 结果合并成同一 baseline |

### 2.2 支持型官方仓库

| 组件 | 官方仓库 | 用途 | 必须记录的内容 |
|---|---|---|---|
| BEIR benchmark | [beir-cellar/beir](https://github.com/beir-cellar/beir) | 数据读取、标准 retrieval evaluation、metrics、runfile 工具 | commit、BEIR package version、dataset loader behavior、metric names |
| Sparse / dense first-stage retrieval | [castorini/pyserini](https://github.com/castorini/pyserini) | BM25、部分 dense retrieval 和 prebuilt index | commit、Java/JDK、Anserini/Lucene 版本、index 来源和参数 |
| Sentence-Transformers | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | cross-encoder 和 embedding model loading | commit/package、Transformers/Torch 兼容性、模型 revision |
| Fast-PLAID | [lightonai/fast-plaid](https://github.com/lightonai/fast-plaid) | MICE-P/ColBERT-style multi-vector indexing | 仅在 MICE-P 复现时列为核心依赖，记录 index config/quantization |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | model architecture / tokenizer / inference | commit/package、model loading flags、remote code |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | ORT FP32/INT8 backend comparison | version/build、Execution Provider、graph optimization、thread config |

### 2.3 当前 clone 脚本必须修正的事项

当前 `scripts/clone_baselines.sh` 已列出 GAINER、CEC-Ranking、EarlyExiting MonoBERT、SEE、MICE、Fast-PLAID 和 Pyserini，但尚有以下缺口：

1. 加入 **`asash/shallow-cross-encoders`**，它是 CPU 低延迟比较最重要的直接 baseline。
2. PRISM 不应添加虚假的 GitHub clone 地址；增加独立 artifact 下载步骤和 license/hash manifest。
3. 不得把 `git clone --depth 1` 的“当前 HEAD”当成可复现 pin。必须把最终 commit SHA 固定在 `manifests/baselines_manifest.json`；重跑时 checkout 同一 SHA。
4. manifest 中同时保存：`repo_url`、`git_commit`、`git_tag`（若有）、`git_dirty`、`diff_sha256`、`license`、`retrieved_at_utc`、`paper_doi`、`reproduction_status`。
5. 当前 `requirements.txt` 不包含 Shallow CE / SEE / PyTerrier / ir-datasets / RCPS / Experimaestro 的全部依赖。**不要把不同年代的 baseline 硬塞进同一套 pip 环境。** 每个官方项目使用独立 environment 或容器；CPU-CAST 的共同评测 harness 另建固定环境。

---

## 3. 每个核心 baseline 的详细复现流程

### 3.1 B01 — Early Exiting MonoBERT（Xin et al.）

**官方入口**：[castorini/earlyexiting-monobert](https://github.com/castorini/earlyexiting-monobert)

**原文机制必须保留：**

- 输入为 query-document pair，BERT 每一层后有独立分类器。
- 每个 pair 独立做早退判断。
- 对 positive / negative prediction 使用非对称阈值；negative 更容易早退，positive 通常需要更强信心才退出。
- 退出候选使用其退出 head 的 positive probability 排序。
- 统计平均 exit layer / 真实 early-exit 执行成本，不能只把所有层输出离线算完后把“平均层数”称为真实 wall-clock。

**原文数据和流程：**

1. MS MARCO Passage：下载官方 `triples.train.small.tar.gz`；运行 `convert_data.py`，生成论文所需的 unique training pairs（README 示例为 `train.uniq.416k.tsv`）。
2. 下载 `top1000.dev.tar.gz`，运行 `partition_eval.py dev`；官方实现按 500 queries 分区。
3. 下载 `collectionandqueries.tar.gz`，保留原始 collection、query 和 `qrels.dev.small.tsv`。
4. ASNQ：从 [alexa/wqa_tanda](https://github.com/alexa/wqa_tanda) / 官方数据链接下载，运行 `preprocess.py` 与 `partition_eval.py dev`。
5. TREC-DL 2019：获取官方 2019 passage qrels、MS MARCO test queries 和 collection，运行官方 `bm25.py` / Pyserini first-stage retrieval。
6. 按 README 建立独立 legacy env（Python 3.7.7、PyTorch 1.3.1、CUDA 10.1 及原依赖）。
7. 训练：`scripts/train.sh bert base DATASET all`，分别运行原文使用的 MS MARCO / ASNQ 设置。
8. 论文完整曲线：`scripts/eval.sh bert base DATASET all PARTITIONS`，再在 evaluation 目录运行 `direct_eval.py --each_layer` 和指定 positive/negative threshold 的评估。
9. 真正 early-exit：`scripts/eval_ee.sh bert base DATASET all PARTITIONS PC NC`，再运行 `direct_eval.py --sp_folder ...`。

**必须重建的结果：**

- MS MARCO Passage：MRR@10 与阈值/平均退出层的曲线。
- ASNQ：nDCG@10、MRR 和速度/平均退出层曲线。
- TREC-DL 2019：原论文的迁移评测。
- 每个 threshold 组合必须记录：阈值、每层退出率、正负退出分布、平均 exit layer、质量指标、实际耗时。

**CPU-CAST 第二套实验（R2/R3）：**

- 先在原始模型上验证实现；然后才实现 MiniLM-L6 / L2 对应 exit head 的 CPU port。
- 原模型的 BERTBASE 数值不能与 MiniLM 数值直接混在一张“复现论文结果”表里。
- CPU port 要保留非对称 positive/negative 退出规则，不能替换成 global top-k keep fraction。
- 同一 query 的候选必须在同一 candidate set 下比较；退出候选的 score 来源严格遵循方法定义。

### 3.2 B02 — Shallow Cross-Encoders（Petrov et al.）

**官方入口**：[asash/shallow-cross-encoders](https://github.com/asash/shallow-cross-encoders)

这项工作是 CPU-CAST 的**必需强基线**，优先级高于继续调当前 wavefront。它表明在给定 latency budget 下，浅层模型可以因为更低的每候选成本而处理更多候选，并超过深模型。

**官方方法复现步骤：**

1. 独立 clone 官方 repository，固定 commit。
2. 建立 PyTerrier 环境，安装 PyTerrier、PyTerrier-PISA、ir-datasets、ir-measures、TensorBoard、PyTorch、Transformers；设置 Java / `JAVA_HOME`。记录 JDK 版本。
3. 运行 `bm25ids2tensor.py`：为 MS MARCO trainset 产生 top-1000 candidate doc IDs，并预 tokenise。记录 query count、candidate count、截断长度和缓存 sha。
4. 按论文使用浅模型 backbone：`prajjwal1/bert-tiny`、`prajjwal1/bert-mini`、`prajjwal1/bert-small`；保存 Hugging Face immutable revision。
5. 训练使用官方 `train_shallow_crossencoder.py`，重点配置 gBCE 参数 `t=0.75`。**README 当前代码默认 negatives 为 16，而论文核心实验使用 128 negatives per positive。** 必须显式覆盖默认参数复现论文主设置，并附加 16/32/128 negatives sensitivity；不能拿默认 16 的结果冒充论文主结果。
6. 使用 `evaluate_tinybert.py` 对 TREC-DL 2019 / 2020、MS MARCO dev 的官方 evaluation protocol 评测；检查脚本中 checkpoint path 与模型 backbone，不要沿用错误/旧 checkpoint 路径。
7. 重建 latency-vs-nDCG curve，至少覆盖论文重点 latency budget，并报告 candidate count 从小到大的变化。
8. 额外记录 CPU 线程数、CPU 型号、batch size、tokenization boundary、是否计入 BM25 和 tokenizer。论文的端到端 latency 包含 BM25/tokenization 时，不能与只计模型 forward 的 CPU-CAST 数值直接比较。

**CPU-CAST 共同评测：**

- 把 TinyBERT-gBCE 当作独立完整 reranker，而不只是 `cascade_shallow` 第一阶段。
- 与 MiniLM-L6 full-depth、MiniLM-L2/L4、`cascade_shallow` 在相同 query/candidate pool 和相同计时边界下测量。
- 需要同时有 reranker-only（固定候选池）与 end-to-end（retrieval + tokenization + reranking）两种结果。
- 主图之一必须是同一个 CPU 预算下的 nDCG@10，而不是仅报告单 query forward speed。

### 3.3 B03 — SEE（Busolin et al., SIGIR 2025）

**官方入口**：[veneres/SEE-SIGIR25](https://github.com/veneres/SEE-SIGIR25)

**官方环境要点：** README 给出了 NVIDIA Docker runtime、GPU device、`~/.ir_datasets`、Hugging Face cache、Torch cache 和独立 `see-data` volume。它并不是一个纯 CPU 环境。GPU 资源不可用时，必须将“原始 GPU reproduction”标为受限，而不是跳过后称已完整复现。

**官方脚本应按顺序跑：**

1. `rq1.sh /data/rq1 <batch_size>`
2. `rq2.sh`
3. `rq3_run_all.sh`
4. `python src/04_bm25_beir_runs.py`
5. `rq4_run_all.sh`
6. `r5_run_all.sh`（按当前 README 命令原样执行，核对脚本名是否有版本差异）
7. `replicate-paper.ipynb` 重建原论文图表。

**模型与数据来源：**

- `veneres/monobert-msmarco`
- `veneres/monobert-asnq`
- `veneres/monoelectra-msmarco`
- `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `naver/trecdl22-crossencoder-electra`
- README 指向的 Google Drive 文件夹中的 CSV / 模型必须逐个下载、保存来源、文件名、大小、SHA-256、许可证/引用要求。

**算法复现要求：**

- 按论文的 similarity-based early-exit 规则计算 query-document token embedding 相似度；不能用简单的 query score top-k 或随机 prune 代替 SEE。
- 先执行作者设定的 layer/location 评测，再在 CPU-CAST 模型上做 CPU port。
- 分别检查候选 recall、ranking metric、退出数量、平均深度和真实耗时。
- 因 SEE 官方说明其最佳优势可在进入第一个 Transformer block 前触发，必须单独测这个入口；不要只对照某个后层 checkpoint。

**CPU-CAST 的关键公平性问题：** SEE 原文的预筛位置、输入 token、模型权重和候选集可能不同。报告必须同时给出 paper-faithful 和 common-pool 两组结果，不能只选一种对 CPU-CAST 有利的 protocol。

### 3.4 B04 — CEC-Ranking（Li et al., EMNLP 2022）

**官方入口**：[alexlimh/CEC-Ranking](https://github.com/alexlimh/CEC-Ranking)

**官方步骤：**

1. 克隆固定 commit，安装 repository 的 `requirement.txt`，单独按官方说明安装 Torch。
2. 安装并固定 [RCPS](https://github.com/aangelopoulos/rcps) 依赖。
3. 下载官方 MS MARCO 和 Quora 压缩数据，分别核对文件数、query 数、candidate 数、qrels 和 hash。
4. 运行 `bash cec_prediction.sh` 完成 calibration / prediction / test。
5. 重建论文要求的 risk target、confidence level、candidate set size、MRR@10 与 coverage 结果。
6. 检查随机种子和 calibration/test split 是否与论文实现一致；记录每次重复的阈值，而不是只留平均阈值。

**与 CPU-CAST 对照时：**

- CEC 的输入是 first-stage retrieval score / candidate threshold；CPU-CAST 的方法则可能在 cross-encoder layer 中继续或退出。
- 必须同时报告它对 candidate recall / MRR@10 的控制，以及 CPU-CAST 对 cross-encoder depth 的额外计算。
- 不可以把 CEC 的期望 ranking risk 直接等同于 per-query Top-K exact match。
- 如果将 CEC 适配到固定 candidate run，应明确标记为 “CEC adapted to CPU-CAST frozen candidate pool”，与原文复现分开。

### 3.5 B05 — GAINER（官方 GitHub / preregistered runs）

**官方入口**：[ibrahimjamadi98/GAINER](https://github.com/ibrahimjamadi98/GAINER)

官方 README 当前描述：

- `GAINER_v3.ipynb`：主 registered run，包含 H1–H3、BEIR 数据、BM25 pools、trajectory extraction、exit heads、gain rankers、baseline suite、leave-one-dataset-out driver、measured latency、statistics 和 figures。
- `GAINER_v3_1.ipynb`：H4–H6，含 calibrated uncertainty heads / value-of-computation；重用 v3 cache。
- `results/v3/`、`results/v3_1/`：有 preregistration、package/device/config manifest、per-query parquet、latency 和图表。
- `QUICK_TRIAL=True` 只用于 smoke test；**正式复现必须按完整配置运行，不能把 QUICK_TRIAL 当作主结果。**
- 官方 README 指出原复现流程是 Google Colab GPU runtime。若当前环境无 GPU，完整原始训练复现标记为 pending，不得改称 full reproduction。

**执行流程：**

1. Clone GAINER 官方 repo，锁 commit、requirements、notebook hash。
2. 先校验官方 preregistration JSON 的 SHA-256，确认运行脚本没有事先修改假设或配置。
3. 先运行 QUICK_TRIAL 检查环境、缓存路径和输出 schema；这次 run 存入 `smoke_test/`，不可混进正式结果。
4. 再运行 `GAINER_v3.ipynb` 的完整 registered run；运行完保存全部 raw parquet、训练产物、模型、环境文件、运行日志、GPU 设备和耗时。
5. 运行 `GAINER_v3_1.ipynb`，确保其读取同一个 v3 cache 且未重新生成不同 query/candidate pool。
6. 对七个公开 BEIR dataset、两个 backbone、所有 registered seeds 和预算点重建官方表格。
7. 官方 notebook 中的 Current Rank、Boundary、Random、Fixed Depth、Fixed Residual、Patience、SEE、Oracle 等基线逐一核对，不允许只复制最终 GAINER 表格。
8. 进行 independent result audit：抽样 query 对比 per-query depth allocation、gain target、最终 score 和实际 forward rows；验证 allocation 不使用尚未计算的未来层信息。
9. 将 GAINER 迁移到 CPU-CAST 共同 harness 时，保持模型、checkpoint、candidate pool、budget 定义和训练目标不变；CPU wall-clock 单独报告。

**关键区别：** CPU-CAST 的 `scripts/train_gainer_heads.py` 是 GAINER-style adaptation，不是自动等于官方 GAINER。只有当训练目标、trajectory sampling、query split、features、gain ranker、预算分配、数据和 evaluation 逐项对齐，才可以称作官方方法复现。

### 3.6 B06 — PRISM（官方 artifact 与独立 CPU port）

**来源：**

- 官方软件入口：[monolithic_forwarding_ae](https://ipads.se.sjtu.edu.cn:1312/opensource/monolithic_forwarding_ae)
- Artifact record：[Zenodo 18809731](https://zenodo.org/records/18809731)
- 论文 DOI：[10.1145/3767295.3803572](https://doi.org/10.1145/3767295.3803572)

**复现步骤：**

1. 下载官方 artifact 压缩包；当前 Zenodo 页面记录 tarball 约 257.7 MB。
2. 记录 Zenodo version、发布日期、文件名、文件长度、页面提供的 MD5；本地重新计算 SHA-256。
3. 解压至只读目录，保存未修改原始 artifact；另建 patch / CPU port 工作目录。
4. 完成原始 README 的依赖、数据、模型和 device setup；按论文设备/配置能运行的微基准先行。
5. 重建不同 CV threshold、cluster/pruning、Top-K precision / latency / memory 结果；保留所有 trigger 与 non-trigger 查询。
6. 原文在 Apple M2 / RTX 5070 的结果与 CPU VM 的结果必须分开表述；若缺少原设备，标记为 original-hardware reproduction incomplete。
7. 对 CPU-CAST 自己的 `calibrate_prism_cv.py` 做 independent implementation audit：聚类标签和 centers 必须来自同一次最优初始化；强制 trigger 测试必须有正触发率；不能只报告 threshold sweep 全部不触发的结果。

**公平比较注意：** PRISM 核心目标是 Top-K selection/precision，不等价于全列表排序指标。至少报告 Precision@K、Recall@K、Top-K set overlap、nDCG@10、full ranking correlation、实际计算量和延迟。任何只保留 Top-K、没有给出其余候选次序的结果，不应冒充完整 reranker 排名结果。

**R1/R2 资源边界（逐项判定，禁止混同）：** PRISM 原文硬件为 Apple M2 与 RTX 5070，本沙箱为 4 vCPU / 3.95 GiB cgroup。逐项状态如下：

| 实验 | 状态 | 依据 |
|---|---|---|
| 原始硬件复现（M2 / RTX 5070） | `R1-blocked-resource`（partial） | 硬件不可得；已按计划失败分类记录 |
| 本沙箱 0.6B reranker 实验 | `R2 = 可尝试` | 官方 Qwen3-Reranker-0.6B 为 BF16，权重约 1.19 GiB；转 FP32 约 2.22 GiB，在 3.95 GiB cgroup 内紧张但**不预先判死**——需计入激活、KV/cache、batch 与运行时开销后实测 |
| 8B 全量常驻加载 | `RESOURCE-blocked` | 官方 BF16 权重 5 个分片合计约 16.38 GB，远超内存上限 |
| 8B 流式/分层 CPU port | 单独评估 | PRISM 的 layer streaming 与 chunked execution 正是为降低峰值内存设计；不能仅凭权重总大小断言任何实现都绝对不可行 |

禁止事项：不把量化 checkpoint 或改变模型结构的结果冒充原论文模型复现；沙箱上的 0.6B 实验是 R2 port 或资源受限的 artifact 检查，**不是原硬件的 paper-faithful reproduction**。

### 3.7 B07 — MICE（EMNLP 2026）

**官方入口：**

- [xpmir/mice](https://github.com/xpmir/mice)
- [xpmir/mice Hugging Face collection](https://huggingface.co/collections/xpmir/mice)
- [xpmir/cross-encoders](https://github.com/xpmir/cross-encoders)
- [Experimaestro](https://github.com/experimaestro/experimaestro-python)

MICE 是通过 bottom contextualization 和 top interaction 分离来减少 cross-encoder interaction 成本的结构性方法，不是简单 early-exit。应将其作为相邻但强竞争的 architecture baseline。

**执行流程：**

1. Clone 官方 repo 并锁 commit；使用 `uv`，保存 `uv.lock`。
2. 按官方步骤 `uv sync`；先运行：
   `uv run experimaestro run experiments/emnlp26/mice_MiniLM_l8+3_bias.yaml --run-mode DRY_RUN`。
3. 检查 Experimaestro task DAG、外部依赖、launcher 配置、模型权重和数据路径。
4. 取得官方 HF released checkpoint，锁定 revision；先进行 inference parity 和 CPU latency smoke test。
5. 对主要配置进行复现：MiniLM MICE、Ettin-68M MICE、attention-bias ablation、layer split `l_X/l_Y` ablations。
6. 优先选用与 CPU-CAST 最接近的 MiniLM backbone 和 BEIR / TREC-DL 数据；其余论文配置尽可能由原始环境执行。
7. 另行检查 MICE-P：PLAID indexing、projection bottleneck、压缩和 retrieval pipeline；不能把 MICE-P 的端到端结果与 MICE reranking-only 结果放在同一列。
8. 完整训练受 GPU/launcher 限制时，至少完成 published checkpoint CPU inference，并在报告中把 training reproduction 标为 partial，而不是 full。

### 3.8 支持组件 — Pyserini / BEIR / Fast-PLAID

- Pyserini 用于可复现 BM25 / first-stage retrieval；需单独安装 Java 21 或对应官方版本，并固定 Anserini/Lucene。
- BEIR 用于标准 dataset loading 和 retrieval metrics；必须使用固定 commit/package，核对它对 query、qrels、missing documents 和 ties 的处理。
- Fast-PLAID 仅在 MICE-P/late-interaction retrieval 任务内作为索引和检索执行层；若只跑 cross-encoder reranking，不应把它加入所有 baseline 环境。

**Pyserini/Java 资源协议（本沙箱实测：无 java、无 sudo、no new privileges）：**

1. **首选路径**：用户目录下安装 Temurin/OpenJDK 21 tarball（无需系统 root），设置 `JAVA_HOME` 与 `PATH`；记录 JDK 发行版、版本、下载 URL 与安装包 SHA-256。Pyserini 官方当前说明以 Java 21 + Python 3.12 为基础；本沙箱 Python 为 3.12.15，Python 侧兼容，但安装后必须实测 `pyserini` import 与 Lucene 检索可运行。
2. **注意**：Pyserini 的 BM25 检索通过 Lucene/JVM 执行——装不装 Java 影响的是**所有** Pyserini 检索路径（包括用 prebuilt index 检索），不只是本地建索引。MS MARCO prebuilt index 只解决 MS MARCO Passage 的索引构建成本，**对 SciFact/NFCorpus 的候选生成没有替代作用**。
3. **降级路径（仅当 Java 确实无法安装）**：冻结已有的 SciFact/NFCorpus BM25 run，或用独立 Python BM25 实现（如 `rank_bm25`）生成 R2 候选池；后者必须明确标记为替代实现（`bm25-python-substitute`），不得称为 Pyserini 官方复现，且需报告与已知官方 run 的差异。
4. 无论哪条路径，BM25 参数（k1/b/analyzer/tokenizer）与 index/run 来源都必须写入 manifest。

---

## 4. Benchmark 清单与严格运行协议

### 4.1 Benchmark 分层

不同论文 benchmark 的 query、candidate set、qrels、指标和模型设置不同。Phase I 必须维护两套数据表：

**A. Paper-faithful benchmark：** 每篇论文都按其原始 benchmark、官方 candidate runs 和指标复现。不能强行把所有论文的原始测试改成 SciFact top-100。

**B. CPU-CAST harmonized benchmark：** 所有可比算法在相同 query IDs、相同 qrels、相同 candidate IDs、相同 max length、相同计时边界和相同 CPU 上重新运行。该集合是 CPU-CAST 新论文的主要横向对比。

### 4.2 核心 BEIR 数据集

官方代码和数据入口：

- BEIR GitHub：[beir-cellar/beir](https://github.com/beir-cellar/beir)
- HF 组织：[huggingface.co/BeIR](https://huggingface.co/BeIR)
- 数据集集合：[BeIR datasets](https://huggingface.co/datasets?search=BeIR)

当前 CPU-CAST manifest 已列出以下原始统计。这里的 `qrels_rows` 是 qrels 文件行数，不是唯一 query 数；Phase I 必须另外计算并记录 unique query IDs、无相关文档的 query 数、每个 query 的 candidate count 分布。

| Dataset | Corpus docs | Queries in corpus metadata | qrels rows (train/dev/test) | 主要作用 |
|---|---:|---:|---|---|
| SciFact | 5,183 | 1,109 | 919 / — / 339 | 主 smoke test 与 scientific fact retrieval |
| NFCorpus | 3,633 | 3,237 | 110,575 / 11,385 / 12,334 | biomedical domain、较丰富的训练 qrels |
| ArguAna | 8,674 | 1,406 | — / — / 1,406 | argument retrieval、zero-shot/OOD |
| FiQA | 57,638 | 6,648 | 14,166 / 1,238 / 1,706 | financial question answering retrieval |

**数据下载规则：**

1. 保留 HF 原始 parquet 和 qrels TSV，不仅保存转换后的 JSONL。
2. HF `main` 是可变引用，不是版本 pin。首次下载时通过 HF Hub 获取 dataset repo immutable commit SHA；在下载脚本中使用该 SHA 作为 revision。
3. qrels 所在的 `{dataset}-qrels` repository 也要单独 pin revision 和 SHA-256。
4. 原始文件、解压文件、转换后的 corpus/queries/qrels 都分别保存 SHA-256。
5. 下载脚本不得在发现目标文件存在时无条件跳过内容校验；必须先验证大小和 hash，不一致即报错。
6. 不得将 test qrels 用于 candidate generator、模型训练、exit threshold、风险校准、超参数选择或 query sampling。
7. ArguAna 当前 manifest 只有 test qrels；不能在 ArguAna test 上调阈值。要么使用其他 domain 上训练/校准的 zero-shot policy，要么把它作为纯 test-only OOD 数据集。

**BEIR 每个数据集的运行流程：**

1. 通过官方 BEIR loader 检查 corpus / queries / qrels。
2. 验证每个 qrels docid 在 corpus 中存在；报告缺失比例，不静默丢弃。
3. 对每个 split 输出 unique qid 数、qrels rows、相关等级分布、每 query relevant doc 数分布。
4. 生成 candidate pool 时对整个 split 的 queries 运行同一 retriever config。
5. 记录 top-100 / top-1000 candidate recall；确认 candidate pool 对 qrels 中 relevant docs 的覆盖上限。
6. 每个 reranker 只能对同一 frozen candidate pool 重新排序；保持 docid 映射不变。
7. 使用 BEIR 官方 evaluator 或等价的 `ir-measures` / `pytrec_eval` 复核 metrics；对核心表格至少双实现核验。

### 4.3 MS MARCO Passage 与 TREC Deep Learning

官方入口：

- [Microsoft MS MARCO repository / dataset documentation](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [Microsoft MS MARCO datasets](https://github.com/microsoft/msmarco/blob/master/Datasets.md)
- [TREC Deep Learning 2019](https://trec.nist.gov/data/deep/2019.html)
- [TREC Deep Learning 2020](https://trec.nist.gov/data/deep/2020.html)
- Early-exit paper 官方 data preparation 见 [castorini/earlyexiting-monobert](https://github.com/castorini/earlyexiting-monobert)
- Pyserini reproduction guides：[MS MARCO V1 Passage](https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html)

**必须保存的原始文件类型：**

- MS MARCO passage collection；
- `triples.train.small.tsv` 或论文要求的对应训练 triples；
- `top1000.dev` candidates；
- 官方 dev qrels；
- TREC-DL 2019 / 2020 queries；
- 对应官方 qrels；
- BM25 runfile 与其 index / parameter manifest。

**运行要求：**

1. 训练、dev、TREC-DL test 的 query ID 严格分开。
2. MS MARCO dev 用于模型/质量验证时，不得再作为最终 test 报告。
3. TREC-DL 2019 和 2020 qrels 只用于最终评测或预先明确的 paper-faithful protocol；绝不用于训练或 threshold selection。
4. MS MARCO Passage 的主要 ranking metric 是 MRR@10；TREC-DL 重点报告 nDCG@10，并按对应 track 的官方 qrels/metric 处理。
5. 对每个 query 输出完整 qid list、candidate count、relevant count、candidate recall。
6. BM25 结果需与官方 baseline run 或论文设置核验；不同 Lucene / analyzer / tokenization 会显著改变候选集。
7. 原始 reranker quality 和 candidate generation ceiling 分开报告；不能因候选集未召回 relevant doc 而将责任全部归给 reranker。

### 4.4 ASNQ（只对原论文 Early Exiting MonoBERT / 相应工作使用）

官方入口：[alexa/wqa_tanda](https://github.com/alexa/wqa_tanda)；官方数据下载地址在 EarlyExiting MonoBERT README 中。

执行步骤：

1. 下载原始 ASNQ archive，记录 URL、下载时间、SHA-256。
2. 使用官方 preprocess script，不得自行重写标签转换后直接声称原文复现。
3. 使用官方 qrel / partition script。
4. 核对 train/dev query/pair 数、positive/negative 分布、每 query candidate 数。
5. 按原文报 nDCG@10、MRR 及平均 exit depth。
6. 若把 ASNQ 纳入 CPU-CAST harmonized CPU comparison，必须先制作固定 candidate list，并对每种方法使用同一 list；不能直接将原始 ASNQ paper run 与 BEIR RRF pool 拼在同一比较中。

### 4.5 Candidate generation：必须有统一且冻结的候选集合

CPU-CAST 是两阶段 retrieval/reranking 项目。候选集质量会影响 reranker 上限，因此必须把 candidate generation 单独测量。

**至少生成以下候选 runs：**

| Run ID | Retrieval | 设置 | 用途 |
|---|---|---|---|
| CAND-BM25-1K | Pyserini/Anserini BM25 | 官方/论文参数；记录 tokenizer/analyzer、k1、b、index commit | 标准 lexical baseline |
| CAND-DENSE-1K | BGE-small dense | 固定 model revision、query/doc instruction、normalize、similarity | dense candidate baseline |
| CAND-RRF-1K | BM25 + dense RRF | 固定 RRF constant（例如 k=60，若沿用则写入 config） | hybrid candidate pool |
| CAND-RRF-100 | CAND-RRF-1K 截断至 100 | 保留原始 run 排名与分数 | 现有 Stage 4 兼容 |
| CAND-BM25-100 | BM25 top-100 | 相同 query set | candidate budget ablation |

**必须遵守：**

- 生成 runfile 后立即写入 SHA-256，后续所有 reranker 读同一个不可变 runfile。
- candidate run 至少包含 qid、docid、rank、score、run_name；建议 TREC run 格式：`qid Q0 docid rank score run_name`。
- 去重规则、并列分数排序规则、缺失文档处理必须固定。
- 每个候选池必须报告 candidate Recall@100 / Recall@1000；候选不足 100 的 query 单独统计。
- 不得为某个 baseline 单独使用更好的 candidate set，除非论文原始复现单独标记。
- 若使用 RRF，必须保存 BM25 run、dense run 和融合 run 三者；不得只保存最终融合文件而无法审计。

### 4.6 两套评测表必须并存

**Table A — Paper-faithful：** 每个 baseline 按其原论文数据、候选生成、模型、指标和硬件设置复现。

**Table B — Common CPU：** 在统一候选池上执行每个算法的 CPU port，所有方法共享：
- 同一 query list；
- 同一 qrels；
- 同一 corpus、tokenization policy 和 max length；
- 同一 candidate run；
- 同一 CPU quota、线程数和 backend；
- 同一 warm-up、计时边界和重复次数；
- 同一 metric implementation。

两张表不能合并成“一个数字”。若原论文使用 top-1000、CPU-CAST 使用 top-100，应明确显示它们不具备直接比较关系。

---

## 5. 共同 CPU 评测协议（R2 / R3）

### 5.1 目标硬件环境

当前 Stage 4 报告环境为 KVM VM，4 vCPU quota（cgroup `cpu.max=400000/100000`）、约 3.95 GiB memory；主要 inference baseline 使用 2 Torch intra-op threads。Phase I 必须保留该环境作为主要 CPU-CAST target，同时重新执行硬件探针，确保配额、内存和 CPU flags 在新 run 中记录。

每次记录：
- `/proc/cpuinfo`、`lscpu`、ISA flags；
- cgroup CPU quota / cpuset / memory limit；
- NUMA / cache 信息（若暴露）；
- Torch / ORT / OpenMP / MKL thread settings；
- CPU frequency policy / throttling（能访问时）；
- kernel / OS / container image digest；
- 运行时背景负载与同时运行的进程；
- 模型加载、warm-up、cache cold/warm 状态。

不能把当前 KVM 的结论泛化成“所有 CPU”。如果没有第二种真实 CPU 平台，论文只陈述该目标环境上的经验，不声称架构普适性。

### 5.2 时间边界

每个 benchmark 都需要以下至少三种测量边界：

1. **Reranker-only**：候选 pair tokens 已预构造，只测模型、compaction、scoring、排序。
2. **Model-service**：包括 tokenizer + pair construction + model inference + final sort。
3. **End-to-end retrieval**：包括 first-stage retrieval + candidate fetch + tokenizer + reranker + final ranking。

所有方法必须使用相同边界。预处理在某方法中计时、另一方法中排除属于无效比较。

冷启动和 warm inference 分开。冷启动包括模型加载、ORT graph optimization、index load、tokenizer 初始化等；主 latency 结果使用 warm service，冷启动结果单独报告。

### 5.3 运行矩阵

先复现论文原始 batch/length 设置，再运行 CPU-CAST 统一矩阵。统一矩阵至少覆盖：

| 维度 | 取值 | 必须观察 |
|---|---|---|
| Threads | 1 / 2 | latency、throughput、CPU saturation |
| Runtime | PyTorch FP32 / ORT FP32 / ORT INT8（可用时） | quality delta、model latency、RSS |
| Batch | 1 / 2 / 4；支持的 baseline 可加 8 | batching gain、padding、tail |
| Candidate count | 10 / 25 / 50 / 100 / 1000（paper-faithful 时） | quality、candidate recall、cost |
| Sequence length | 64 / 128 / 256；真实动态 padding 另测 | attention / FFN / memory effect |
| Query length | short / medium / long buckets | query-dependent cost |
| Document length | quartiles + fixed max-length ablation | truncation sensitivity |
| Concurrency | 1 / 2 clients | queueing / contention |
| Arrival process | burst / open-loop Poisson / closed-loop | p50/p95/p99、goodput |
| Precision | FP32 / INT8 | ranking stability、per-layer error |
| Compaction | none / fixed threshold / method-native | gather/scatter overhead |

### 5.4 延迟与资源指标

每个 run 至少输出：

- per-query latency p50 / p90 / p95 / p99 / mean；
- QPS / docs-per-second；
- query queue wait；
- tokenization / pair packing；
- embedding；
- each transformer layer 或 layer block；
- checkpoint scoring / exit head；
- gather / scatter / concat / index-copy；
- final scoring / sorting；
- CPU time（user/system）、CPU-seconds per query；
- peak RSS、model bytes、index bytes；
- batch size distribution、active rows distribution；
- request timeout / dropped request / error counts；
- 若硬件传感器真实可用，报告 energy；无传感器则不估算能耗。

**p99 的样本量：** 每个负载点建议至少收集 10,000 个 request completions，并做至少 5 次独立 run。小规模 qrels 数据集可重复回放 query arrival trace 来测服务延迟，但必须将重复请求与独立 query 数区分；不能把重复 query 当作新增独立质量样本。

### 5.5 统计协议

- 训练型 baseline 至少复现论文 seeds；CPU-CAST 新增对照至少 5 个 seeds。
- quality metrics 使用 query-level paired bootstrap，至少 10,000 resamples，报告 95% CI。
- baseline 和 CPU-CAST 使用相同 qid 和相同 candidate list，做 paired comparison。
- 需要时报告 paired randomization test / Wilcoxon signed-rank，并对多重比较做 Holm correction。
- 每数据集单独报告；跨数据集只额外报告宏平均和分布，不能用一个总平均掩盖 domain failure。
- 报告每 query distribution、worst decile、候选数分桶、文档长度分桶。
- latency run 的顺序应随机化或交替，避免某方法总在 CPU cache warm 或系统负载较低时运行。
- 对于 stochastic baselines（Random / GAINER training / initialization），记录 seed 和 per-query output。

---

## 6. Ranking output、metric 和 baseline 一致性审计

### 6.1 排名输出必须有统一 schema

每个 query 的 reranker output 保存：

- `qid`
- `docid`
- `raw_score`
- `rank`
- `exit_layer` 或 `depth`
- `was_pruned`
- `score_source`（full model / exit head / shallow model / fallback）
- `candidate_run_id`
- `model_revision`
- `baseline_id`

对于候选被早退/剪枝的方法，必须记录退出时分数的语义和 fallback。不能把一个未校准的 intermediate score 和 full-depth score 不加说明地混在一起，也不能把“只选出 Top-K 的集合”伪装为全列表排序。

### 6.2 统一指标与原论文指标

| Benchmark / 目标 | 核心指标 | 辅助指标 |
|---|---|---|
| BEIR retrieval/reranking | nDCG@10 | MAP@10/100、Recall@100/1000、Precision@10、MRR@10 |
| MS MARCO Passage | MRR@10 | Recall@100/1000、nDCG@10（明确协议） |
| TREC-DL 2019/2020 | nDCG@10 | MRR@10、MAP、Recall |
| ASNQ | nDCG@10、MRR | recall / average exit layer |
| CEC | 原文 expected risk、coverage、MRR@10 | candidate set size / recall |
| PRISM | Precision@K / Top-K set recall | nDCG@10、Top-K overlap、ranking correlation |
| Adaptive depth | qrels nDCG@10 regret | MRR regret、teacher Top-K agreement、depth/cost |
| Serving | p50/p95/p99 / QPS / SLO goodput | CPU-sec、RSS、queue wait |

必须将以下概念分开：
- **candidate recall**：first-stage 有没有找到相关文档；
- **reranker ranking quality**：同一 candidate set 中的排序质量；
- **teacher consistency**：是否与 full-depth model 的排名/Top-K 一致；
- **gold relevance quality**：相对于 qrels 的 nDCG/MRR；
- **Top-K set precision**：选出来的集合中有多少属于 full model 或 gold 的 Top-K。

它们不是同一种指标，不得互相代替。

---

## 7. 数据与模型版本锁定规范

当前数据/模型脚本使用 HF `main` revision，不能把分支名当不可变 pin。Phase I 必须修正版本管理。

### 7.1 Hugging Face model manifest

至少锁定以下模型：

| 用途 | Hugging Face repo id |
|---|---|
| 主要 full-depth ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 浅层主模型 | `cross-encoder/ms-marco-TinyBERT-L-2-v2` |
| 浅层/深度 ablation | `cross-encoder/ms-marco-MiniLM-L-2-v2`、`cross-encoder/ms-marco-MiniLM-L-4-v2` |
| 更深 cross-encoder 对照 | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| dense candidate generator | `BAAI/bge-small-en-v1.5` |
| SEE 论文模型 | `veneres/monobert-msmarco`、`veneres/monobert-asnq`、`veneres/monoelectra-msmarco`、`naver/trecdl22-crossencoder-electra` |
| MICE | `xpmir/mice` collection 对应 model IDs |

每个 repo 通过 HF Hub API 获取 immutable commit SHA。保存：
- repo id、revision SHA、下载时间；
- config/tokenizer/weights 文件名、大小、SHA-256；
- tokenizer fast/slow、special tokens、max position embeddings；
- model architecture、num layers、hidden size、classifier shape；
- license 和 model card；
- trust_remote_code 是否开启。

不能只记录 repo id 或文件总大小。

### 7.2 Dataset manifest

每个数据源保存：
- HF dataset repo id + immutable revision；
- 官方 URL / release name；
- raw file hash、converted file hash；
- corpus / query / qrels counts；
- split 的 unique qids、qrels rows、unique relevant docs；
- preprocess script commit / hash；
- text concatenation 方式（title + text 等）、截断 policy；
- qrels relevance mapping；
- license / citation。

### 7.3 Run manifest

每个实验的 JSON metadata 至少包括：

```json
{
  "schema_version": "phase1-v1",
  "run_id": "...",
  "baseline_id": "B02-shallow-gbce",
  "paper_doi": "...",
  "source_repo": "https://github.com/...",
  "source_commit": "...",
  "source_dirty": false,
  "source_diff_sha256": null,
  "environment_lock_sha256": "...",
  "container_digest": "...",
  "python": "...",
  "torch": "...",
  "transformers": "...",
  "onnxruntime": "...",
  "dataset_repo_revision": "...",
  "qrels_sha256": "...",
  "candidate_run_sha256": "...",
  "model_repo_revision": "...",
  "seed": 0,
  "cpu_model": "...",
  "cpu_flags": ["..."],
  "cgroup_cpu_quota": "...",
  "memory_limit_bytes": 0,
  "threads": 2,
  "runtime": "torch",
  "precision": "fp32",
  "batch_size": 1,
  "max_length": 256,
  "cli": "...",
  "start_utc": "...",
  "end_utc": "..."
}
```

任何没有 source commit、dataset hash、qrels hash、model revision 和完整 CLI 的 run 都不得进入主结果表。

---

## 8. 代码库应新增的 Phase I 目录与运行入口

建议使用以下结构，所有 raw artifacts append-only，禁止覆盖权威 run：

```
reproduction/
  README.md
  envs/
    cpu_cast_cpu.lock
    earlyexit_legacy.yml
    shallow_ce.yml
    see_docker/
    cec.yml
    mice_uv.lock
  baselines/
    earlyexit_monobert/
    shallow_cross_encoders/
    see/
    cec/
    gainer/
    prism/
    mice/
  datasets/
    download_beir.py
    download_msmarco.py
    download_asnq.py
    validate_dataset.py
    build_candidate_runs.py
  harness/
    run_reranker.py
    run_online_trace.py
    evaluate_runs.py
    compute_latency.py
    compare_outputs.py
  manifests/
    baselines.yaml
    datasets.json
    models.json
    environments.json
  configs/
    paper_faithful/
    common_cpu/
  reports/
    reproduction_matrix.csv
    phase1_reproduction_report.md
    discrepancy_log.md
artifacts/
  reproduction/
    <baseline>/<dataset>/<run_id>/
      run.json
      console.log
      metrics.json
      run.trec
      per_query.parquet
      latency.json
      hashes.txt
```

推荐命令接口（实际脚本名可按实现调整，但配置结构必须一致）：

```bash
# 验证数据与 qrels
python reproduction/datasets/validate_dataset.py --config configs/common_cpu/scifact.yaml

# 生成候选 run（每种 retriever 独立输出）
python reproduction/datasets/build_candidate_runs.py --config configs/common_cpu/scifact_candidates.yaml

# 先跑官方 reference / full-depth
python reproduction/harness/run_reranker.py --config configs/common_cpu/minilm_l6_full.yaml

# 再跑某一 baseline 的 CPU port
python reproduction/harness/run_reranker.py --config configs/common_cpu/see_minilm_l6.yaml

# 统一评估 quality
python reproduction/harness/evaluate_runs.py --run-dir artifacts/reproduction/...

# 检查官方输出与 CPU port 的逐 query 差异
python reproduction/harness/compare_outputs.py --reference ... --candidate ...
```

这些是接口规划，不是声称当前仓库已有上述文件。Phase I 的工程任务之一就是创建它们。

---

## 9. Phase I 执行顺序、交付物与验收标准

### Phase I-0 — Inventory / provenance freeze

**任务：**

1. 从干净 checkout 的 `8d45d280a461eb37fcc560c5e7d68254e43a2e5b`（当前代码基线快照，见文首 SHA 表）开始；v5 实验实际执行代码为 `549fa7d8e782aaf6d5ce90fa5f53da1d89b22b48`；`f2ab2fe84ac79ebeea45c4d8099d24f6d637795c` 仅为历史参照（PR 创建时的旧 base），不再作为起点。
2. 建立 baseline inventory：官方 repo / DOI / paper status / data / models / metric / original device / artifact access。
3. 修正 baseline clone 脚本，加入 shallow-cross-encoders；对 PRISM 添加官方 artifact 下载及 hash 流程。
4. 固定 repo commit、HF model revisions、dataset revisions、environment lock。
5. 对当前所有旧 run 做 provenance audit：无法恢复确切 code SHA / model revision / dataset hash 的记录列为 legacy / non-authoritative。
6. 生成一份 baseline priority matrix 和缺失资源矩阵（GPU、Java、内存、Apple M2/RTX 5070 等）。

**交付：**（统一使用现有 manifest 体系，不新建平行文件，避免双重事实来源）

- `manifests/baselines_manifest.json`：由现有 `manifests/baselines_manifest.txt` 迁移并结构化（不另建 `baselines.yaml`）。
- `manifests/models_manifest.json`：直接扩展现有文件（HF immutable commit SHA + 每文件 SHA-256 已在当前 tip 落地；注意 `f2ab2fe` 旧 base 中尚无这些字段）（不另建 `models.json`）。
- `manifests/datasets_manifest.json`：直接扩展现有文件（qrels SHA-256、candidate run hash；补齐 nfcorpus train.tsv 哈希）（不另建 `datasets.json`）。
- `manifests/environments.json`：新增（Python/依赖锁、CPU/cgroup 配置、JDK 来源与哈希）。
- `infra/results/discrepancy_log.md`：唯一差异登记处（原始 trace 继续放在 `artifacts/runs/`；不新建根目录 `reports/`）。

**初始 discrepancy log 条目（Phase I-0 时必须登记）：**

1. v5 运行记录（`artifacts/runs/bench_stage4_load_v5.jsonl`）的 `version` 字段写作 `v4`：脚本版本常量未随代码更新同步；结果以 `git_sha=549fa7d8e782aaf6d5ce90fa5f53da1d89b22b48` 追溯，不因 version 标签作废。
2. v4 的中断运行（`v4_partial_crashed.jsonl`、`v4_partial_oom.jsonl`：meta 解包崩溃、scatter-back 缺失、OOM kill、负延迟计时约定错误）已废弃，标记为 non-authoritative，仅作调试记录。
3. 历史记录中部分 `git_sha` 使用短 SHA 或缺失 dirty 标记：自本计划生效起，所有正式实验必须记录完整 40 位 SHA + `git_dirty` 标志 + 工作树 diff hash。

**验收：** 所有 baseline 的官方代码/模型/数据来源可追溯；所有无法获取项有明确阻塞原因和替代策略。

### Phase I-1 — Benchmark / candidate pipeline validation

**任务：**

1. 校验 SciFact、NFCorpus、FiQA、ArguAna 的 raw files 和 qrels。
2. 下载 MS MARCO Passage、TREC-DL 2019/2020、ASNQ 原文需要的数据。
3. 生成 BM25 / dense / RRF candidate runs。
4. 完成 qrels integrity / candidate recall / missing doc audit。
5. 锁定 train/dev/calibration/test query IDs；禁止 query overlap。
6. 选择一个 small smoke dataset（SciFact）和一个更丰富数据集（NFCorpus）验证 pipeline；再逐步扩展到全 harmonized benchmark。

**交付：**
- 所有数据原始文件和转换文件 hash；
- 每 dataset/split 的 qid / qrels / doc statistics；
- candidate run files；
- candidate Recall@100 / Recall@1000；
- retrieval reproducibility report。

**验收：** 数据、qrels 和候选 run 可从 clean checkout 重建；所有候选池冻结且可验证。测试集没有参与模型/阈值/候选融合选择。

### Phase I-2 — 官方 baseline reproduction

推荐按以下顺序完成，以优先验证最有可能改变论文结论的基线：

1. Full-depth HF cross-encoder + BEIR evaluator；
2. Shallow Cross-Encoders / gBCE；
3. `cascade_shallow`；
4. Early Exiting MonoBERT；
5. SEE；
6. CEC-Ranking；
7. GAINER；
8. PRISM；
9. MICE / MICE-P（按资源分级）。

**每个 baseline 的最小完成标准：**

- 官方 repo/commit / release / artifact hash 已保存；
- 原始依赖可装或不能装的失败原因已保存；
- 至少复现主要质量表和一项核心效率结果；
- 官方结果与本次重跑结果的对照表；
- 全部 query-level results / runfiles / metrics；
- R1/R2/R3 状态明确；
- 与 CPU-CAST 共同候选池对比；
- 不一致结果有 root-cause analysis，而不是只写“hardware difference”。

### Phase I-3 — Common CPU port 与同条件比较

所有算法在共同 CPU 环境下运行：

- 固定 2 threads 主设置，并有 1-thread sensitivity；
- 同 query、qrels 和 candidate run；
- 同 tokenizer / max length / padding / truncation；
- 同计时边界；
- 每个 baseline 既报告 quality 又报告 measured runtime；
- 区分 offline throughput、single-query latency 和 online request latency；
- 不用仿真 latency 替代 real latency。

对于原论文只能在 GPU/M2/RTX5070 环境运行的 baseline，明确分别标记：
- original-hardware reproduction；
- CPU-CAST port;
- missing original-hardware replication。

### Phase I-4 — Strict analysis / independent audit

每个 baseline 写单独 report，回答以下问题：

1. 算法在数学上做了什么？和论文描述是否一致？
2. 候选选择/退出决策依赖哪些已计算信息？有无未来信息泄漏？
3. 训练数据、calibration queries、test queries 是否严格分离？
4. 实际使用的模型和 paper model 是否完全相同？若不同，具体改变了什么？
5. 第一阶段 candidate recall 是否相同？
6. 排名分数是否来自 full model、exit head、similarity、cluster 还是 fallback？
7. reported metric 是否与原论文完全一致？
8. 原论文 latency 是否包括 retrieval / tokenizer / sorting？当前实现是否相同？
9. CPU 侧的主要瓶颈是 Transformer、gather/scatter、tokenization、batch fragmentation 还是 queue wait？
10. 质量退化集中于哪类 query、candidate count、document length 或 domain？
11. 与 CPU-CAST 当前实现的差异究竟是算法、代码、数据还是硬件造成的？
12. 该 baseline 是否构成新方法的真正有效竞争对手？

**独立核验：** 由不负责实现该 baseline 的另一轮审查脚本/研究人员从 clean checkout 复跑至少一个 dataset 和一个主要配置；结果与主 run 做逐 query 对比。

### Phase I-5 — Gate 1：停止/继续决策

以下项目必须全部满足，才允许进入 Phase II：

- [ ] P0 baselines 的官方代码或 artifact 全部取得并 pin；
- [ ] Shallow Cross-Encoders 官方训练和评测完成；
- [ ] Xin20、SEE、CEC、GAINER 的原始方法设置被准确理解且至少主结果重建；
- [ ] PRISM 官方 artifact 已尝试复现；如设备不可得，完整标记为受限并完成独立 CPU port；
- [ ] MICE 至少完成官方 released checkpoint inference 和主要 CPU port；训练资源缺失必须记录为 partial；
- [ ] BEIR / MS MARCO / TREC-DL / ASNQ 的 benchmark 入口和 qrels 核验完成；
- [ ] 所有 common comparison 的 query/candidate/qrels 一致；
- [ ] 不存在 test-query calibration / query leakage；
- [ ] 所有运行都有 git SHA、HF revision、dataset/qrels/candidate hashes 和完整 CLI；
- [ ] 主结果具备 query-level 95% CI；
- [ ] p95/p99 使用足够的真实 request samples，或明确不报告；
- [ ] 有完整的 baseline-vs-paper discrepancy report；
- [ ] 至少一个独立 clean-checkout rerun 已通过。

**未通过时的处理：** 继续修复复现或明确资源阻塞；不得提前开始 Phase II 并把未复现 baseline 排除在论文表格之外。

### Phase I-a — Early viability gate（早期可行性筛选；不是 Gate 1）

**定位与规则冲突的解决：** I-a 是 Gate 1 **之内**的早期筛选门，不是它的替代品，也不是缩短版的 Gate 1。它与"Phase I 完成前不进入 Phase II"纪律的关系如下：

- **I-a 通过**：允许开始**有限的 Phase II 原型探索**（风险控制/退出信号算法原型）；但不得形成论文级方法结论，不得宣称已建立完整 SOTA 比较，GAINER/PRISM/MICE/CEC 等竞争方法不得从论文评价中排除。
- **Gate 1 通过**：才允许形成正式的论文级方法结论与完整比较。
- **I-a 失败**：优先重新考虑算法方向，而不是继续堆 scheduler 功能；Gate 1 的其余复现工作照常推进。
- **Gate 1 未通过**：不得宣称新方法优于所有强基线。

**范围（最小证据集）：** Shallow Cross-Encoders（官方 checkpoint inference，标 `R1-partial-checkpoint-only`）、full-depth MiniLM-L6、`cascade_shallow`、SEE 官方规则；数据集仅 SciFact + NFCorpus；R2/R3 闭环。

**工时：** "约 2 周"仅为 timebox，不是工期预测。启动前必须先做一次 dry run，实测下载、推理、候选生成与评估耗时，再给出资源预算；dry run 结果记入 discrepancy log。

**验收条件（全部满足才算通过）：**

1. SciFact/NFCorpus 的 query IDs、qrels、候选 run 在任何实验前冻结，写入 manifest；candidate recall 单独报告。
2. 所有方法使用同一候选池、相同候选数量、相同 tokenizer/truncation 与 CPU 计时边界。
3. Shallow CE 与 SEE 必须使用可追溯的模型权重、退出层与规则配置（HF revision/文件 SHA、退出层编号、规则参数入 manifest）；不得只写"checkpoint inference"或"SEE rules"。
4. 报告 nDCG@10、MRR、candidate recall、CPU-seconds/query、真实延迟与 95% CI（query-level paired bootstrap）。请求完成样本不足 10,000 时，不得把 p99 当作可靠尾延迟结论。

**已知局限（必须同时写进 Gate A 的定义）：** I-a 只覆盖 2 个 BEIR 数据集，且不含 GAINER、CEC-Ranking、PRISM——因此 I-a 通过只能证明"方向值得继续投入"，不能证明"相对全部相关方法具有新颖性"。Gate A（算法有效性）的最终判定仍以 Gate 1 的完整 baseline 矩阵为准。

---

## 10. 质量复现容差与失败分类

“完美复现”不是要求跨硬件 bitwise latency 相等，而是要求算法、数据、代码、指标和证据链无歧义。Phase I 按下述方式判定。

### 10.1 数值复现标准

- 同一权重、同一输入、同一 runtime 的 full-depth score：优先要求最大绝对误差 `<=1e-4`；若 kernel / batch order 导致浮点差异，需报告误差分布和 rank swaps。
- 同一 paper-faithful protocol 的核心 ranking metric：目标差异不超过 0.01 absolute metric points，或落在论文报告的重复/置信区间内；超出时必须做根因排查。
- 不能仅以 nDCG/MRR 接近证明复现成功；还必须核对 query list、candidate IDs、score source、阈值、exit distribution 和计算量。
- Runtime 与论文数值通常不能跨硬件直接对齐；必须验证趋势、算法实际执行路径、测量边界，并将硬件差异写清楚。
- 任何超出容差的数据不得删除；必须保留并解释。

### 10.2 失败类型必须分类

- `DATA`：数据版本、qrels、query IDs、candidate list 不一致；
- `MODEL`：模型权重、tokenizer、revision、head architecture 不一致；
- `METHOD`：退出策略、阈值、预算、训练/损失与原文不同；
- `METRIC`：nDCG/MRR/Recall、ties、unjudged documents、relevance mapping 不一致；
- `ENV`：依赖、kernel、Java、CUDA、ORT provider 等版本冲突；
- `HARDWARE`：CPU/GPU、memory、thread、quota 不同；
- `TIMING`：计时区域、warm-up、tokenization、queueing、batching 不一致；
- `BUG`：实现错误或数据映射错误；
- `RESOURCE`：确实缺少 GPU / M2 / RTX 5070 / RAM / Java runtime 等资源。

每个失败项必须有：复现步骤、日志、影响范围、是否阻碍论文主张、下一步修复或降级策略。

---

## 11. Phase I 的交付清单（必须落盘）

1. **Baseline reproduction matrix**：每个 baseline 的 paper、official repo、commit、environment、datasets、metrics、R1/R2/R3 status。
2. **Dataset audit report**：每个 dataset / split 的 corpus/queries/qrels counts、hashes、missing IDs、candidate recall。
3. **Model audit report**：model revision、config、tokenizer、weight file hashes、license。
4. **Candidate run library**：BM25/dense/RRF top-100/top-1000 的不可变 TREC run files。
5. **Paper-faithful tables**：按原论文 protocol 重建的主要结果。
6. **Harmonized CPU tables**：所有方法同 candidate pool、同 CPU、同 timerange 的 nDCG/MRR/latency/CPU-seconds/RSS。
7. **Quality statistics**：per-query parquet、paired bootstrap CI、seed variance、domain breakdown。
8. **Latency statistics**：真实 request traces、p50/p95/p99、QPS、service/queue/preprocessing breakdown。
9. **Discrepancy log**：逐 baseline 对比官方论文值与本次复现值。
10. **Reproduction report**：严格、负面结果与限制均保留的完整分析。
11. **Clean rerun instructions**：一个新环境从零开始重现 Phase I 主要表格的脚本和命令。

---

## 12. Phase I 结束后才允许讨论的 Phase II / III

Phase I 交付前，不进入算法开发。通过 Gate 1 后，才根据严格比较结果选择最有证据支持的研究方向：

- 若 Shallow / MICE 在同预算下已占据 Pareto frontier，必须优先解释当前 adaptive-depth 方案为何仍值得存在；不可将弱 baseline 作为创新对象。
- 若 SEE / PRISM / GAINER 的质量—计算 frontier 已经覆盖当前策略，必须找出它们尚未解决且可实验验证的具体缺口。
- 若 risk control 只是把已有 CEC / CRC / Conformal@K 直接套用到 early exit，不能默认构成 ICML 级理论贡献。
- 若 wavefront 只降低 p50、提高 p95/p99，则先放弃“在线系统改进”主张，直到有真实 SLO/goodput 优势。
- 新算法只在 Phase I 观察到明确 failure mode、而且强基线未解决时才开发。

---

## 13. 必须阅读并在复现报告中引用的原始资料

### 论文与方法

- [Early Exiting BERT for Efficient Document Ranking (2020)](https://doi.org/10.18653/v1/2020.sustainlp-1.11)
- [Shallow Cross-Encoders for Low-Latency Retrieval (ECIR 2024)](https://doi.org/10.1007/978-3-031-56063-7_10)
- [Efficient Re-ranking with Cross-encoders via Early Exit (SIGIR 2025)](https://doi.org/10.1145/3726302.3729962)
- [Certified Error Control of Candidate Set Pruning (EMNLP 2022)](https://doi.org/10.18653/v1/2022.emnlp-main.23)
- [GAINER preprint / artifact](https://github.com/ibrahimjamadi98/GAINER)
- [PRISM (EuroSys 2026)](https://doi.org/10.1145/3767295.3803572)
- [MICE (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1736/)
- [BEIR (NeurIPS 2021 Datasets & Benchmarks)](https://openreview.net/forum?id=wCu6T5xFjeJ)
- [Resources for Brewing BEIR (SIGIR 2024)](https://dl.acm.org/doi/10.1145/3626772.3657862)

### Benchmark 与工具

- [BEIR GitHub](https://github.com/beir-cellar/beir)
- [Pyserini GitHub](https://github.com/castorini/pyserini)
- [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [TREC Deep Learning 2019](https://trec.nist.gov/data/deep/2019.html)
- [TREC Deep Learning 2020](https://trec.nist.gov/data/deep/2020.html)
- [ASNQ / TANDA](https://github.com/alexa/wqa_tanda)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)

---

## 最终项目纪律

**Phase I 的任务不是证明 CPU-CAST 是最好的，而是找出它相对于最强方法究竟还剩下什么可证明的优势。**

完成 Phase I 后，项目应该能够回答：

1. 每个 baseline 的官方方法到底是什么、实现在哪、如何运行？
2. 官方论文数据、qrels、candidate pools、模型和 metrics 能否完整重现？
3. 它们在 CPU-CAST 的统一 CPU / 同候选池设置下谁更好、差异来自哪里？
4. 当前 CPU-CAST 的哪一项优势在最强 baseline 下仍成立？
5. 哪些失败是算法本身、哪些是 runtime、哪些是数据或测量问题？
6. 哪一个尚未解决的研究问题值得进入下一阶段？

在这些问题有数据、脚本、commit、hash、逐 query 输出和统计证据回答之前，**不得进入 Stage 5，也不得将 Phase I 描述为“基本完成”。**
