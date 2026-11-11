# CPU-CAST Phase I-B：全部基线官方复现与公平比较执行计划

> 状态：下一阶段强制执行计划；不是实验结果  
> 前置：先通过 PHASE1_REPRODUCTION_AUDIT.md 的 P0 修复与环境验收  
> 目标：逐项完成所有核心 baseline 的 R1（paper-faithful reproduction）、R2（CPU-CAST common-port）、R3（harmonized comparison），并交付可独立复跑的证据包。

---

## 0. 目标与不可妥协的执行纪律

本阶段不新增 CPU-CAST 算法，不调整 scheduler 来追求更好数字。唯一目标是：**把相关方法和 benchmark 真实跑起来，弄清原论文到底做了什么，在原始环境复现论文结果，再在共同 CPU 环境做公平比较，并独立解释差异。**

全部基线必须纳入矩阵。任何基线若因设备/数据/软件原因无法完成，都要留下可验证 blocker，不能悄悄从论文表格移除。

三个复现等级独立登记：

- **R1 — paper-faithful**：原论文数据、模型、官方代码/配置、阈值、预算、指标和原始硬件/软件条件；论文关键结果可重建。
- **R2 — CPU-CAST port**：算法语义不变，在锁定 CPU 环境实现/移植；须与官方实现或原始中间输出做 parity。
- **R3 — harmonized comparison**：同一 qid、同一 qrels、同一 candidate pool、相同线上计时边界与统计协议下比较。

不得把 R2 proxy 写成 R1；不得把原论文结果表格直接与不同候选池/模型/数据预算的 CPU 数字放在一张表中直接排序。

## 1. 项目交付结构

项目根目录应形成以下结构：

- manifests/baselines_manifest.json
- manifests/environments.json
- manifests/models_manifest.json
- manifests/datasets_manifest.json
- manifests/candidate_runs/<dataset>_<retriever>_<split>.json
- manifests/runs/<baseline>/<R1|R2|R3>/<run_id>.json
- infra/env/<env_id>/{system.json,software-lock.txt,container-digest.txt,smoke.json}
- infra/results/discrepancy_log.md
- infra/results/reproduction_matrix.csv
- infra/results/baseline_reports/<baseline>.md
- infra/results/phase1b_final_report.md
- artifacts/reproduction/<baseline>/<R1|R2|R3>/<dataset>/<run_id>/{run_meta.json,raw_console.log,raw_metrics.json,per_query.parquet,per_candidate.parquet,run.trec,latency_requests.jsonl,model_manifest.json,data_manifest.json,environment_manifest.json,output_hashes.sha256}

所有结果都必须能从 run_meta.json 回溯：完整代码 SHA、依赖 lock、环境、原始数据、qrels、candidate run、模型 revision、参数、seed 和输出文件 hashes。

## 2. Gate 0：先把执行环境做成可审计矩阵

当前沙箱约为 4 vCPU quota、3.95 GiB cgroup memory、Python 3.12、PyTorch CPU-only、无 Java、无 NVIDIA GPU、无 sudo。它适合做部分 R2 CPU port，不足以单独完成所有 R1。

### 2.1 环境分组

| Env ID | 用途 | 必需基础 | 当前状态 | 验收 |
|---|---|---|---|---|
| E00 | CPU-CAST common CPU | Python 3.12.15；项目统一 Torch/Transformers/ORT；Java 21 另配 | 部分存在，lock 不完整 | clean rebuild、lock hash、smoke + env probe |
| E01 | MonoBERT legacy | 官方 README：Python 3.7.7、PyTorch 1.3.1、CUDA 10.1；独立 legacy container | 当前沙箱不满足 | 官方训练/评估脚本 smoke，CUDA/driver/model load |
| E02 | Shallow CE official | 官方 requirements；Torch 2.2.2、Transformers 4.39.3、Python Terrier 0.10.0、PyTerrier PISA、ir-datasets、ir-measures、TensorBoard；Java 环境 | 当前环境不匹配 | 数据预处理、一次 mini train/eval |
| E03 | SEE official | 官方 Dockerfile、NVIDIA runtime/GPU、数据挂载、HF/torch cache、官方依赖 | 无 GPU，不能在当前沙箱 R1 | rq1/rq2/rq3/rq4/rq5 smoke 至少一条，模型/数据齐全 |
| E04 | CEC official | 官方 requirement.txt + Torch + RCPS | 未隔离 | 官方 calibration/test pipeline 对 MS MARCO、Quora 可运行 |
| E05 | GAINER official | Python 3.13.15、Torch 2.11.0+cu130、Transformers 5.18.0、Sentence Transformers 6.1.0、NumPy 2.1.3、Pandas 2.2.3、bm25s 0.3.12；官方 run manifest 为 RTX PRO 6000 Blackwell | 当前 CPU 沙箱不满足 | 原 notebook 及 preregistered runs 可从头复跑 |
| E06 | PRISM official artifact | 按 Zenodo artifact 内说明构建；分别记录原 M2/RTX 5070 环境与 CPU port | 原始设备不可用 | artifact hash 验证、官方 benchmark 执行或明确 R1 blocked |
| E07 | MICE official | git clone --recurse-submodules；锁定 submodule；uv sync；Experimaestro | 未完成 | official config dry-run、released checkpoint inference、原训练任务可复跑 |
| E08 | MICE-P | MICE + Fast-PLAID 的独立 index/retrieval environment | 未建立 | end-to-end index/search/eval pipeline |
| E09 | Retrieval tools | Pyserini/Anserini、Java 21；BEIR/ir-datasets 独立数据工具 | Java 缺失 | Pyserini BM25 runfile 可重建并 SHA 匹配 |

**不得在同一个 Python 环境中安装全部 baseline 的依赖。** MonoBERT 的旧 PyTorch、SEE 的 GPU stack、GAINER 的新 CUDA stack 与项目 common CPU 环境必须隔离。

### 2.2 Java 安装

优先在用户目录下载 Temurin/OpenJDK 21 tarball，不需要 root；记录官方下载 URL、版本、SHA-256、解压路径和 java -version。设置 JAVA_HOME 和 PATH，并验证 Pyserini/Anserini 与 PyTerrier 能调用 Java。

如需要从源代码 build Anserini，额外验证 Maven 版本和 Anserini/Lucene commit。预建 MS MARCO index 只省去本地 index build，不会取消 Lucene JVM 依赖，也不构成 SciFact/NFCorpus 候选 run。

### 2.3 资源监控策略

- 单个重模型 baseline 一个进程；模型切换后完全退出进程，不把所有模型常驻内存。
- 记录 memory.current、memory.peak、memory.events、RSS、swap、CPU quota、CPU throttling、CPU steal、affinity。
- 为 3.95 GiB cgroup 留出至少 15–20% 安全余量；超出即停止该 run，保存 OOM/峰值 trace。
- GPU 任务记录显存峰值、GPU 型号/驱动/CUDA/cuDNN、功耗/利用率（可获得时）。
- 每次实验前后保存 environment JSON；不能只依赖 pip freeze。

### 2.4 Gate 0 通过条件

- 所有环境都有可复建 lock 或 container digest；
- Python/JDK/CUDA/GPU 版本和线程配置明确；
- CPU common env 和每个 baseline 环境互不污染；
- 所有模型和数据均可通过 SHA 校验；
- 至少一个 clean checkout 在目标环境里成功运行 smoke test；
- 原始设备不可用项明确列入 R1 resource blocker，而不是伪装成 complete。

## 3. Gate 1：数据与候选池冻结

### 3.1 数据集范围

| 数据系列 | 数据集 / split | 主要对应基线 |
|---|---|---|
| BEIR 主集合 | SciFact、NFCorpus、FiQA-2018、ArguAna、SCIDOCS、TREC-COVID、Touché-2020 | GAINER；SEE；MICE；R3 通用比较 |
| MS MARCO Passage | 官方 train triples、dev top-1000、collection/qrels | MonoBERT；Shallow CE；CEC；SEE |
| TREC Deep Learning | TREC-DL 2019、2020 passage queries/qrels | MonoBERT；SEE；通用 IR 评估 |
| ASNQ | 官方 TANDA/ASNQ train/dev | MonoBERT |
| Quora | CEC 官方数据包及 split | CEC-Ranking |
| PRISM 专用 | 官方 artifact 所依赖数据/模型/权重 | PRISM R1 |
| MICE/MICE-P | MICE 官方配置使用的 in-domain/OOD collections 和 index | MICE / MICE-P |

GAINER 官方 preregistered v3 run 覆盖 7 个 BEIR 数据集：SciFact、FiQA、NFCorpus、ArguAna、SCIDOCS、TREC-COVID、Touché-2020；backbones 为 Ettin-68M 与 MiniLM-L12；5 seeds；top-k=50；max length=384；budget={0.25,0.35,0.5,0.65,0.8,1.0}。不得用当前本地的单 dataset / MiniLM-L6 / 256 token 实验替代这一设置。

### 3.2 固定原始数据与转换

每个数据集都要：
1. 固定源仓库完整 commit SHA；对 MS MARCO、ASNQ、Quora、TREC-DL 记录官方下载 URL、release/date 和 SHA-256。
2. 保存原始 archive/parquet/TSV，不要只保留转换文件。
3. 转换后 corpus、queries、qrels 再分别计算 SHA-256。
4. 记录唯一 docid/qid 数、重复数、缺失引用、qrels relevance distribution、split counts。
5. 验证 train/dev/calibration/test query IDs 不重叠；官方论文若有固定 split，就保留其原始 protocol，不自行更改。
6. 用 clean download 测试数据仓库是否可重建；缓存命中一定要校验哈希。

### 3.3 Candidate pool 协议

首先为每个 dataset 构建并归档 **official first-stage run**（优先用论文规定的检索器/参数和 Pyserini/官方 runfile）。每条 run 记录：
- Retriever code/commit、index SHA、analyzer/tokenizer、k1/b、stopwords/stemming、top-K；
- qid-set SHA、TREC runfile SHA、每 query candidate count；
- candidate Recall@10/100/1000、macro/micro 定义、missing qrels/doc report。

然后额外构建 common candidate pool：
- BM25 official run；
- Dense run（固定模型 revision、query/passage encoding instructions、normalization、batch、precision）；
- RRF run（固定 k 与 tie-breaking）；
- top-1000、top-100 等不同池大小作为独立条件。

项目当前 rank_bm25 whitespace BM25 只可保留为 proxy，不得伪装成 Pyserini/Anserini run。对每个候选池计算 Recall@K；低召回池不能用 reranker nDCG 直接比较模型能力而不同时报告 candidate ceiling。

**所有方法共享同一候选 doc IDs，但模型必须使用自己的 tokenizer。** 不同 tokenizer 不应强行共用 token IDs；统一的是原始 query/document text、候选集、明确的最大文本/截断规则和计时边界，模型自身 tokenizer 作为方法组成记录。另报 native-paper tokenization setting。

### 3.4 Gate 1 验收

- 所有源文件与转换后数据校验通过；
- 所有 official / common candidate runs 有 SHA；
- candidate recall 审计完成；
- qid split、训练/校准/测试隔离经过脚本验证；
- 同一 query 的候选 doc IDs 在 R3 各方法间完全一致；
- tie-break 规则确定且跨平台可重复。

## 4. Gate 2：逐基线 R1 官方复现

每个 baseline 单独建立 reproduction card：论文版本、官方仓库、commit、环境、数据、模型、脚本、配置、表格编号、原始输出路径、差异、R1 判定人。每条任务结束必须产出 baseline report。

### B01 — Early Exiting MonoBERT / Xin20

官方入口：castorini/earlyexiting-monobert。

R1 步骤：
1. 按 README 建 Python 3.7.7 / PyTorch 1.3.1 / CUDA 10.1 legacy env；保存依赖与 driver 信息。
2. 获取 MS MARCO Passage 的 triples train small、unique 416k training data、top1000.dev、collection/qrels；按官方脚本划分 evaluation partitions。
3. 下载并核验 ASNQ 官方数据，执行官方 preprocess/partition。
4. 下载 TREC-DL 2019 官方 qrels、MS MARCO 2019 queries 和 collection；按 README 用官方 BM25 pipeline。
5. 按 scripts/train.sh bert base DATASET all 复现 MS MARCO、ASNQ 训练；TREC-DL 按原文使用 MS MARCO model。
6. 按 scripts/eval_ee.sh 与 scripts/eval.sh，扫正负阈值、保存各层输出、重建 direct_eval.py 表格。
7. 对照论文的 quality、平均 exit layer、latency 和不同阈值曲线。

R2/R3：
- 在 MiniLM/BERT common CPU runtime 上复刻正负不对称退出语义；
- 对同一模型层输出逐样本与官方/legacy reference 比较；
- 报告 modern transformer port 与官方模型结构差异，绝不可称为 Xin20 R1。

### B02 — Shallow Cross-Encoders (ECIR 2024)

官方入口：asash/shallow-cross-encoders。

R1 步骤：
1. 锁定官方 commit；创建独立 E02 环境；安装官方 requirements。
2. 用官方 bm25ids2tensor.py 构建 MS MARCO train 的 top-1000 candidate tensors；锁定 ir-datasets/PyTerrier/PISA 和 Java 版本。
3. 复现论文规定的 backbone（prajjwal BERT tiny/mini/small 等）、训练数据、negative sampling 与 gBCE；README 默认 negs=16、建议 t=0.75，必须逐项对照论文主实验配置，确认主实验是否应覆盖 README 默认值，不能盲用默认设置。
4. 完成官方训练、validation、checkpoint selection、多个 seed（按论文报告设置）。
5. 用 evaluate_tinybert.py 按论文评测脚本重建质量和 latency 表格；保存训练日志、checkpoint SHA、tensorized data SHA。
6. 若官方 release 有已训练 checkpoint，则 checkpoint inference 单独记为 R1-partial-checkpoint-only，不能替代训练复现。

R2/R3：
- 使用已发布浅层 checkpoint 和重新训练 checkpoint 分开比较；
- common pool 上比较 TinyBERT-only、Mini/MicroBERT、cascade_shallow 和 full-depth；
- 计入第一阶段筛选、tokenization、selected-doc retokenization、第二阶段排序全部真实成本。

### B03 — SEE (SIGIR 2025)

官方入口：veneres/SEE-SIGIR25。

R1 步骤：
1. 固定官方 repo commit、Dockerfile、模型版本、Google Drive 外部数据和 CSV 的 SHA-256；
2. 在有 NVIDIA runtime 的机器 build 官方 Docker image，保存 image digest、CUDA/driver/GPU、挂载的 ir_datasets/HF/torch cache 和共享内存配置；
3. 按 README 执行 rq1.sh、rq2.sh、rq3_run_all.sh、src/04_bm25_beir_runs.py、rq4_run_all.sh、r5_run_all.sh；
4. 执行 replicate-paper.ipynb 重建表格/图；
5. 核对 README 列出的模型，包括 monobert、monobert-asnq、monoelectra-msmarco、MiniLM-L12、trecdl22 Electra，以及外部 CSV/数据；
6. 遵守仓库 citation license，归档下载来源和 SHA。

R2/R3：
- 抽取其真实 exit rule / similarity feature / checkpoint schedule；在 common CPU 上重写后与官方输出对齐；
- 当前 see_token 只是项目 proxy，不能作为 SEE 的 R2 parity 结果。

### B04 — CEC-Ranking (EMNLP 2022)

官方入口：alexlimh/CEC-Ranking。

R1 步骤：
1. 独立环境安装官方 requirement.txt 和 Torch；固定 RCPS repo SHA；
2. 下载官方 MS MARCO 与 Quora archives，记录 SHA，检查 split 文件；
3. 执行 cec_prediction.sh；复现 calibration/test split、risk control、candidate set pruning、coverage/risk 与质量统计；
4. 重建论文主要 risk/coverage 曲线和表格；
5. 对 RCPS 参数、风险水平、随机种子和 test coverage 做核验。

R2/R3：
- 同候选 pool 实现同一 RCPS 逻辑；
- 重点比较 candidate pruning 的 coverage/error risk，不把它误写成 layer-wise cross-encoder early-exit，也不以 Top-K ordering guarantee 冒充其原始保证。

### B05 — GAINER (2026 preprint / registered runs)

官方入口：ibrahimjamadi98/GAINER。

R1 必须使用官方 GAINER_v3.ipynb 与 GAINER_v3_1.ipynb，且保留官方 preregistration：
- v3 preregistered SHA-256：e77241073ba22c17b36151adcffacaf3e7f47ff36b3fd38cbebae413026b4746
- v3.1 preregistered SHA-256：1f4050281d4dbf8a27543aa6672f7295ea86880db50d1aecea2818ec52ef11b1

必须严格复现：
- 7 个 BEIR 数据集；
- 两个 backbone：Ettin-68M 和 MiniLM-L12；
- 5 seeds（0–4）；
- TOP_K=50、MAX_LENGTH=384；
- budgets 0.25/0.35/0.5/0.65/0.8/1.0；
- main config：DAGGER training、target mix、balanced sampling；
- 官方所有对照：Current Rank、Boundary、Random、Fixed depth、FixedResidual、Patience、SEE、Oracle；
- v3.1 calibrated uncertainty/VOC family；
- 保存 preregistration、run manifest、raw result files、per-query Parquet、measured latency、100,000 permutations、10,000 bootstrap，以及所有表格与图。

官方 README 明确 QUICK_TRIAL=true 只做两数据集、单 seed smoke test；它不能用于正式复现。正式 run 使用 QUICK_TRIAL=false。先将原始 registered outputs 与复跑结果逐文件比较，再单独分析漂移。

R2/R3：
- 先修复本项目的 features_flat / features_at normalization mismatch；
- 使用相同的测量结果计算真实 quality/latency；
- 每个实际运行 query 保存 scores、exit layers、feature vectors；
- 与官方方法做候选顺序、决策和计算预算的逐 query parity；
- 如果当前 CPU 无法训练原配置，使用远程 GPU R1，CPU 只承担 R2/R3 推理。

### B06 — PRISM (EuroSys 2026)

官方入口：作者 artifact 页面和 Zenodo record 18809731。当前 CPU-CAST PRISM 代码是 reimplementation，不能作为官方 artifact run。

R1 步骤：
1. 下载官方 artifact，记录 Zenodo version、tarball size、MD5 与自行计算的 SHA-256；
2. 保存论文/appendix、官方 artifact README、脚本、checkpoint 来源及许可证；
3. 按官方 artifact 原本的 device、model sizes、chunk/streaming、top-K precision 设置执行；
4. 重建论文表格，特别分别核对 Top-K Precision/Recall、selection cost、memory peak、latency；
5. 对无法获取 M2/RTX 5070 的情况记录 R1_BLOCKED_RESOURCE，不能将 CPU 结果冒充原硬件 R1。

R2/R3：
- 先按原代码结构完成 0.6B CPU/内存路径；
- 8B 全量常驻模型在 3.95 GiB cgroup 中不可行；但 PRISM 的 streaming/layer-loading 本就是降低峰值内存的路径，所以 8B streaming 必须独立评估，不能仅凭模型总权重大小断言完全不可能；
- 区分 FP32/BF16/INT8 checkpoint，任何量化变体单独标记，不得冒充原模型；
- Top-K selection 方法必须单独报告 Precision@K、Recall@K、Top-K overlap 和完整排序指标，不得将只输出集合的方法假称 full reranker。

### B07 — MICE (EMNLP 2026)

官方入口：xpmir/mice。

R1 步骤：
1. 固定主 repo commit 和所有 recursive submodule commit；
2. 执行官方 git clone --recurse-submodules 或对已有 clone 执行 git submodule update --init --recursive；
3. 使用仓库 uv lock 同步依赖，保存 uv version 和 lock SHA；
4. 按官方 Experimaestro 配置执行 MiniLM-L12 的 training/evaluation config；
5. 对已发布 checkpoint 运行 inference；分别核对 in-domain 与 OOD 数据、MICE 与 standard cross-encoder 的 quality/latency；
6. 保存 experiment config、checkpoint SHA、tokenizer/model config、submodule SHA、per-query results。

R2/R3：
- CPU port 必须保持 MICE interaction architecture、layer/interaction masking 与 score semantics；
- 与 full cross-encoder、shallow CE、SEE 同候选池比较；
- 不把 MICE reranking 与 MICE-P end-to-end retrieval 合并。

### B08 — MICE-P / Fast-PLAID

独立 retrieval extension，不与 MICE reranking-only 合表。

R1：
- 核验 MICE-P 原论文/补充材料、index format、document encoder、Fast-PLAID version、index build flags、retrieval parameters；
- 若原始可用 artifact/config 有缺失，标明 partial，不自行补造论文配置；
- 重建端到端 retrieval results 与 index/storage/build-time。

R2/R3：
- 固定语料、collection split、index SHA、query set 和 top-K；
- 对比 retrieval recall、index size、build time、query latency、reranking quality；
- 使用 Fast-PLAID 独立 lock，避免 Pyserini/FAISS/JVM stack 互相污染。

### B09 — Retrieval / data support baselines

- Pyserini/Anserini BM25：Java 21、Lucene version、index SHA、analyzer/k1/b；优先官方 prebuilt index 复现 MS MARCO。
- BEIR evaluator：锁定 package commit/version 与 metric definitions。
- BGE/dense retriever：锁定模型 commit、embedding instruction、normalization、precision、FAISS/index version。
- RRF：固定 k、source ranks、tie-break 和 missing-document 处理。
- rank_bm25 proxy 作为补充对照，不替代官方 BM25。

## 5. Gate 3：R2 CPU-CAST common-port

在 R1 原始结果可取得后，按每个 baseline 建立方法保持不变的 CPU port：

1. 使用同一 query text、document text、candidate doc IDs 和 relevance qrels；
2. tokenizer 按各模型官方 tokenizer 执行，token IDs 不跨模型复用；
3. 固定并报告输入 truncation budget，另提供 paper-native max length 与 common-length 两种设置；
4. 输出每个候选的 score/rank、decision/exit layer、实际 forward layers/tokens；
5. 对于可对齐的 reference case 做逐候选输出和 exit decision parity；
6. 记录 CPU threads、OMP/MKL/ORT threads、affinity、batching、sequence bucket、RSS、cgroup memory、CPU throttling；
7. latency 分为 preprocessing、model forward、selection/compaction、queue wait 和 end-to-end；
8. 模型加载时间与 warm request time 分开。

R2 的成功标准是“语义一致且有逐 query parity 证据”，不是“能在 CPU 上 import/运行”。

## 6. Gate 4：R3 公平比较与统计

### 6.1 两张主表

- **Table A — Paper-faithful**：每个 baseline 在原数据、原候选/召回、原模型/训练/阈值/硬件设置下复现，报告与论文差异。
- **Table B — Harmonized CPU**：统一 query IDs、qrels、candidate pool、文本截断预算、计时边界和硬件。每个模型使用自身 tokenizer；同一个 candidate doc ID 集必须一致。

不同 baseline 的目标不一样，不能强行只用一个 nDCG 指标排名：
- full ranking: nDCG@10、MRR@10、MAP、Recall@K；
- early exit / reranking: nDCG loss、teacher Top-K agreement、平均层当量、候选执行比例；
- CEC: risk/coverage、candidate set size、pruning validity；
- PRISM: Precision@K、Recall@K、Top-K overlap、selection cost；
- MICE-P: retrieval recall、index size、query latency；
- system: p50/p90/p95/p99、QPS/goodput、CPU-seconds/query、RSS、queue wait、failure/timeout。

### 6.2 统计协议

- 每个训练方法尽可能 5 seeds；若原文规定 seeds，优先按原文。
- Quality confidence interval 采用 paired query bootstrap（至少 10,000 draws）；训练 seed 作为上层因素，不把 query 与 seed 错当作完全独立重复。
- 对 GAINER 的正式假设检验保留官方 paired sign-flip/randomization、Holm correction 和官方 preregistration。
- query 子集固定 qid hash；指标逐 query 保存，支持重算。
- 每个 workload latency 至少 5 次独立运行；每个负载点有 10,000 个真实请求完成样本才报告 p99。未达到就降级报告 p50/p90 或不报告 p99。
- 不把重复同一批 30 个 query 当作 10,000 个独立请求；请求负载必须重新采样/扩展并保存实际 arrival trace。
- 对实验运行顺序随机化；记录 CPU 负载、后台进程、热身次数、温度/频率（可用时）。

## 7. 交付顺序与阶段退出条件

### Work Package 0 — P0 修复与回归（阻断后续所有比较）

交付：
- SerialScheduler 正确性修复 + unit tests；
- GAINER feature normalization / measured quality 修复 + sim/measured parity；
- cascade timing boundary 修复；
- provenance logger v2；
- data/model/cache hash validation；
- environment probe v2；
- old results status register。

退出条件：clean checkout 下测试全通过；受影响算法与 reference parity 通过；无 dirty run 可混入主表。

### Work Package 1 — 环境与原始资产

交付：
- E00–E09 环境锁；
- GPU/M2/Java 资源可用矩阵；
- 所有 baseline repo/artifact commit + SHA；
- 官方数据档案与模型 manifest；
- 每个 baseline reproduction card。

退出条件：所有 R1 需要的资源可用，或每项 blocker 被明确标注并指定替代资源。没有 GPU 时不能把 GPU R1 勾为完成。

### Work Package 2 — Benchmark/data freeze

交付：
- 七个 BEIR datasets + MS MARCO + TREC-DL + ASNQ + Quora 的 source/transformed hashes；
- official candidate runs；
- common candidate runs；
- qid split manifest；
- candidate recall report；
- qrels/data integrity report。

退出条件：所有 R3 输入有 immutable SHA、split 无泄漏、候选池可 clean rebuild。

### Work Package 3 — 官方 R1 baseline reproduction

建议并行执行，但每条 baseline 使用独立环境和 reproduction card：
1. Shallow Cross-Encoders；
2. Early Exiting MonoBERT；
3. SEE；
4. CEC-Ranking；
5. GAINER v3 + v3.1；
6. PRISM official artifact；
7. MICE；
8. MICE-P；
9. Pyserini/BEIR retrieval baseline。

每个完成后立即出一份 baseline report，并由不负责运行该 baseline 的 reviewer 从 clean checkout 复跑至少一个 dataset/config。

### Work Package 4 — R2 CPU common ports

只对 R1 已明确其算法语义、输入、输出和关键超参的 baseline 开始。R1 blocked 的也可做 R2，但必须保留 blocked label。

### Work Package 5 — R3 unified comparison

所有 baseline 在共同 query/candidate/qrels/CPU 环境上运行，生成成套 per-query results、quality/latency frontier、CI 和 discrepancy analysis。

### Work Package 6 — Gate 5 final audit

最终所有下列项完成后 Phase I 才通过：
- 每个核心 baseline R1/R2/R3 状态有明确证据；
- 所有 P0 findings 已关闭；
- 所有官方论文主表重建或逐项说明 resource blocker；
- 所有 R3 comparison 输入与计时边界统一；
- 所有运行都有完整 provenance 和 per-query raw outputs；
- 独立 clean-checkout rerun 通过；
- Stage 4 v5 invalid SerialScheduler comparison 已由修复后的配对运行取代；
- 旧报告已标注 retracted/superseded；
- 由独立审查者完成最终 discrepancy audit。

## 8. 工作量与资源现实

这是多篇论文、多套环境、多种数据和多个 GPU 实验的复现项目，不是一个单一 benchmark 脚本任务。按当前资源条件，完整 R1 不可能只靠 4-vCPU/3.95-GiB CPU 沙箱完成。

可按依赖并行：
- CPU 侧：数据校验、R2 harness、CEC 部分流程、CPU inference、统计审计；
- GPU 侧：MonoBERT legacy、Shallow CE 训练、SEE、GAINER、MICE/PRISM 相关训练/原始硬件实验；
- Java 侧：Pyserini、PyTerrier/PISA；
- 独立审查侧：clean checkout、hash 校验、结果复算。

在 GPU/Java 资源及时可用的前提下，可将执行拆成约 6–10 周的 timebox；这不是完成保证，任何 legacy dependency、官方外部数据缺失、GPU 排队或原始设备不可用都会改变周期。若没有原始设备，报告必须保留 R1 blocker；不能用时间压力降低复现等级。

## 9. 最终必须交付的文件

- manifests/baselines_manifest.json：repo/artifact/source/license/commit/status；
- manifests/environments.json：环境锁、container digest、资源、probe hash；
- manifests/models_manifest.json：model commit revision + 每文件 SHA；
- manifests/datasets_manifest.json：数据源 revision + 原始/转换文件 SHA；
- manifests/candidate_runs/：每个 candidate run 的参数/qid/doc/hash；
- infra/results/reproduction_matrix.csv：R0/R1/R2/R3 状态、run IDs、blockers；
- infra/results/discrepancy_log.md：paper-vs-reproduction 差异；
- infra/results/baseline_reports/*.md：每个 baseline 的单项报告；
- infra/results/phase1b_final_report.md：综合结论和 Gate 5 决策；
- artifacts/reproduction/：完整 raw logs、per-query、per-candidate、latency traces、output hashes。

## 10. 资源依赖与官方入口

- [MonoBERT early-exiting](https://github.com/castorini/earlyexiting-monobert)
- [Shallow Cross-Encoders](https://github.com/asash/shallow-cross-encoders)
- [SEE SIGIR'25](https://github.com/veneres/SEE-SIGIR25)
- [CEC-Ranking](https://github.com/alexlimh/CEC-Ranking)
- [GAINER](https://github.com/ibrahimjamadi98/GAINER)
- [PRISM EuroSys artifact](https://zenodo.org/records/18809731)
- [MICE](https://github.com/xpmir/mice)
- [Fast-PLAID](https://github.com/lightonai/fast-plaid)
- [Pyserini installation](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
- [BEIR](https://github.com/beir-cellar/beir)
