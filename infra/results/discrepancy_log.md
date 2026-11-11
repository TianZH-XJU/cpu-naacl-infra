# CPU-CAST 初始 discrepancy log（Phase I-B）

> 状态：初始审计登记；下列条目均未因本文件创建而自动关闭。  
> 记录原则：保留原始 run 和 hash，不删除旧产物；新结果通过 replacement run 关联旧记录。

| ID | 严重性 | 对象 | 问题 | 受影响证据 | 当前状态 | 关闭条件 |
|---|---|---|---|---|---|---|
| D001 | P0 | SerialScheduler | 逐层 forward 后未将 hidden state 写回请求；下一层从 embedding hidden 开始 | Stage 4 v5 real_serial_online 与 Serial/Wavefront latency comparison | OPEN | 修复并增加全深度/early-exit parity tests；同 trace 重新配对运行；旧 v5 对照标记 invalid comparator |
| D002 | P0 | SerialScheduler | non-compacted 路径的 active-local finished mask 被用于索引全量 cur，分数映射空间不一致 | Stage 4 v5 SerialScheduler scores/exit decisions | OPEN | 非连续 active-row 单测通过；逐候选 score 与 reference 一致 |
| D003 | P0 | GAINER port | training features_at 与 measured features_flat 的 rank fraction denominator 不一致 | bench_stage2b_gainer / qsplit GAINER-style results | OPEN | 统一 feature schema/transform；measured outputs directly produce quality; simulated/measured per-query parity |
| D004 | P0 | Dataset download | HF resolve/main 可变；cache 只检查文件非空；下载脚本不生成/校验 qrels SHA；当前 manifest 虽有部分事后补录 qrels SHA，但缺 source revision 与转换文件 SHA；qrels 宽泛异常被报告成 split 不存在 | dataset manifests、all dependent candidate runs | OPEN | pin immutable source revisions; verify cache SHA; hash raw and transformed files; classify split/network/checksum failures |
| D005 | P0 | Candidate retrieval | 当前 BM25 是 rank_bm25 whitespace proxy，未固定 Lucene analyzer / params；zero-score docs filtered | all candidate run manifests and R3 results | OPEN | official Pyserini/Anserini runs with fixed index/analyzer/params; proxy clearly separated; runfile SHA recorded |
| D006 | P0 | Provenance | generic logger lacks complete git/data/model/env/seed provenance; Stage 4 records only short SHA and incomplete diff hash | Stage 2/2b/3/4 historical and future logs | OPEN | logger v2 writes 40-char SHA, staged/unstaged/untracked fingerprints, lock/input/output hashes and full CLI |
| D007 | P1 | Query sampling | Stage 2/2b/4 use first-N qids in file/dict order; no frozen seeded qid manifest | small-sample quality and latency results | OPEN | predeclared qid/split manifests, seeded sampling, qid hashes and query-stratum report |
| D008 | P1 | Tail latency | Stage 4 v5 uses 30 requests per repetition x 3; p99 is not a stable estimate | v5 p50/p95/p99 claims | OPEN | >=10,000 real completions per load point, >=5 independent runs; otherwise report p50/p90 or omit p99 |
| D009 | P1 | cascade_shallow timing | model loading, initial tokenization, selected re-tokenization and per-query timers have different boundaries | Stage 4 cascade_shallow p50/total vs active-set latency | OPEN | separate cold-start/warm end-to-end/model-only measurements; equivalent timing boundaries |
| D010 | P1 | metrics | pytrec_eval aggregate Recall is macro; helper recall_at_k is micro; naming does not distinguish | candidate recall and ranking metric reports | OPEN | explicit macro/micro names and toy-qrels metric unit tests |
| D011 | P1 | raw outputs | Stage 2 report says per-query details are in bench_stage2.jsonl, but file only contains aggregate rows | Stage 2 bootstrap/independent re-computation | OPEN | save per-query/per-candidate raw tables; summaries reproducible from raw tables |
| D012 | P1 | historical GAINER results | stage2b_stage3_report has older qsplit result values and is not clearly superseded | GAINER results table | OPEN | mark superseded/non-authoritative and link replacement run, code SHA, data/model manifest |
| D013 | P1 | hardware probe | probe_hardware virtualization test is invalid; psutil memory is host-visible not cgroup effective; CPU tests may exceed quota | environment comparisons and resource claims | OPEN | unified probe records cgroup cpu/memory/cpuset/affinity/throttling and target-process counters |
| D014 | P1 | data loader | duplicate doc/query/qrels keys silently overwrite; qrels header is unconditionally discarded | all training/evaluation datasets | OPEN | duplicate detection, schema-aware header, join integrity and validation report |
| D015 | P1 | environment locks | requirements contain unpinned dependencies; no baseline-specific immutable environment locks | all official baseline reproduction claims | OPEN | lock files/container digests per baseline and clean rebuild smoke tests |

## 已撤回/降级的历史结论

1. Stage 4 v3 的 measured_service_replay 尾延迟恶化结论已由项目 v5 报告撤回；不得重新引用为当前事实。
2. Stage 4 v5 的 SerialScheduler 与 Wavefront 相对 latency comparison 因 D001/D002 标记为 **INVALID_COMPARATOR_PENDING_RERUN**。v5 原始 trace 仅保留为调试证据。
3. Stage 2b 的旧 GAINER 数字在 qsplit run 后标记 **SUPERSEDED**；且当前 measured route 另有 D003，因此不能作为官方 GAINER 对比。
4. Stage 2 中 see_token、untrained score/budget proxy、Prism cluster reimplementation 只能作为项目 proxy；不能标记为 SEE/GAINER/PRISM 官方复现。

每项关闭必须更新状态、关闭提交、修复测试、replacement run ID 和输出 hashes。禁止只改 Markdown 状态文字而没有可核验证据。
