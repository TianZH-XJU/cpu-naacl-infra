"""Cross-query wavefront scheduling for active-set reranking (plan section 10.3).

Candidates from different in-flight queries that are at the SAME transformer
layer are executed together in one batch. Per-row next_layer pointers track
progress; checkpoints apply an exit policy.

v2 (post-review f0873cd4):
- online mode: requests carry an arrival offset; the scheduler activates them
  when the run clock passes their arrival (real staggered-arrival serving)
- request-level traces: every request records submission time, first-layer
  time, completion time, and a per-layer execution log
- inline final scoring: rows that survive the last layer are scored inline
  right after the last layer, not in a separate final pass
- pooler signal: checkpoints can score via ranker.score_hidden (the real
  Stage-4 path) or via GAINER-style exit heads
- max_batch overflow is tested explicitly (no lost or double-executed rows)

Correctness invariants (tested in scripts/test_wavefront.py):
- full-depth wavefront == per-query active-set execution (same policy)
- pooler-path wavefront == per-query active-set execution (same policy)
- batch overflow: no lost rows, no double execution, next_layer consistent
"""
import time

import torch


class Request:
    def __init__(self, qid, input_ids, attention_mask, token_type_ids,
                 arrival=0.0):
        self.qid = qid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.n = input_ids.shape[0]
        self.arrival = float(arrival)   # seconds offset from run start
        self.submit_time = None         # wall clock when activated
        self.first_exec_time = None     # wall clock of first layer execution
        self.finish_time = None         # wall clock when last row completed
        self.layer_log = []             # [(layer, rows, exec_rows, scored_rows, t_ms)]
        self.done = None
        self.next_layer = None
        self.exit_layer = None
        self.scores = None
        self.hidden = None
        self.ext_mask = None

    # rows still needing execution, in execution order
    def active_rows(self):
        return torch.nonzero(~self.done, as_tuple=True)[0]


class WavefrontScheduler:
    def __init__(self, ranker, exit_heads=None, keep_frac=0.5, keep_fn=None,
                 checkpoints=(1, 3), max_batch=256, signal="head"):
        self.ranker = ranker
        self.exit_heads = exit_heads or []
        self.keep_frac = keep_frac
        self.keep_fn = keep_fn
        self.checkpoints = sorted(checkpoints)
        self.max_batch = max_batch
        self.signal = signal
        self.layer_events = []

    # -- embedding ----------------------------------------------------------
    def _embed(self, requests):
        L = self.ranker.num_layers
        seq_lens = {r.input_ids.shape[1] for r in requests}
        if len(seq_lens) > 1:
            raise ValueError(
                f"wavefront scheduler requires a single length bucket, got seq lens {sorted(seq_lens)}"
            )
        for r in requests:
            r.hidden, r.ext_mask = self.ranker.embed(
                r.input_ids, r.attention_mask, r.token_type_ids)
            r.done = torch.zeros(r.n, dtype=torch.bool)
            r.next_layer = torch.zeros(r.n, dtype=torch.long)
            r.exit_layer = torch.full((r.n,), L, dtype=torch.long)
            r.scores = torch.zeros(r.n)

    def _final_score_undone(self, requests):
        with torch.inference_mode():
            for r in requests:
                idx = torch.nonzero(~r.done, as_tuple=True)[0]
                if idx.numel():
                    r.scores[idx] = self.ranker.score_hidden(r.hidden[idx])
                    r.done[idx] = True

    # -- main loop ----------------------------------------------------------
    def run(self, requests, online=False):
        t0 = time.perf_counter()
        L = self.ranker.num_layers
        # reset per-run timing state (stale finish_time/submit_time from a
        # previous run would produce negative latencies — review P0-5)
        for r in requests:
            r.submit_time = None
            r.first_exec_time = None
            r.finish_time = None
            r.layer_log = []
        pending = list(requests)
        active = []
        if not online:
            self._embed(pending)
            active = pending
            pending = []
        with torch.inference_mode():
            while True:
                if online:
                    now = time.perf_counter() - t0
                    newly = [r for r in pending if r.arrival <= now]
                    for r in newly:
                        self._embed([r])
                        r.submit_time = now
                        active.append(r)
                    pending = [r for r in pending if r.arrival > now]
                ready = {}
                for r in active:
                    idx = torch.nonzero(~r.done, as_tuple=True)[0]
                    for row in idx.tolist():
                        nl = int(r.next_layer[row])
                        if nl < L:
                            ready.setdefault(nl, []).append((r, row))
                if not ready:
                    if online and pending:
                        nxt = min(r.arrival for r in pending)
                        wait = nxt - (time.perf_counter() - t0)
                        if wait > 0:
                            time.sleep(wait)
                        continue
                    break
                layer = min(ready, key=lambda k: (-len(ready[k]), k))
                rows = ready[layer][: self.max_batch]
                by_req = {}
                for r, row in rows:
                    by_req.setdefault(r.qid, (r, []))[1].append(row)
                h_parts, m_parts, meta = [], [], []
                for r, row_list in by_req.values():
                    t_idx = torch.tensor(row_list, dtype=torch.long)
                    h_parts.append(r.hidden.index_select(0, t_idx))
                    m_parts.append(r.ext_mask.index_select(0, t_idx))
                    meta.append((r, row_list))
                h = torch.cat(h_parts, dim=0)
                m = torch.cat(m_parts, dim=0)
                t_layer0 = time.perf_counter()
                h = self.ranker.forward_layer(h, m, layer)
                dt = time.perf_counter() - t_layer0
                off = 0
                for r, row_list in meta:
                    t_idx = torch.tensor(row_list, dtype=torch.long)
                    r.hidden = r.hidden.index_copy(0, t_idx, h[off:off + len(row_list)])
                    if r.first_exec_time is None:
                        r.first_exec_time = t_layer0
                    off += len(row_list)
                self.layer_events.append((layer, len(rows), len(by_req), dt * 1000.0))

                if layer in self.checkpoints and (self.exit_heads or self.signal == "pooler"):
                    stage = self.checkpoints.index(layer)
                    if self.signal == "pooler":
                        eh = self.ranker.score_hidden(h).reshape(-1)
                    else:
                        eh = self.exit_heads[stage](h[:, 0, :].float()).reshape(-1)
                    if self.keep_fn is not None:
                        keep_rows = self.keep_fn(eh, layer)
                    else:
                        k = max(1, int(round(self.keep_frac * eh.numel())))
                        keep_rows = eh >= torch.topk(eh, k).values[-1]
                else:
                    keep_rows = torch.ones(h.shape[0], dtype=torch.bool)
                    eh = None

                off = 0
                for r, row_list in meta:
                    t_idx = torch.tensor(row_list, dtype=torch.long)
                    seg_eh = eh[off:off + len(row_list)] if eh is not None else None
                    kl = keep_rows[off:off + len(row_list)]
                    off += len(row_list)
                    t_keep = t_idx[kl]
                    t_exit = t_idx[~kl]
                    if seg_eh is not None and t_exit.numel():
                        r.exit_layer[t_exit] = layer + 1
                        r.scores[t_exit] = seg_eh[~kl]
                        r.done[t_exit] = True
                    r.next_layer[t_keep] = layer + 1
                    if not self.checkpoints or layer >= self.checkpoints[-1]:
                        r.exit_layer[t_keep] = L

                if layer == L - 1:
                    off = 0
                    for r, row_list in meta:
                        t_idx = torch.tensor(row_list, dtype=torch.long)
                        kl = keep_rows[off:off + len(row_list)]
                        off += len(row_list)
                        idx = t_idx[kl]
                        if idx.numel():
                            r.scores[idx] = self.ranker.score_hidden(r.hidden[idx])
                            r.done[idx] = True

                # per-request ledger: rows executed, rows scored, timings
                scored = len(rows) if (layer in self.checkpoints
                                       and (self.exit_heads or self.signal == "pooler")) else 0
                for r, row_list in meta:
                    r.layer_log.append((layer, len(row_list), len(row_list),
                                        scored if layer in self.checkpoints else 0,
                                        dt * 1000.0))

                now = time.perf_counter()
                for r, _ in meta:
                    if r.finish_time is None and bool(r.done.all()):
                        r.finish_time = now
        self._final_score_undone(requests)
        return {r.qid: r.scores.clone() for r in requests}
