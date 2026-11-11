"""Cross-query wavefront scheduling for active-set reranking (plan section 10.3).

Candidates from different in-flight queries that are at the SAME transformer
layer are executed together in one batch. Per-row next_layer pointers track
progress; checkpoints apply an exit policy.

v3 (post-review 899e580d):
- online mode: requests carry an arrival offset; the scheduler activates them
  when the run clock passes their arrival (real staggered-arrival serving)
- length buckets are validated UP FRONT for ALL requests (online included),
  not lazily at embed time — review P1-3
- structured event trace: every scheduler event records an event_id, the
  participating request ids, per-request and batch row counts, scored rows,
  compaction state and a gather/forward/scatter timing split (review P1-4);
  per-request layer_log entries reference their event_id so the real schedule
  can be rebuilt from the raw log
- inline final scoring: rows that survive the last layer are scored inline
  right after the last layer, not in a separate final pass
- pooler signal: checkpoints can score via ranker.score_hidden (the real
  Stage-4 path) or via GAINER-style exit heads
- max_batch overflow is tested explicitly (no lost or double-executed rows)
- SerialScheduler: a real one-worker FIFO online scheduler (sleep-until-
  arrival, completion-driven) with the same exit policy and the same
  per-layer timing split — review P0-2

Correctness invariants (tested in scripts/test_wavefront.py):
- full-depth wavefront == per-query active-set execution (same policy)
- pooler-path wavefront == per-query active-set execution (same policy)
- batch overflow: no lost rows, no double execution, next_layer consistent
- SerialScheduler (scripts/test_serial_scheduler.py): full-depth output ==
  forward_full; fixed-threshold path == per-query active-set reference on
  BOTH the compacted and non-compacted paths, including non-contiguous
  active rows and no-survivor / all-survivor / single-candidate edges;
  Serial and Wavefront agree per candidate under the same policy
"""
import time

import torch


def _check_lengths(requests):
    """Reject mixed length buckets UP FRONT, before any request arrives.

    Online mode embeds lazily per request, so a per-request check would never
    see the cross-request conflict; validate the whole set at run() start.
    """
    seq_lens = {int(r.input_ids.shape[1]) for r in requests}
    if len(seq_lens) > 1:
        raise ValueError(
            f"scheduler requires a single length bucket, got seq lens {sorted(seq_lens)}"
        )


class Request:
    def __init__(self, qid, input_ids, attention_mask, token_type_ids,
                 arrival=0.0):
        self.qid = qid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.n = input_ids.shape[0]
        self.arrival = float(arrival)   # seconds offset from run start
        # All three timing fields are ABSOLUTE time.perf_counter() values;
        # the caller converts to run-relative ms by subtracting its own t0
        # (single convention across WavefrontScheduler and SerialScheduler).
        self.submit_time = None         # wall clock when activated
        self.first_exec_time = None     # wall clock of first execution
        self.finish_time = None         # wall clock when last row completed
        self.layer_log = []             # structured per-layer entries (see run())
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
        # P1-3: validate the length bucket for EVERY request up front —
        # online requests have not arrived yet but must still be checked.
        _check_lengths(requests)
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
                        t_e0 = time.perf_counter()
                        self._embed([r])
                        r.submit_time = time.perf_counter()
                        r.first_exec_time = time.perf_counter()
                        r.embed_ms = (time.perf_counter() - t_e0) * 1000.0
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
                event_id = len(self.layer_events)
                event_start_clock = time.perf_counter()

                # -- gather (per request, timed) --
                t_g0 = time.perf_counter()
                h_parts, m_parts, meta = [], [], []
                for r, row_list in by_req.values():
                    t_i0 = time.perf_counter()
                    t_idx = torch.tensor(row_list, dtype=torch.long)
                    h_parts.append(r.hidden.index_select(0, t_idx))
                    m_parts.append(r.ext_mask.index_select(0, t_idx))
                    meta.append((r, row_list,
                                 (time.perf_counter() - t_i0) * 1000.0))
                h = torch.cat(h_parts, dim=0)
                m = torch.cat(m_parts, dim=0)
                gather_ms = (time.perf_counter() - t_g0) * 1000.0

                # -- forward --
                t_f0 = time.perf_counter()
                h = self.ranker.forward_layer(h, m, layer)
                forward_ms = (time.perf_counter() - t_f0) * 1000.0

                compacted = len(rows) < sum(r.active_rows().numel()
                                            for r, _ in by_req.values())

                # -- checkpoint policy --
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
                    scored_batch = len(rows)
                else:
                    keep_rows = torch.ones(h.shape[0], dtype=torch.bool)
                    eh = None
                    scored_batch = 0

                # -- scatter (per request, timed) --
                t_s0 = time.perf_counter()
                off = 0
                for r, row_list, g_ms in meta:
                    t_i0 = time.perf_counter()
                    t_idx = torch.tensor(row_list, dtype=torch.long)
                    seg = h[off:off + len(row_list)]
                    seg_eh = eh[off:off + len(row_list)] if eh is not None else None
                    kl = keep_rows[off:off + len(row_list)]
                    off += len(row_list)
                    t_keep = t_idx[kl]
                    t_exit = t_idx[~kl]
                    # write the forward output back into the request's hidden
                    # state (regression guard: without this every layer would
                    # re-execute on the raw embeddings)
                    r.hidden = r.hidden.index_copy(0, t_idx, seg)
                    if r.first_exec_time is None:
                        r.first_exec_time = event_start_clock
                    if seg_eh is not None and t_exit.numel():
                        r.exit_layer[t_exit] = layer + 1
                        r.scores[t_exit] = seg_eh[~kl]
                        r.done[t_exit] = True
                    r.next_layer[t_keep] = layer + 1
                    if not self.checkpoints or layer >= self.checkpoints[-1]:
                        r.exit_layer[t_keep] = L
                    s_ms = (time.perf_counter() - t_i0) * 1000.0
                    # P1-4: structured per-request entry; scored_rows is the
                    # request's own share (its rows through score_hidden),
                    # NOT the global batch count
                    r.layer_log.append({
                        "event_id": event_id,
                        "layer": layer,
                        "request_rows": len(row_list),
                        "batch_rows": len(rows),
                        "scored_rows": len(row_list) if scored_batch else 0,
                        "compacted": bool(len(row_list) < r.n),
                        "gather_ms": round(g_ms, 3),
                        "forward_ms": round(forward_ms, 3),
                        "scatter_ms": round(s_ms, 3),
                    })
                scatter_ms = (time.perf_counter() - t_s0) * 1000.0

                self.layer_events.append({
                    "event_id": event_id,
                    "layer": layer,
                    "request_ids": list(by_req.keys()),
                    "request_rows": {qid: len(rows_q)
                                     for qid, (_, rows_q) in by_req.items()},
                    "batch_rows": len(rows),
                    "scored_rows": scored_batch,
                    "compacted": bool(compacted),
                    "gather_ms": round(gather_ms, 3),
                    "forward_ms": round(forward_ms, 3),
                    "scatter_ms": round(scatter_ms, 3),
                })

                if layer == L - 1:
                    off = 0
                    for r, row_list, _ in meta:
                        t_idx = torch.tensor(row_list, dtype=torch.long)
                        kl = keep_rows[off:off + len(row_list)]
                        off += len(row_list)
                        idx = t_idx[kl]
                        if idx.numel():
                            r.scores[idx] = self.ranker.score_hidden(r.hidden[idx])
                            r.done[idx] = True

                now = time.perf_counter()
                for r, _, _ in meta:
                    if r.finish_time is None and bool(r.done.all()):
                        r.finish_time = now
        self._final_score_undone(requests)
        return {r.qid: r.scores.clone() for r in requests}


class SerialScheduler:
    """Real one-worker FIFO online scheduler (review P0-2).

    Requests are processed to completion as they arrive: the run clock waits
    for each arrival, then the request executes through the same exit policy
    as the serial reference path (per-layer compaction, fixed thresholds).
    Every request records its actual arrival, submission, first-execution and
    completion times plus a per-layer gather/forward/scatter timing split.
    """

    def __init__(self, ranker, thresholds=None, keep_fn=None,
                 checkpoints=(1, 3), compact_threshold=0.75, compact_interval=1):
        self.ranker = ranker
        self.thresholds = thresholds
        self.keep_fn = keep_fn
        self.checkpoints = sorted(checkpoints)
        self.compact_threshold = compact_threshold
        self.compact_interval = compact_interval
        self.layer_events = []

    def run(self, requests, online=True):
        t0 = time.perf_counter()
        L = self.ranker.num_layers
        _check_lengths(requests)
        for r in requests:
            r.submit_time = None
            r.first_exec_time = None
            r.finish_time = None
            r.layer_log = []
        pending = sorted(requests, key=lambda r: r.arrival)
        results = {}
        with torch.inference_mode():
            for r in pending:
                now_rel = time.perf_counter() - t0
                if online and r.arrival > now_rel:
                    time.sleep(r.arrival - now_rel)
                r.submit_time = time.perf_counter()
                r.first_exec_time = r.submit_time
                t_e0 = time.perf_counter()
                r.hidden, r.ext_mask = self.ranker.embed(
                    r.input_ids, r.attention_mask, r.token_type_ids)
                r.embed_ms = (time.perf_counter() - t_e0) * 1000.0
                r.done = torch.zeros(r.n, dtype=torch.bool)
                r.next_layer = torch.zeros(r.n, dtype=torch.long)
                r.exit_layer = torch.full((r.n,), L, dtype=torch.long)
                r.scores = torch.zeros(r.n)
                for layer in range(L):
                    idx = torch.nonzero(~r.done, as_tuple=True)[0]
                    if idx.numel() == 0:
                        break
                    compacted = (idx.numel() < r.n
                                 and idx.numel() / r.n < self.compact_threshold
                                 and layer % self.compact_interval == 0)
                    # -- gather --
                    t_g0 = time.perf_counter()
                    if compacted:
                        h = r.hidden.index_select(0, idx)
                        m = r.ext_mask.index_select(0, idx)
                    else:
                        h = r.hidden
                        m = r.ext_mask
                    gather_ms = (time.perf_counter() - t_g0) * 1000.0
                    # -- forward --
                    t_f0 = time.perf_counter()
                    h = self.ranker.forward_layer(h, m, layer)
                    forward_ms = (time.perf_counter() - t_f0) * 1000.0
                    # write the forward output back into the request's hidden
                    # state (regression guard: without this every layer would
                    # re-execute on the raw embeddings)
                    if compacted:
                        r.hidden = r.hidden.index_copy(0, idx, h)
                    else:
                        r.hidden = h
                    # -- checkpoint scoring (= scatter) --
                    scored_rows = 0
                    t_s0 = time.perf_counter()
                    if layer in self.checkpoints and self.thresholds is not None:
                        cur = self.ranker.score_hidden(h)
                        # honest ledger: rows ACTUALLY through score_hidden
                        scored_rows = int(h.shape[0])
                        # when not compacted, h covers all n rows but only the
                        # active positions are decided — select them explicitly
                        # and keep the finished mask in the SAME index space as
                        # the scores it is applied to (active-local, not full-n:
                        # cur[finished] would misindex on the non-compacted path)
                        cur_active = cur if compacted else cur[idx]
                        keep_active = cur_active >= self.thresholds[layer]
                        keep = torch.zeros(r.n, dtype=torch.bool)
                        keep[idx] = keep_active
                        finished = ~keep_active
                        r.exit_layer[idx[finished]] = layer + 1
                        r.scores[idx[finished]] = cur_active[finished]
                        r.done[idx[finished]] = True
                    scatter_ms = (time.perf_counter() - t_s0) * 1000.0
                    event_id = len(self.layer_events)
                    r.layer_log.append({
                        "event_id": event_id,
                        "layer": layer,
                        "request_rows": int(idx.numel()),
                        "batch_rows": int(idx.numel()),
                        "scored_rows": scored_rows,
                        "compacted": bool(compacted),
                        "gather_ms": round(gather_ms, 3),
                        "forward_ms": round(forward_ms, 3),
                        "scatter_ms": round(scatter_ms, 3),
                    })
                    self.layer_events.append({
                        "event_id": event_id,
                        "layer": layer,
                        "request_ids": [r.qid],
                        "request_rows": {r.qid: int(idx.numel())},
                        "batch_rows": int(idx.numel()),
                        "scored_rows": scored_rows,
                        "compacted": bool(compacted),
                        "gather_ms": round(gather_ms, 3),
                        "forward_ms": round(forward_ms, 3),
                        "scatter_ms": round(scatter_ms, 3),
                    })
                # final scoring of survivors (mirrors the serial reference)
                idx = torch.nonzero(~r.done, as_tuple=True)[0]
                if idx.numel():
                    t_s0 = time.perf_counter()
                    r.scores[idx] = self.ranker.score_hidden(r.hidden[idx])
                    r.done[idx] = True
                    s_ms = round((time.perf_counter() - t_s0) * 1000.0, 3)
                    event_id = len(self.layer_events)
                    entry = {
                        "event_id": event_id,
                        "layer": L - 1,
                        "request_rows": int(idx.numel()),
                        "batch_rows": int(idx.numel()),
                        "scored_rows": int(idx.numel()),
                        "compacted": False,
                        "gather_ms": 0.0,
                        "forward_ms": 0.0,
                        "scatter_ms": s_ms,
                    }
                    r.layer_log.append(entry)
                    # the event trace must contain every id a layer_log cites
                    self.layer_events.append({**entry,
                                              "request_ids": [r.qid],
                                              "request_rows": {r.qid: int(idx.numel())}})
                r.finish_time = time.perf_counter()
                results[r.qid] = r.scores.clone()
        return results
