"""Deterministic toy-ranker parity tests for both schedulers (CI-safe).

Uses a tiny randomly-initialized BERT (2 layers, hidden 32, seq len 16) so no
model download is needed; the schedulers only touch the ranker interface
(embed / forward_layer / score_hidden), so parity results transfer to the
real model. Covers the D001/D002 regression classes:

- Serial full depth == forward_full
- Serial fixed-threshold path == per-query reference, compacted AND
  non-compacted, with non-contiguous active rows
- Serial edges: no survivor, all survivor, single candidate
- Serial == Wavefront per candidate under the same policy
- Wavefront full depth == forward_full, max_batch overflow
"""
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast.scheduler import Request, SerialScheduler, WavefrontScheduler  # noqa: E402

torch.manual_seed(0)


class ToyRanker:
    """Same interface as ActiveSetRanker, tiny random BERT inside."""

    def __init__(self, num_layers=2, hidden=32, seq_len=16, vocab=100):
        cfg = BertConfig(vocab_size=vocab, hidden_size=hidden,
                         num_hidden_layers=num_layers,
                         num_attention_heads=4, intermediate_size=64,
                         max_position_embeddings=64)
        self.bert = BertModel(cfg).eval()
        self.classifier = torch.nn.Linear(hidden, 1)
        # widen the score spread so threshold decisions sit far from the
        # ~1e-6 float noise introduced by different batch compositions
        # (otherwise rows near the threshold flip between schedulers)
        with torch.no_grad():
            self.classifier.weight.mul_(20.0)
        self.num_layers = num_layers

    def embed(self, input_ids, attention_mask, token_type_ids):
        ext_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) \
            * torch.finfo(torch.float32).min
        hidden = self.bert.embeddings(input_ids=input_ids,
                                      token_type_ids=token_type_ids)
        return hidden, ext_mask

    def forward_layer(self, hidden, ext_mask, layer_idx):
        out = self.bert.encoder.layer[layer_idx](hidden, attention_mask=ext_mask)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def score_hidden(self, hidden):
        pooled = self.bert.pooler(hidden)
        return self.classifier(pooled).squeeze(-1)

    def forward_full(self, input_ids, attention_mask, token_type_ids):
        hidden, ext_mask = self.embed(input_ids, attention_mask, token_type_ids)
        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, ext_mask, i)
        return self.score_hidden(hidden)


def build_requests(n_docs=(8, 11, 6), seq_len=16, vocab=100, arrival=0.0):
    reqs = []
    for i, n in enumerate(n_docs):
        reqs.append(Request(
            f"q{i}",
            torch.randint(1, vocab, (n, seq_len)),
            torch.ones(n, seq_len, dtype=torch.long),
            torch.zeros(n, seq_len, dtype=torch.long),
            arrival=arrival))
    return reqs


def per_query_reference(ranker, r, thresholds, checkpoints):
    L = ranker.num_layers
    n = r.n
    with torch.inference_mode():
        hidden, ext_mask = ranker.embed(r.input_ids, r.attention_mask,
                                        r.token_type_ids)
        done = torch.zeros(n, dtype=torch.bool)
        exit_layer = torch.full((n,), L, dtype=torch.long)
        scores = torch.zeros(n)
        for layer in range(L):
            idx = torch.nonzero(~done, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            h = hidden.index_select(0, idx)
            m = ext_mask.index_select(0, idx)
            h = ranker.forward_layer(h, m, layer)
            hidden = hidden.index_copy(0, idx, h)
            if layer in checkpoints and thresholds is not None:
                cur = ranker.score_hidden(h).reshape(-1)
                keep = cur >= thresholds[layer]
                x_idx = idx[~keep]
                exit_layer[x_idx] = layer + 1
                scores[x_idx] = cur[~keep]
                done[x_idx] = True
        idx = torch.nonzero(~done, as_tuple=True)[0]
        if idx.numel():
            scores[idx] = ranker.score_hidden(hidden[idx])
    return scores, exit_layer


ranker = ToyRanker()
L = ranker.num_layers
checkpoints = tuple(c for c in (0, 1) if c < L)
TOL = 1e-5


def calibrate_threshold(ranker, layer):
    """Place the threshold inside the widest gap of the pooled score
    distribution. Transformer layer outputs are batch-composition sensitive
    at the ~1e-2 level (GEMM blocking), so a threshold next to a score would
    flip that row's decision between batch shapes; a wide-gap midpoint makes
    every decision robust to that noise."""
    vals = []
    with torch.inference_mode():
        for r in build_requests():
            h, m = ranker.embed(r.input_ids, r.attention_mask,
                                r.token_type_ids)
            for l in range(layer + 1):
                h = ranker.forward_layer(h, m, l)
            vals.extend(ranker.score_hidden(h).reshape(-1).tolist())
    s = sorted(vals)
    lo, hi = max(1, len(s) // 10), min(len(s) - 1, (9 * len(s)) // 10)
    gap, pos = max((s[i + 1] - s[i], i) for i in range(lo, hi))
    assert gap > 0.1, f"no robust threshold gap at layer {layer}: {gap:.3f}"
    return (s[pos] + s[pos + 1]) / 2.0


thr = {c: calibrate_threshold(ranker, c) for c in checkpoints}

# ---- Test A: serial full depth -----------------------------------------------
reqs = build_requests()
res = SerialScheduler(ranker, thresholds=None, checkpoints=()).run(reqs,
                                                                   online=False)
max_diff = max((ranker.forward_full(r.input_ids, r.attention_mask,
                                    r.token_type_ids) - res[r.qid])
               .abs().max().item() for r in reqs)
print(f"Test A (serial full depth): max_abs_diff = {max_diff:.2e}")
assert max_diff < TOL

# ---- Test B: serial threshold path, compacted + non-compacted -----------------
for compact_threshold in (0.75, 2.0):
    reqs = build_requests()
    res = SerialScheduler(ranker, thresholds=thr, checkpoints=checkpoints,
                          compact_threshold=compact_threshold).run(reqs,
                                                                   online=False)
    diff, exit_mm, exited, noncontig = 0.0, 0, 0, False
    for r in reqs:
        ref_s, ref_e = per_query_reference(ranker, r, thr, checkpoints)
        diff = max(diff, (ref_s - res[r.qid]).abs().max().item())
        exit_mm += int((ref_e != r.exit_layer).sum())
        exited += int((r.exit_layer < L).sum())
        alive = torch.nonzero(r.exit_layer != checkpoints[0] + 1,
                              as_tuple=True)[0]
        if 0 < alive.numel() < r.n:
            noncontig = noncontig or bool(
                (alive[1:] - alive[:-1] > 1).any()) or \
                alive[0].item() > 0 or alive[-1].item() < r.n - 1
    print(f"Test B (compact={compact_threshold}): max_abs_diff = {diff:.2e}, "
          f"exit mismatches = {exit_mm}, exited = {exited}, "
          f"non-contiguous = {noncontig}")
    assert diff < TOL and exit_mm == 0 and exited > 0
    assert noncontig, "expected non-contiguous active rows (D002 trigger)"
print("PASS: serial threshold path == per-query reference (both paths)")

# ---- Test C: serial edges ------------------------------------------------------
low_thr = {c: 1e3 for c in checkpoints}
reqs = build_requests()
res = SerialScheduler(ranker, thresholds=low_thr,
                      checkpoints=checkpoints).run(reqs, online=False)
assert all(bool((r.exit_layer == checkpoints[0] + 1).all()) for r in reqs)
high_thr = {c: -1e3 for c in checkpoints}
reqs = build_requests()
res = SerialScheduler(ranker, thresholds=high_thr,
                      checkpoints=checkpoints).run(reqs, online=False)
assert all(bool((r.exit_layer == L).all()) for r in reqs)
for r in reqs:
    ref = ranker.forward_full(r.input_ids, r.attention_mask, r.token_type_ids)
    assert (ref - res[r.qid]).abs().max().item() < TOL
single = Request("q1", torch.randint(1, 100, (1, 16)),
                 torch.ones(1, 16, dtype=torch.long),
                 torch.zeros(1, 16, dtype=torch.long))
res = SerialScheduler(ranker, thresholds=thr,
                      checkpoints=checkpoints).run([single], online=False)
ref_s, ref_e = per_query_reference(ranker, single, thr, checkpoints)
assert (ref_s - res["q1"]).abs().max().item() < TOL
assert int((ref_e != single.exit_layer).sum()) == 0
print("PASS: serial edges (no survivor / all survivor / single candidate)")

# ---- Test D: serial == wavefront per candidate ---------------------------------
base = build_requests()
reqs_s = [Request(r.qid, r.input_ids.clone(), r.attention_mask.clone(),
                  r.token_type_ids.clone()) for r in base]
reqs_w = [Request(r.qid, r.input_ids.clone(), r.attention_mask.clone(),
                  r.token_type_ids.clone()) for r in base]
res_s = SerialScheduler(ranker, thresholds=thr,
                        checkpoints=checkpoints).run(reqs_s, online=False)
res_w = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                           keep_fn=lambda s, layer: s >= thr[layer]).run(reqs_w)
diff = max((res_s[r.qid] - res_w[r.qid]).abs().max().item()
           for r in reqs_s)
exit_mm = sum(int((rs.exit_layer != rw.exit_layer).sum())
              for rs, rw in zip(reqs_s, reqs_w))
print(f"Test D (serial vs wavefront): max_abs_diff = {diff:.2e}, "
      f"exit mismatches = {exit_mm}")
assert diff < TOL and exit_mm == 0

# ---- Test E: wavefront full depth + max_batch overflow -------------------------
reqs = build_requests()
res = WavefrontScheduler(ranker, checkpoints=()).run(reqs)
max_diff = max((ranker.forward_full(r.input_ids, r.attention_mask,
                                    r.token_type_ids) - res[r.qid])
               .abs().max().item() for r in reqs)
assert max_diff < TOL
reqs = build_requests()
res = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                         keep_fn=lambda s, layer: s >= thr[layer],
                         max_batch=4).run(reqs)
diff = max((per_query_reference(ranker, r, thr, checkpoints)[0] - res[r.qid])
           .abs().max().item() for r in reqs)
assert diff < TOL and all(bool(r.done.all()) for r in reqs)
print("PASS: wavefront full depth + max_batch overflow")

print("ALL TOY SCHEDULER PARITY TESTS PASSED")
