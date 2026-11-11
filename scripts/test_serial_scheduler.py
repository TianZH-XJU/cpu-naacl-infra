"""Correctness checks for the SerialScheduler (P0-1/P0-2 parity suite).

Invariants (discrepancy log D001/D002):
- Test A: full depth (thresholds=None) == per-query forward_full (exact).
- Test B: fixed-threshold path with natural compaction == per-query
          active-set reference; scores AND exit_layer must match, and the
          active set must become non-contiguous (the misindexing trigger).
- Test C: forced non-compacted path (compact_threshold > 1) == the same
          reference — this is the exact index space that D002 corrupted.
- Test D: edges — no survivor (all exit at checkpoint 1), all survivor,
          single candidate.
- Test E: Serial and Wavefront agree per candidate under the same policy.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.scheduler import Request, SerialScheduler, WavefrontScheduler  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

path = "models/ms-marco-MiniLM-L-6-v2"
tok = AutoTokenizer.from_pretrained(path)

QUERIES = [
    ("How does photosynthesis work?", [
        "Photosynthesis converts light into chemical energy in plants.",
        "Chlorophyll absorbs light in the blue and red spectrum.",
        "Mitochondria generate ATP through cellular respiration.",
        "The Calvin cycle fixes carbon dioxide into sugars.",
        "Light reactions occur in the thylakoid membranes.",
        "Plants release oxygen as a byproduct of photosynthesis.",
        "Sunlight provides the energy that drives photosynthesis.",
        "Glucose is synthesized from carbon dioxide and water.",
    ]),
    ("What is quantum entanglement?", [
        "Entanglement links quantum states across distance.",
        "Bell inequalities constrain local hidden variable theories.",
        "Quantum teleportation transfers a state using entanglement.",
        "Superposition allows a qubit to hold 0 and 1 at once.",
        "Decoherence destroys entanglement through environment coupling.",
        "EPR paradox questioned completeness of quantum mechanics.",
        "Entangled photons violate Bell inequalities experimentally.",
        "Quantum key distribution uses entanglement for security.",
        "Measurement of one qubit collapses its entangled partner.",
        "No-communication theorem forbids faster-than-light signaling.",
        "Schrödinger's cat illustrates superposition absurdity.",
    ]),
    ("Define machine learning overfitting.", [
        "Overfitting means a model memorizes training noise.",
        "Regularization penalizes large weights to reduce overfitting.",
        "Cross-validation estimates generalization error.",
        "Dropout randomly disables units during training.",
        "Early stopping halts training when validation loss rises.",
        "Bias-variance tradeoff explains underfitting and overfitting.",
    ]),
]

MAX_LEN = 64
checkpoints = (1, 3)


def build_requests(max_len=MAX_LEN, arrival=0.0):
    reqs = []
    for i, (q, docs) in enumerate(QUERIES):
        enc = tok([q] * len(docs), docs, padding="max_length",
                  truncation=True, max_length=max_len, return_tensors="pt")
        reqs.append(Request(f"q{i}", enc["input_ids"], enc["attention_mask"],
                            enc["token_type_ids"], arrival=arrival))
    return reqs


def per_query_threshold_reference(ranker, r, thresholds, checkpoints):
    """Sequential per-query execution: always compact, same exit policy."""
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


def calibrate_threshold(ranker, layer, q=0.35):
    with torch.inference_mode():
        probe = build_requests()
        h0, m0 = ranker.embed(probe[0].input_ids, probe[0].attention_mask,
                              probe[0].token_type_ids)
        for l in range(layer + 1):
            h0 = ranker.forward_layer(h0, m0, l)
        s0 = ranker.score_hidden(h0).reshape(-1)
    return float(torch.quantile(s0, q))


ranker = ActiveSetRanker(path)

# ---- Test A: full depth, no exits -------------------------------------------
reqs = build_requests()
sched = SerialScheduler(ranker, thresholds=None, checkpoints=())
res = sched.run(reqs, online=False)
max_diff_a = 0.0
with torch.inference_mode():
    for r in reqs:
        ref = ranker.forward_full(r.input_ids, r.attention_mask,
                                  r.token_type_ids)
        max_diff_a = max(max_diff_a, (ref - res[r.qid]).abs().max().item())
print(f"Test A (full depth): max_abs_diff = {max_diff_a:.2e}")
assert max_diff_a < 1e-4, "serial full-depth mismatch vs forward_full"
print("PASS: serial full-depth == per-query forward_full")

# ---- Test B: fixed thresholds, natural compaction ----------------------------
thr = {layer: calibrate_threshold(ranker, layer) for layer in checkpoints}
reqs = build_requests()
sched = SerialScheduler(ranker, thresholds=thr, checkpoints=checkpoints,
                        compact_threshold=0.75)
res = sched.run(reqs, online=False)
max_diff_b, exit_mismatch, total_exited, saw_noncontiguous = 0.0, 0, 0, False
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_threshold_reference(
            ranker, r, thr, checkpoints)
        max_diff_b = max(max_diff_b,
                         (ref_scores - res[r.qid]).abs().max().item())
        exit_mismatch += int((ref_exit != r.exit_layer).sum())
        total_exited += int((r.exit_layer < ranker.num_layers).sum())
        alive_after_cp1 = torch.nonzero(
            r.exit_layer != checkpoints[0] + 1, as_tuple=True)[0]
        if 0 < alive_after_cp1.numel() < r.n:
            gaps = bool((alive_after_cp1[1:] - alive_after_cp1[:-1] > 1).any())
            edges = (alive_after_cp1[0].item() > 0
                     or alive_after_cp1[-1].item() < r.n - 1)
            saw_noncontiguous = saw_noncontiguous or gaps or edges
print(f"Test B (threshold path): max_abs_diff = {max_diff_b:.2e}, "
      f"exit mismatches = {exit_mismatch}, exited rows = {total_exited}, "
      f"non-contiguous active rows = {saw_noncontiguous}")
assert exit_mismatch == 0 and max_diff_b < 1e-4 and total_exited > 0
assert saw_noncontiguous, "expected non-contiguous active rows (D002 trigger)"
print("PASS: serial threshold path == per-query reference (compacted)")

# ---- Test C: forced non-compacted path (the D002 index space) ----------------
reqs = build_requests()
sched = SerialScheduler(ranker, thresholds=thr, checkpoints=checkpoints,
                        compact_threshold=2.0)
res = sched.run(reqs, online=False)
max_diff_c, exit_mismatch_c = 0.0, 0
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_threshold_reference(
            ranker, r, thr, checkpoints)
        max_diff_c = max(max_diff_c,
                         (ref_scores - res[r.qid]).abs().max().item())
        exit_mismatch_c += int((ref_exit != r.exit_layer).sum())
print(f"Test C (non-compacted): max_abs_diff = {max_diff_c:.2e}, "
      f"exit mismatches = {exit_mismatch_c}")
assert exit_mismatch_c == 0 and max_diff_c < 1e-4
print("PASS: serial non-compacted path == per-query reference")

# ---- Test D: edges ------------------------------------------------------------
# D1: no survivor — everything exits at checkpoint 1
low_thr = {layer: calibrate_threshold(ranker, layer, q=1.0) + 1e3
           for layer in checkpoints}
reqs = build_requests()
sched = SerialScheduler(ranker, thresholds=low_thr, checkpoints=checkpoints)
res = sched.run(reqs, online=False)
all_exited = all(bool((r.exit_layer == checkpoints[0] + 1).all())
                 for r in reqs)
max_diff_d1 = max((res[r.qid] - per_query_threshold_reference(
    ranker, r, low_thr, checkpoints)[0]).abs().max().item() for r in reqs)
print(f"Test D1 (no survivor): all exited at L{checkpoints[0] + 1} = "
      f"{all_exited}, max_abs_diff = {max_diff_d1:.2e}")
assert all_exited and max_diff_d1 < 1e-4

# D2: all survivor — nothing exits
high_thr = {layer: calibrate_threshold(ranker, layer, q=0.0) - 1e3
            for layer in checkpoints}
reqs = build_requests()
sched = SerialScheduler(ranker, thresholds=high_thr, checkpoints=checkpoints)
res = sched.run(reqs, online=False)
none_exited = all(bool((r.exit_layer == ranker.num_layers).all())
                  for r in reqs)
max_diff_d2 = 0.0
with torch.inference_mode():
    for r in reqs:
        ref = ranker.forward_full(r.input_ids, r.attention_mask,
                                  r.token_type_ids)
        max_diff_d2 = max(max_diff_d2, (ref - res[r.qid]).abs().max().item())
print(f"Test D2 (all survivor): none exited = {none_exited}, "
      f"max_abs_diff = {max_diff_d2:.2e}")
assert none_exited and max_diff_d2 < 1e-4

# D3: single candidate
enc = tok(["What causes Alzheimer's disease?"],
          ["Amyloid plaques accumulate between neurons."],
          padding="max_length", truncation=True, max_length=MAX_LEN,
          return_tensors="pt")
single = Request("q_single", enc["input_ids"], enc["attention_mask"],
                 enc["token_type_ids"])
sched = SerialScheduler(ranker, thresholds=thr, checkpoints=checkpoints)
res = sched.run([single], online=False)
ref_scores, ref_exit = per_query_threshold_reference(ranker, single, thr,
                                                     checkpoints)
max_diff_d3 = (ref_scores - res["q_single"]).abs().max().item()
exit_mismatch_d3 = int((ref_exit != single.exit_layer).sum())
print(f"Test D3 (single candidate): max_abs_diff = {max_diff_d3:.2e}, "
      f"exit mismatches = {exit_mismatch_d3}")
assert exit_mismatch_d3 == 0 and max_diff_d3 < 1e-4
print("PASS: edge cases (no survivor / all survivor / single candidate)")

# ---- Test E: Serial vs Wavefront under the same policy ------------------------
reqs_s = build_requests()
reqs_w = build_requests()
serial = SerialScheduler(ranker, thresholds=thr, checkpoints=checkpoints)
wave = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                          keep_fn=lambda s, layer: s >= thr[layer])
res_s = serial.run(reqs_s, online=False)
res_w = wave.run(reqs_w)
max_diff_e = max((res_s[r.qid] - res_w[r.qid]).abs().max().item()
                 for r in reqs_s)
exit_mismatch_e = sum(int((rs.exit_layer != rw.exit_layer).sum())
                      for rs, rw in zip(reqs_s, reqs_w))
print(f"Test E (serial vs wavefront): max_abs_diff = {max_diff_e:.2e}, "
      f"exit mismatches = {exit_mismatch_e}")
assert exit_mismatch_e == 0 and max_diff_e < 1e-4
print("PASS: serial == wavefront per candidate (same policy)")

print("ALL SERIAL SCHEDULER PARITY TESTS PASSED")
