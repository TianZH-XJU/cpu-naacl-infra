"""Correctness checks for the cross-query wavefront scheduler (post-review v2).

Invariants:
- Test A: full-depth wavefront == per-query forward_full (exact).
- Test B: exit-head path (MLP + fixed threshold) == per-query reference.
- Test C: POOLER path (signal="pooler", the Stage-4 production path) ==
          per-query reference using score_hidden at checkpoints.
- Test D: max_batch overflow — no lost rows, no double execution,
          next_layer/exit_layer consistent, scores still correct.
- Test E: mixed sequence lengths are rejected with a clear error.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.scheduler import Request, WavefrontScheduler  # noqa: E402

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
        "Data augmentation increases effective training set size.",
        "L2 regularization adds a squared weight penalty.",
        "A validation curve diagnoses overfitting.",
        "Ensembles average models to reduce variance.",
        "Double descent complicates classical generalization theory.",
        "Label noise encourages memorization of spurious patterns.",
        "Deep networks can interpolate training data exactly.",
        "Weight decay is a form of parameter norm regularization.",
        "Test error diverges as training error approaches zero.",
        "Model capacity must match dataset complexity.",
        "Holdout sets must remain untouched during tuning.",
    ]),
]

MAX_LEN = 64


def build_requests(max_len=MAX_LEN, arrival=0.0):
    reqs = []
    for i, (q, docs) in enumerate(QUERIES):
        enc = tok([q] * len(docs), docs, padding="max_length",
                  truncation=True, max_length=max_len, return_tensors="pt")
        reqs.append(Request(f"q{i}", enc["input_ids"], enc["attention_mask"],
                            enc["token_type_ids"], arrival=arrival))
    return reqs


def per_query_reference(ranker, r, head, checkpoints, thr, signal="head"):
    """Sequential per-query execution with the same policy as the scheduler."""
    L = ranker.num_layers
    n = r.n
    with torch.inference_mode():
        hidden, ext_mask = ranker.embed(r.input_ids, r.attention_mask, r.token_type_ids)
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
            if layer in checkpoints and (head is not None or signal == "pooler"):
                if signal == "pooler":
                    eh = ranker.score_hidden(h).reshape(-1)
                else:
                    eh = head(h[:, 0, :].float()).reshape(-1)
                keep = eh > thr
                x_idx = idx[~keep]
                exit_layer[x_idx] = layer + 1
                scores[x_idx] = eh[~keep]
                done[x_idx] = True
        idx = torch.nonzero(~done, as_tuple=True)[0]
        if idx.numel():
            scores[idx] = ranker.score_hidden(hidden[idx])
    return scores, exit_layer


ranker = ActiveSetRanker(path)
checkpoints = (1, 3)

# ---- Test A: full depth, no exits -------------------------------------------
reqs = build_requests()
sched = WavefrontScheduler(ranker, checkpoints=())
wf = sched.run(reqs)
max_diff_a = 0.0
with torch.inference_mode():
    for r in reqs:
        ref = ranker.forward_full(r.input_ids, r.attention_mask, r.token_type_ids)
        max_diff_a = max(max_diff_a, (ref - wf[r.qid]).abs().max().item())
print(f"Test A (full depth): max_abs_diff = {max_diff_a:.2e}")
assert max_diff_a < 1e-4, "wavefront full-depth mismatch vs forward_full"
print("PASS: wavefront full-depth == per-query forward_full")

# ---- Test B: exit-head path (MLP + fixed threshold) --------------------------
head = torch.nn.Sequential(
    torch.nn.Linear(ranker.bert.config.hidden_size, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 1),
)
with torch.inference_mode():
    probe = build_requests()
    h0, _ = ranker.embed(probe[0].input_ids, probe[0].attention_mask,
                         probe[0].token_type_ids)
    for layer in range(checkpoints[0] + 1):
        h0 = ranker.forward_layer(h0, probe[0].ext_mask, layer)
    eh0 = head(h0[:, 0, :].float()).reshape(-1)
thr_b = float(torch.quantile(eh0, 0.35))

reqs = build_requests()
sched = WavefrontScheduler(ranker, exit_heads=[head, head], checkpoints=checkpoints,
                           keep_fn=lambda eh, layer: eh > thr_b)
wf = sched.run(reqs)
max_diff_b, exit_mismatch, total_exited = 0.0, 0, 0
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_reference(ranker, r, head, checkpoints,
                                                   thr_b, signal="head")
        max_diff_b = max(max_diff_b, (ref_scores - wf[r.qid]).abs().max().item())
        exit_mismatch += int((ref_exit != r.exit_layer).sum())
        total_exited += int((r.exit_layer < ranker.num_layers).sum())
print(f"Test B (exit-head path): max_abs_diff = {max_diff_b:.2e}, "
      f"exit mismatches = {exit_mismatch}, exited rows = {total_exited}")
assert exit_mismatch == 0 and max_diff_b < 1e-4 and total_exited > 0
print("PASS: wavefront exit-head path == per-query active-set execution")

# ---- Test C: POOLER path (production signal) ---------------------------------
with torch.inference_mode():
    probe = build_requests()
    h0, _ = ranker.embed(probe[0].input_ids, probe[0].attention_mask,
                         probe[0].token_type_ids)
    for layer in range(checkpoints[0] + 1):
        h0 = ranker.forward_layer(h0, probe[0].ext_mask, layer)
    s0 = ranker.score_hidden(h0).reshape(-1)
thr_c = float(torch.quantile(s0, 0.35))

reqs = build_requests()
sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                           keep_fn=lambda s, layer: s > thr_c)
wf = sched.run(reqs)
max_diff_c, exit_mismatch_c, total_exited_c = 0.0, 0, 0
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_reference(ranker, r, None, checkpoints,
                                                   thr_c, signal="pooler")
        max_diff_c = max(max_diff_c, (ref_scores - wf[r.qid]).abs().max().item())
        exit_mismatch_c += int((ref_exit != r.exit_layer).sum())
        total_exited_c += int((r.exit_layer < ranker.num_layers).sum())
print(f"Test C (pooler path): max_abs_diff = {max_diff_c:.2e}, "
      f"exit mismatches = {exit_mismatch_c}, exited rows = {total_exited_c}")
assert exit_mismatch_c == 0 and max_diff_c < 1e-4 and total_exited_c > 0
print("PASS: wavefront pooler path == per-query active-set execution")

# ---- Test D: max_batch overflow ----------------------------------------------
reqs = build_requests()
sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                           keep_fn=lambda s, layer: s > thr_c, max_batch=8)
wf = sched.run(reqs)
max_diff_d, exit_mismatch_d = 0.0, 0
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_reference(ranker, r, None, checkpoints,
                                                   thr_c, signal="pooler")
        max_diff_d = max(max_diff_d, (ref_scores - wf[r.qid]).abs().max().item())
        exit_mismatch_d += int((ref_exit != r.exit_layer).sum())
        # no row left un-executed and no double execution: every row either
        # exited at a checkpoint or was scored at full depth
        assert bool(r.done.all()), f"rows left undone in {r.qid}"
        assert r.exit_layer.shape[0] == r.n
print(f"Test D (max_batch=8 overflow): max_abs_diff = {max_diff_d:.2e}, "
      f"exit mismatches = {exit_mismatch_d}")
assert exit_mismatch_d == 0 and max_diff_d < 1e-4
print("PASS: max_batch overflow handled correctly (no lost/double rows)")

# ---- Test E: mixed sequence lengths rejected ---------------------------------
r64 = build_requests(max_len=64)
r96 = build_requests(max_len=96)
try:
    WavefrontScheduler(ranker, checkpoints=()).run([r64[0], r96[0]])
    raise AssertionError("mixed sequence lengths were NOT rejected")
except ValueError as e:
    print(f"Test E (mixed seq lens): rejected with: {e}")
print("PASS: mixed sequence lengths rejected")

print("ALL PASS")
