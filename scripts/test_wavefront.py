"""Correctness check for the cross-query wavefront scheduler (plan Stage 4).

Invariant: wavefront execution across multiple in-flight requests must produce
the same scores and exit layers as sequential per-query active-set execution
with an identical policy, up to float non-determinism from different GEMM
batch compositions.

Test A: full depth, no exits -> must match per-query forward_full exactly.
Test B: checkpoint exit policy (fixed-threshold keep on an exit head) ->
        scores and exit layers must match a per-query reference loop.
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


def build_requests():
    reqs = []
    for i, (q, docs) in enumerate(QUERIES):
        enc = tok([q] * len(docs), docs, padding="max_length",
                  truncation=True, max_length=MAX_LEN, return_tensors="pt")
        reqs.append(Request(f"q{i}", enc["input_ids"], enc["attention_mask"],
                            enc["token_type_ids"]))
    return reqs


def per_query_reference(ranker, r, head, checkpoints, thr):
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
            if layer in checkpoints and head is not None:
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

# ---- Test B: checkpoint exit policy -----------------------------------------
checkpoints = (1, 3)
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
thr = float(torch.quantile(eh0, 0.35))
print(f"exit threshold = {thr:.4f}")

reqs = build_requests()
sched = WavefrontScheduler(ranker, exit_heads=[head, head], checkpoints=checkpoints,
                           keep_fn=lambda eh, layer: eh > thr)
wf = sched.run(reqs)

max_diff_b = 0.0
exit_mismatch = 0
total_exited = 0
with torch.inference_mode():
    for r in reqs:
        ref_scores, ref_exit = per_query_reference(ranker, r, head, checkpoints, thr)
        max_diff_b = max(max_diff_b, (ref_scores - wf[r.qid]).abs().max().item())
        exit_mismatch += int((ref_exit != r.exit_layer).sum())
        total_exited += int((r.exit_layer < ranker.num_layers).sum())
print(f"Test B (exit policy): max_abs_diff = {max_diff_b:.2e}, "
      f"exit_layer mismatches = {exit_mismatch}, exited rows = {total_exited}")
assert exit_mismatch == 0, "exit decisions differ between wavefront and per-query"
assert max_diff_b < 1e-4, "wavefront scores mismatch per-query reference"
assert total_exited > 0, "exit policy never triggered; test is trivial"
print("PASS: wavefront exit policy == per-query active-set execution")

print("\nlayer events (layer, rows, requests):", sched.layer_events)
print("ALL PASS")
