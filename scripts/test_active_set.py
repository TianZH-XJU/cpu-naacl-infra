"""Correctness check: layer-wise full-depth execution must match the standard
HuggingFace forward pass (plan Stage 4 go/no-go requirement, run early)."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)
path = "models/ms-marco-MiniLM-L-6-v2"
tok = AutoTokenizer.from_pretrained(path)
pairs = [
    ("How does photosynthesis work?", "Photosynthesis converts light into chemical energy in plants."),
    ("What is quantum entanglement?", "Entanglement links quantum states across distance."),
    ("Short one.", "Tiny."),
]
enc = tok(pairs, padding=True, truncation=True, max_length=64, return_tensors="pt")

ref = AutoModelForSequenceClassification.from_pretrained(path).eval()
with torch.inference_mode():
    ref_scores = ref(**enc).logits.squeeze(-1)

ranker = ActiveSetRanker(path)
with torch.inference_mode():
    lw_scores, exit_layer = ranker.run_active_set(
        enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
        checkpoints=[], keep_fn=None,
    )

diff = (ref_scores - lw_scores).abs().max().item()
print("reference :", [round(float(x), 5) for x in ref_scores])
print("layerwise :", [round(float(x), 5) for x in lw_scores])
print("exit_layer:", exit_layer.tolist())
print(f"max_abs_diff = {diff:.2e}")
assert diff < 1e-4, "layer-wise execution does not match standard forward"
print("PASS: layer-wise full-depth == standard forward")
