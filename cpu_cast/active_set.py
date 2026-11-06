"""Layer-wise active-set execution for cross-encoder rerankers.

Implements the CPU-CAST active-set runtime core (plan section 10):

- per-candidate state: hidden states, active flag, exit layer, current score
- layer-by-layer forward with optional compaction of the active set
  (Gate A: compact when the active fraction drops below a threshold)
- checkpoint scoring via the model's own pooler + classification head
- correctness: full-depth layer-wise execution must match the standard
  HuggingFace forward pass (verified in scripts/test_active_set.py).
"""
import torch
from transformers import AutoModelForSequenceClassification


class ActiveSetRanker:
    def __init__(self, model_path, device="cpu"):
        full = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        self.bert = full.bert
        self.dropout = full.dropout
        self.classifier = full.classifier
        self.num_layers = len(self.bert.encoder.layer)
        self.device = device

    def embed(self, input_ids, attention_mask, token_type_ids):
        ext_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
        hidden = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        return hidden, ext_mask

    def forward_layer(self, hidden, ext_mask, layer_idx):
        out = self.bert.encoder.layer[layer_idx](hidden, attention_mask=ext_mask)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def score_hidden(self, hidden):
        pooled = self.bert.pooler(hidden)
        return self.classifier(self.dropout(pooled)).squeeze(-1)

    def forward_full(self, input_ids, attention_mask, token_type_ids):
        hidden, ext_mask = self.embed(input_ids, attention_mask, token_type_ids)
        for i in range(self.num_layers):
            hidden = self.forward_layer(hidden, ext_mask, i)
        return self.score_hidden(hidden)

    def run_active_set(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        checkpoints=None,
        keep_fn=None,
        compact_threshold=0.75,
        compact_interval=1,
    ):
        """Run the encoder layer-by-layer with per-candidate early exit.

        checkpoints: layer indices (0-based; checkpoint fires AFTER that layer)
        keep_fn(scores, layer_idx): bool tensor of candidates that continue
        compact_threshold: compact the active set when active/total < this
        compact_interval: attempt compaction only every k layers (1 = every
            layer); between compactions the full stale batch is executed
        Returns (scores [n], exit_layer [n]); exited candidates keep the score
        from their exit checkpoint, survivors get the full-depth score.
        """
        n = input_ids.shape[0]
        checkpoints = checkpoints or []
        hidden, ext_mask = self.embed(input_ids, attention_mask, token_type_ids)
        active = torch.ones(n, dtype=torch.bool, device=self.device)
        exit_layer = torch.full((n,), self.num_layers, dtype=torch.int64)
        scores = torch.zeros(n)

        for i in range(self.num_layers):
            idx = torch.nonzero(active, as_tuple=True)[0]
            if (
                idx.numel() < n
                and idx.numel() / n < compact_threshold
                and i % compact_interval == 0
            ):
                h = hidden.index_select(0, idx)
                m = ext_mask.index_select(0, idx)
                h = self.forward_layer(h, m, i)
                hidden = hidden.index_copy(0, idx, h)
            else:
                hidden = self.forward_layer(hidden, ext_mask, i)
            if i in checkpoints and keep_fn is not None:
                cur = self.score_hidden(hidden)
                idx_active = torch.nonzero(active, as_tuple=True)[0]
                keep_active = keep_fn(cur[idx_active], i)
                keep = torch.zeros(n, dtype=torch.bool, device=self.device)
                keep[idx_active] = keep_active
                finished = active & ~keep
                exit_layer[finished] = i + 1
                scores[finished] = cur[finished]
                active = active & keep
        scores[active] = self.score_hidden(hidden)[active]
        return scores, exit_layer
