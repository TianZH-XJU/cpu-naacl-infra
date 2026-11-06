"""Effectiveness metrics via pytrec_eval (Recall@100, nDCG@10, MRR@10)."""
import pytrec_eval


def evaluate(qrels, run, measures=("recall_100", "ndcg_cut_10", "recip_rank")):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(measures))
    scores = evaluator.evaluate(run)
    agg = {}
    for measure in measures:
        values = [s[measure] for s in scores.values() if measure in s]
        agg[measure] = sum(values) / len(values) if values else 0.0
    agg["num_queries"] = len(scores)
    return agg


def recall_at_k(qrels, run, k=100):
    hits = 0
    total = 0
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:k]
        retrieved = {d for d, _ in ranked}
        hits += len(relevant & retrieved)
        total += len(relevant)
    return hits / total if total else 0.0
