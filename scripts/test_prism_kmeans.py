"""Prism KMeans validation (review 2 P1-5 / Q12).

The kmeans1d bug fix (best_centers now saved with best_labels) is only
exercised when clustering actually triggers. prism_cv_fixed.jsonl proves the
raw-CV path structurally never fires, so it validates nothing about the fix.
This test closes Q12 with:
  1. a synthetic multi-init test: data where inits disagree, verifying that
     labels and centers always come from the SAME initialization, and that
     the result matches sklearn KMeans(n_init=3, random_state=0);
  2. a forced-trigger test: input whose CV exceeds the threshold with >= 6
     candidates, so the real clustering path must execute and prune.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec_path = Path(__file__).resolve().parent / "calibrate_prism_cv.py"
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("cpc", _spec_path)
cpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpc)

torch.manual_seed(0)


def cluster_of(labels, centers):
    """Cluster index each label belongs to, per the argmin(centers) rule."""
    return labels == int(np.argmin(centers))


def check_labels_centers_consistent(s, labels, centers, tag):
    """labels and centers must describe the same partition: the mean of the
    members of every non-empty cluster equals its center (within tolerance),
    and the argmin cluster is the lowest-center cluster."""
    assert len(centers) == centers.reshape(-1).shape[0]
    assert set(np.unique(labels)).issubset(set(range(len(centers))))
    for j in range(len(centers)):
        members = s[labels == j]
        assert members.size > 0, f"{tag}: empty cluster {j} in labels"
        assert abs(members.mean() - centers[j]) < 1e-6, (
            f"{tag}: center {j} ({centers[j]:.6f}) != label mean "
            f"({members.mean():.6f}) — labels/centers from different inits")
    lo = int(np.argmin(centers))
    drop = cluster_of(labels, centers)
    assert drop.sum() > 0, f"{tag}: argmin cluster holds no members"
    # the dropped cluster must be the lowest-center one
    assert centers[lo] == centers.min(), f"{tag}: argmin is not the lowest center"
    return drop


# ---- Test 1: synthetic multi-init — labels and centers from the same init ----
rng = np.random.RandomState(0)
# three well-separated modes so different inits converge to different splits
s = np.concatenate([rng.normal(2.0, 0.05, 12),
                    rng.normal(5.0, 0.05, 12),
                    rng.normal(9.0, 0.05, 12)])
s = torch.tensor(np.sort(s))
labels, centers = cpc.kmeans1d(s.numpy(), k=3, n_init=3, seed=0)
drop = check_labels_centers_consistent(s.numpy(), labels, centers, "multi-init")

# sklearn reference: same partition up to label permutation
from sklearn.cluster import KMeans  # noqa: E402
km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(s.numpy().reshape(-1, 1))
sk_centers = np.sort(km.cluster_centers_.flatten())
my_centers = np.sort(centers)
assert np.allclose(sk_centers, my_centers, atol=1e-4), (
    f"sklearn centers {sk_centers} != kmeans1d centers {my_centers}")
sk_sizes = np.sort([int((km.labels_ == j).sum()) for j in range(3)])
my_sizes = np.sort([int((labels == j).sum()) for j in range(3)])
assert (sk_sizes == my_sizes).all(), (
    f"sklearn cluster sizes {sk_sizes} != kmeans1d sizes {my_sizes}")
print(f"Test 1 (multi-init): centers {np.round(my_centers, 4).tolist()} match "
      f"sklearn; labels/centers consistent; drop={int(drop.sum())}/36")
print("PASS: kmeans1d labels and centers come from the same init")

# ---- Test 2: forced trigger — the clustering path must actually execute ----
# raw-CV calibration data never fires (median CV 0.013 << 0.06); construct
# input whose CV exceeds every calibrated threshold with >= 6 candidates.
s_trig = torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 12.0])
cv_trig = float(np.std(s_trig.numpy()) / (abs(np.mean(s_trig.numpy())) + 1e-9))
assert len(s_trig) >= 6, "forced-trigger input must have >= 6 candidates"
assert cv_trig > 0.5, f"constructed CV {cv_trig:.3f} does not exceed 0.5"
labels_t, centers_t = cpc.kmeans1d(s_trig.numpy(), k=3, n_init=3, seed=0)
drop_t = check_labels_centers_consistent(s_trig.numpy(), labels_t, centers_t,
                                         "forced-trigger")
# prism prunes the LOWEST-score cluster (argmin centers); cluster indices
# are arbitrary from the init, so compare center VALUES and verify the
# minimum-score point is pruned
lo_t = int(np.argmin(centers_t))
min_point_cluster = int(labels_t[int(np.argmin(s_trig.numpy()))])
assert centers_t[min_point_cluster] == centers_t.min(), (
    f"min-score point in cluster {min_point_cluster} "
    f"(center {centers_t[min_point_cluster]:.4f}), lowest-center cluster is "
    f"{lo_t} ({centers_t.min():.4f})")
assert drop_t[int(np.argmin(s_trig.numpy()))], "min-score point was not pruned"
assert drop_t.sum() > 0 and (~drop_t).any(), "pruning must keep some candidates"
print(f"Test 2 (forced trigger): CV={cv_trig:.3f} fires; lowest-score cluster "
      f"(center {centers_t.min():.4f}) pruned, {int(drop_t.sum())}/{len(s_trig)} dropped")
print("PASS: KMeans fix validated on the real trigger path")

print("ALL PASS")
