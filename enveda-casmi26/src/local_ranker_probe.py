"""Evaluate the published V17 ranking features on paired query-ID holdout groups.

This is a local diagnostic: the published group ID has no verified structure mapping.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "outputs/v17_tie_probe_20261105_01/assets/rank_train.npz"
OUTPUT = ROOT / "outputs/local_ranker_probe_20261119_01"
EXPECTED_SHA = "41ccec87fba5f8ab255aaf5a723567f25bd766064265c5cc2b83ba2cf15f57f8"
SEED = 20261119


def score_queries(pred: np.ndarray, y: np.ndarray, mode: np.ndarray, group: np.ndarray):
    keys = np.column_stack((mode, group))
    cuts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, len(y)]
    ranks = np.empty(len(cuts) - 1, dtype=np.float64)
    for i, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        order = np.argsort(-pred[start:end], kind="stable")
        positive_rank = int(np.flatnonzero(y[start:end][order] > 0)[0]) + 1
        ranks[i] = 1 / positive_rank if positive_rank <= 25 else 0
    return keys[cuts[:-1]], ranks


def paired_ci(delta: np.ndarray, group: np.ndarray):
    unique = np.unique(group)
    by_group = np.array([delta[group == gid].mean() for gid in unique])
    rng = np.random.default_rng(SEED + 1)
    draws = rng.integers(0, len(unique), size=(5000, len(unique)))
    return np.quantile(by_group[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main():
    asset_sha = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if asset_sha != EXPECTED_SHA:
        raise ValueError("published rank_train asset SHA mismatch")
    with np.load(ASSET) as z:
        x, y, mode, group = (z[k] for k in ("X", "Y", "M", "G"))
    if x.shape != (142762, 31) or len(y) != len(x):
        raise ValueError("published feature schema changed")
    unique = np.unique(group)
    if len(unique) != 819 or not np.array_equal(unique, np.arange(819)):
        raise ValueError("published group IDs changed")
    rng = np.random.default_rng(SEED)
    heldout = np.sort(rng.permutation(unique)[:164])
    valid = np.isin(group, heldout)
    train = ~valid
    if set(group[train]) & set(group[valid]):
        raise ValueError("query group leakage")
    params = dict(max_depth=6, max_iter=500, learning_rate=0.03,
                  min_samples_leaf=80, l2_regularization=1.0)
    predictions = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for prior in (0.30, 0.60):
        weights = np.where(mode[train] == 0, prior, 1 - prior)
        model = HistGradientBoostingClassifier(random_state=0, **params)
        model.fit(x[train], y[train], sample_weight=weights)
        name = f"prior_{prior:.2f}"
        predictions[name] = model.predict_proba(x[valid])[:, 1]
        joblib.dump(model, OUTPUT / f"{name}.joblib")
    predictions["dual_prior_mean"] = (predictions["prior_0.30"] + predictions["prior_0.60"]) / 2
    predictions["mode_conditioned"] = np.where(
        mode[valid] == 0, predictions["prior_0.60"], predictions["prior_0.30"]
    )
    # Columns 21:25 are the four fragmentation features in the published V17 source.
    no_fragment_cols = np.r_[0:21, 25:31]
    no_fragment = HistGradientBoostingClassifier(random_state=0, **params)
    no_fragment.fit(
        x[train][:, no_fragment_cols], y[train],
        sample_weight=np.where(mode[train] == 0, 0.30, 0.70),
    )
    predictions["no_fragment_prior_0.30"] = no_fragment.predict_proba(
        x[valid][:, no_fragment_cols]
    )[:, 1]
    joblib.dump(no_fragment, OUTPUT / "no_fragment_prior_0.30.joblib")
    predictions["library_only"] = x[valid, 0]
    predictions["analog_only"] = x[valid, 5]
    scores = {}
    reference_keys = None
    for name, pred in predictions.items():
        keys, rr = score_queries(pred, y[valid], mode[valid], group[valid])
        if reference_keys is None:
            reference_keys = keys
        elif not np.array_equal(reference_keys, keys):
            raise ValueError("paired query order changed")
        scores[name] = rr
    metrics = {}
    for name, rr in scores.items():
        metrics[name] = {
            "mrr_at_25": float(rr.mean()),
            "mode_0_mrr_at_25": float(rr[reference_keys[:, 0] == 0].mean()),
            "mode_1_mrr_at_25": float(rr[reference_keys[:, 0] == 1].mean()),
            "top1_rate": float(np.mean(rr == 1)),
            "top25_recall": float(np.mean(rr > 0)),
        }
    comparisons = {}
    for left, right in (
        ("dual_prior_mean", "prior_0.30"),
        ("mode_conditioned", "prior_0.30"),
        ("prior_0.30", "no_fragment_prior_0.30"),
    ):
        delta = scores[left] - scores[right]
        comparisons[f"{left}_minus_{right}"] = {
            "mean": float(delta.mean()),
            "paired_group_bootstrap_95_ci": paired_ci(delta, reference_keys[:, 1]),
        }
    report = {
        "scope": "published V17 features; paired G-ID holdout; G is not verified molecular identity",
        "asset_sha256": asset_sha,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_seed": SEED,
        "train_group_count": int(len(unique) - len(heldout)),
        "valid_group_count": int(len(heldout)),
        "valid_query_count": int(len(reference_keys)),
        "train_candidate_count": int(train.sum()),
        "valid_candidate_count": int(valid.sum()),
        "heldout_group_ids": heldout.tolist(),
        "params": params,
        "no_fragment_columns": no_fragment_cols.tolist(),
        "metrics": metrics,
        "comparisons": comparisons,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "heldout_group_ids"}, indent=2))


if __name__ == "__main__":
    main()
