"""Evaluate fixed one-bond and ICEBERG rank fusion on frozen pool controls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, FOLD, bootstrap, fragment_score, one_bond_masses, query_rows, rr, sha
from iceberg_local_probe import LAMBDA, PKG, candidate_meta, frozen, paths, z


FRAGMENT_WEIGHT = 0.25


def ice_reorder(order: list[int], base_score: np.ndarray, original: list[str],
                scores: dict[str, float | None], meta: dict[str, dict]) -> list[int]:
    result = order.copy()
    groups = defaultdict(list)
    for slot, idx in enumerate(order):
        groups[meta[original[idx]]["formula"]].append(slot)
    for slots in groups.values():
        if len(slots) < 2:
            continue
        values = [scores.get(meta[original[order[slot]]]["smiles"]) for slot in slots]
        if sum(v is not None for v in values) < 2:
            continue
        combined = z([float(base_score[order[slot]]) for slot in slots]) + LAMBDA * z(values)
        sorted_items = [order[slots[j]] for j in np.argsort(-combined, kind="stable")]
        for slot, item in zip(slots, sorted_items):
            result[slot] = item
    return result


def main(partition: str) -> None:
    cand, ice_out = paths(partition)
    prep = json.loads((ice_out / "prepare.json").read_text())
    ice_report = json.loads((ice_out / "report.json").read_text())
    score_path = ice_out / "ice_out.json"
    if sha(cand) != prep["sha256"]["candidates"] or sha(score_path) != ice_report["sha256"]["scores"]:
        raise ValueError("Candidate or ICEBERG score identity changed")
    controls, labels = frozen(cand)
    keys = {x[0] for entry in controls for x in entry["a_top25"]}
    meta = candidate_meta(keys)
    spectra = query_rows({int(entry["query_id"]) for entry in controls})
    if len(spectra) != len(controls):
        raise ValueError("Control spectra join incomplete")
    fragments = {key: one_bond_masses(meta[key]["smiles"]) for key in sorted(keys)}
    scores = json.loads(score_path.read_text())
    rows = []
    for entry in controls:
        qid = int(entry["query_id"])
        original = [x[0] for x in entry["a_top25"]]
        truth = labels[qid]["truth_key"]
        frag = np.array([fragment_score(fragments[key], spectra[qid]) for key in original])
        frag_rank = np.empty(len(original), dtype=int)
        frag_rank[np.argsort(-frag, kind="stable")] = np.arange(len(original))
        combined = 1 / (np.arange(len(original)) + 1) + FRAGMENT_WEIGHT / (frag_rank + 1)
        fragment_order = list(np.argsort(-combined, kind="stable"))
        base_scores = np.array([float(x[1]) for x in entry["a_top25"]])
        ice_order = ice_reorder(list(range(len(original))), base_scores, original, scores[str(qid)], meta)
        joint_order = ice_reorder(fragment_order, combined, original, scores[str(qid)], meta)
        ranks = {"baseline": original,
                 "fragment": [original[i] for i in fragment_order],
                 "iceberg": [original[i] for i in ice_order],
                 "joint": [original[i] for i in joint_order]}
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_candidates": truth in original,
                     "rr": {name: rr(order, truth) for name, order in ranks.items()}})
    prior = {int(row["query_id"]): row for row in json.loads((ice_out / "rows.json").read_text())}
    if any(row["rr"]["iceberg"] != prior[row["query_id"]]["ice_rr"] for row in rows):
        raise ValueError("ICEBERG-only ranking failed to replay its scored report")
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    values = {name: np.array([np.mean([r["rr"][name] for r in group]) for group in groups.values()])
              for name in ("baseline", "fragment", "iceberg", "joint")}
    coverage = np.array([group[0]["truth_in_candidates"] for group in groups.values()])
    report = {"partition": partition, "molecules": len(groups), "queries": len(rows),
              "weights": {"fragment": FRAGMENT_WEIGHT, "iceberg": LAMBDA},
              "truth_coverage": float(coverage.mean()),
              "mrr": {name: float(v.mean()) for name, v in values.items()},
              "conditional_mrr": {name: float(v[coverage].mean()) for name, v in values.items()},
              "paired_delta": {name: {"mean": float((values["joint"] - values[name]).mean()),
                                      "bootstrap95": bootstrap(values["joint"] - values[name]),
                                      "improved": int((values["joint"] > values[name]).sum()),
                                      "worsened": int((values["joint"] < values[name]).sum())}
                               for name in ("baseline", "fragment", "iceberg")},
              "sha256": {"candidates": sha(cand), "ice_scores": sha(score_path),
                           "ice_report": sha(ice_out / "report.json"),
                           "features": sha(FOLD / "query_features.parquet"),
                           "labels": sha(FOLD / "scoring_labels_corrected.parquet"),
                           "source": sha(Path(__file__))},
              "scope": "Frozen in-pool controls only; no tuning on final partition."}
    out = PKG / f"{partition}_ensemble"
    out.mkdir(exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("molecules", "mrr", "paired_delta")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("confirm", "final"))
    main(parser.parse_args().partition)
