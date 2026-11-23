"""Independent final-partition check of the fixed 0.25 one-bond fusion rule."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bond_fragment_probe import ROOT, FOLD, CAMPAIGN, bootstrap, fragment_score, one_bond_masses, query_rows, rr, sha


CANDIDATES = ROOT / "outputs/bond_fragment_final_candidates_20261119_01"
OUT = ROOT / "outputs/bond_fragment_final_20261119_01"
WEIGHT = 0.25
MSGYM_POOL = ROOT / "outputs/orthogonal_simulator_20261119_01/msgym_pool.parquet"


def main() -> None:
    receipt = json.loads((CANDIDATES / "generation.json").read_text())
    paths = {
        "candidates": CANDIDATES / "candidates.jsonl",
        "ids": CAMPAIGN / "final_ids.json",
        "source": FOLD / "generate_candidates.py",
        "features": FOLD / "query_features.parquet",
    }
    for name, field in (("candidates", "candidate_sha256"), ("ids", "ids_sha256"),
                        ("source", "source_sha256"), ("features", "feature_sha256")):
        if sha(paths[name]) != receipt[field]:
            raise ValueError(f"Final generator {name} identity mismatch")
    frozen_ids = json.loads(paths["ids"].read_text())["query_ids"]
    candidates = [json.loads(line) for line in paths["candidates"].read_text().splitlines()]
    if sorted(int(r["query_id"]) for r in candidates) != sorted(frozen_ids):
        raise ValueError("Final candidate rows do not match frozen partition IDs")
    labels = {int(r["query_id"]): r for r in pq.read_table(FOLD / "scoring_labels_corrected.parquet").to_pylist()}
    msgym_keys = set(pq.read_table(MSGYM_POOL, columns=["ik14_std"])["ik14_std"].to_pylist())
    control = [r for r in candidates if labels[int(r["query_id"])]["arm"] == "control"]
    spectra = query_rows({int(r["query_id"]) for r in control})
    if len(spectra) != len(control):
        raise ValueError("Final control spectra join is incomplete")
    keys = {x[0] for r in control for x in r["a_top25"]}
    meta = pq.read_table(FOLD / "reduced_pool/pool_meta.parquet", columns=["key", "smiles"])
    smiles = {r["key"]: r["smiles"] for r in meta.to_pylist() if r["key"] in keys}
    if keys - smiles.keys():
        raise ValueError("Final candidate structure mapping is incomplete")
    fragments = {key: one_bond_masses(smiles[key]) for key in sorted(keys)}
    rows = []
    for entry in control:
        qid = int(entry["query_id"])
        truth = labels[qid]["truth_key"]
        original = [x[0] for x in entry["a_top25"]]
        frag = np.array([fragment_score(fragments[key], spectra[qid]) for key in original])
        frag_rank = np.empty(len(original), dtype=int)
        frag_rank[np.argsort(-frag, kind="stable")] = np.arange(len(original))
        combined = 1 / (np.arange(len(original)) + 1) + WEIGHT / (frag_rank + 1)
        ranked = [original[i] for i in np.argsort(-combined, kind="stable")]
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_msgym_pool": truth in msgym_keys,
                     "truth_in_candidates": truth in original,
                     "baseline_rr": rr(original, truth), "fusion_rr": rr(ranked, truth)})
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    base = np.array([np.mean([r["baseline_rr"] for r in group]) for group in groups.values()])
    fused = np.array([np.mean([r["fusion_rr"] for r in group]) for group in groups.values()])
    diff = fused - base
    covered = np.array([group[0]["truth_in_candidates"] for group in groups.values()])
    overlap = np.array([group[0]["truth_in_msgym_pool"] for group in groups.values()])
    strata = {}
    for name, mask in (("present_in_msgym_pool", overlap), ("absent_from_msgym_pool", ~overlap)):
        values = diff[mask]
        strata[name] = {"molecules": int(mask.sum()), "baseline_mrr": float(base[mask].mean()),
                        "fusion_mrr": float(fused[mask].mean()), "delta": float(values.mean()),
                        "bootstrap95": bootstrap(values), "improved": int((values > 0).sum()),
                        "worsened": int((values < 0).sum())}
    report = {"experiment": "independent final-partition one-bond pool-control fusion",
              "fixed_weight": WEIGHT, "molecules": len(groups), "queries": len(rows),
              "truth_in_candidates": float(np.mean([r["truth_in_candidates"] for r in rows])),
              "mrr": {"baseline": float(base.mean()), "fusion": float(fused.mean())},
              "conditional_mrr": {"baseline": float(base[covered].mean()),
                                  "fusion": float(fused[covered].mean())},
              "delta": float(diff.mean()), "bootstrap95": bootstrap(diff),
              "msgym_pool_strata": strata,
              "improved_groups": int((diff > 0).sum()), "worsened_groups": int((diff < 0).sum()),
              "input_sha256": {name: sha(path) for name, path in paths.items()} |
                              {"labels": sha(FOLD / "scoring_labels_corrected.parquet"),
                               "pool_meta": sha(FOLD / "reduced_pool/pool_meta.parquet"),
                               "msgym_pool": sha(MSGYM_POOL),
                               "score_source": sha(Path(__file__)),
                               "fragment_core_source": sha(ROOT / "src/bond_fragment_probe.py")},
              "scope": "In-pool final controls only; candidate list and baseline order are frozen. MassSpecGym pool membership is not proof of a model training example."}
    OUT.mkdir(exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("molecules", "mrr", "delta", "bootstrap95", "improved_groups", "worsened_groups")}))


if __name__ == "__main__":
    main()
