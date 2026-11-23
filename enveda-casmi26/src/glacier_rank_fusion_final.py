"""Evaluate the confirm-selected GLACIER rank fusion on frozen final controls."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, bootstrap, rr, sha
from glacier_local_probe import PKG, candidate_meta, frozen, paths, rerank_multi
from glacier_rank_fusion_probe import OUT, WEIGHTS, rank_order


def main() -> None:
    selection = json.loads((OUT / "confirm.json").read_text())
    if selection["source_sha256"] != sha(Path(rank_order.__code__.co_filename)):
        raise ValueError("Confirm fusion source SHA changed")
    weight = selection["selected_weight"]
    if weight not in WEIGHTS:
        raise ValueError("No confirm-positive weight selected")
    cand, _ = paths("final")
    gl_dir = PKG / "final_probe"
    prior = json.loads((gl_dir / "report.json").read_text())
    inputs = {"candidates": cand, "gl_scores": gl_dir / "gl_out.json",
              "gl_report": gl_dir / "report.json", "gl_rows": gl_dir / "rows.json",
              "confirm_selection": OUT / "confirm.json"}
    if sha(cand) != prior["sha256"]["candidates"] or sha(inputs["gl_scores"]) != prior["sha256"]["scores"]:
        raise ValueError("Frozen candidate or GLACIER score SHA changed")
    controls, labels = frozen(cand)
    meta = candidate_meta({item[0] for row in controls for item in row["a_top25"]})
    scores_by_qid = json.loads(inputs["gl_scores"].read_text())
    gl_rows = {int(row["query_id"]): row for row in json.loads(inputs["gl_rows"].read_text())}
    rows = []
    for row in controls:
        qid = int(row["query_id"])
        keys = [item[0] for item in row["a_top25"]]
        scores = [float(item[1]) for item in row["a_top25"]]
        formulas = [meta[key]["formula"] for key in keys]
        smiles = [meta[key]["smiles"] for key in keys]
        truth = labels[qid]["truth_key"]
        if qid not in gl_rows or gl_rows[qid]["identity_group"] != labels[qid]["identity_group"]:
            raise ValueError(f"Frozen row identity mismatch: {qid}")
        gl_order = rerank_multi(smiles, keys, scores, formulas, [scores_by_qid[str(qid)]], [0.5], top_n=25)
        gl_rr = rr([keys[pos] for pos in gl_order], truth)
        if not np.isclose(gl_rr, gl_rows[qid]["gl_rr"]):
            raise ValueError(f"Frozen GLACIER replay mismatch: {qid}")
        ranked_order = rank_order(keys, scores, formulas, smiles, scores_by_qid[str(qid)], weight)
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_candidates": truth in keys, "base_rr": rr(keys, truth),
                     "gl_rr": gl_rr, "rank_rr": rr([keys[pos] for pos in ranked_order], truth)})
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    base = np.asarray([np.mean([row["base_rr"] for row in group]) for group in groups.values()])
    gl = np.asarray([np.mean([row["gl_rr"] for row in group]) for group in groups.values()])
    ranked = np.asarray([np.mean([row["rank_rr"] for row in group]) for group in groups.values()])
    if not np.isclose(base.mean(), prior["base_mrr"]) or not np.isclose(gl.mean(), prior["gl_mrr"]):
        raise ValueError("Frozen baseline or GLACIER report mismatch")
    covered = np.asarray([group[0]["truth_in_candidates"] for group in groups.values()])
    delta = ranked - gl
    result = {"partition": "final", "selected_weight": weight, "molecules": len(groups),
              "truth_coverage": float(covered.mean()), "base_mrr": float(base.mean()),
              "gl_mrr": float(gl.mean()), "rank_mrr": float(ranked.mean()),
              "delta_vs_glacier": float(delta.mean()), "bootstrap95": bootstrap(delta),
              "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum()),
              "conditional_mrr": {"glacier": float(gl[covered].mean()),
                                  "rank": float(ranked[covered].mean())},
              "inputs_sha256": {name: sha(path) for name, path in inputs.items()},
              "source_sha256": sha(Path(__file__)),
              "scope": "Frozen final pool controls, one weight selected on confirm; structural split is disjoint, but final has been viewed in preceding GLACIER experiments."}
    path = OUT / "final.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
