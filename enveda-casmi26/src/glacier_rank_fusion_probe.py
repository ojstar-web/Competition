"""Screen same-formula rank fusion against the frozen GLACIER score fusion."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, bootstrap, rr, sha
from glacier_local_probe import PKG, candidate_meta, frozen, paths, rerank_multi


OUT = ROOT / "outputs/glacier_rank_fusion_20261119_01"
WEIGHTS = (0.25, 0.5, 1.0, 2.0)


def rank_order(keys: list[str], scores: list[float], formulas: list[str],
               smiles: list[str], gl: dict, weight: float) -> list[int]:
    order = list(range(len(keys)))
    groups = defaultdict(list)
    for pos, formula in enumerate(formulas[:25]):
        groups[formula].append(pos)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        values = [gl.get(smiles[pos]) for pos in positions]
        if any(value is None or not np.isfinite(value) for value in values):
            continue
        n = len(positions)
        base_rank = np.empty(n, dtype=int)
        gl_rank = np.empty(n, dtype=int)
        base_rank[np.argsort(-np.asarray([scores[pos] for pos in positions]), kind="stable")] = np.arange(n)
        gl_rank[np.argsort(-np.asarray(values), kind="stable")] = np.arange(n)
        fused = base_rank + weight * gl_rank
        chosen = [positions[pos] for pos in np.argsort(fused, kind="stable")]
        for slot, src in zip(positions, chosen):
            order[slot] = src
    return order


def main() -> None:
    cand, _ = paths("confirm")
    gl_dir = PKG / "confirm_probe"
    report = json.loads((gl_dir / "report.json").read_text())
    inputs = {"candidates": cand, "gl_scores": gl_dir / "gl_out.json",
              "gl_report": gl_dir / "report.json", "gl_rows": gl_dir / "rows.json"}
    if sha(cand) != report["sha256"]["candidates"] or sha(inputs["gl_scores"]) != report["sha256"]["scores"]:
        raise ValueError("Frozen candidate or GLACIER score SHA changed")
    controls, labels = frozen(cand)
    meta = candidate_meta({item[0] for row in controls for item in row["a_top25"]})
    gl_scores = json.loads(inputs["gl_scores"].read_text())
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
        gl_order = rerank_multi(smiles, keys, scores, formulas, [gl_scores[str(qid)]], [0.5], top_n=25)
        gl_rr = rr([keys[pos] for pos in gl_order], truth)
        if not np.isclose(gl_rr, gl_rows[qid]["gl_rr"]):
            raise ValueError(f"Frozen GLACIER replay mismatch: {qid}")
        vals = {str(weight): rr([keys[pos] for pos in rank_order(
            keys, scores, formulas, smiles, gl_scores[str(qid)], weight)], truth)
                for weight in WEIGHTS}
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_candidates": truth in keys, "gl_rr": gl_rr, "rank_rr": vals})
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    gl = np.asarray([np.mean([row["gl_rr"] for row in group]) for group in groups.values()])
    if not np.isclose(gl.mean(), report["gl_mrr"]):
        raise ValueError("Frozen GLACIER MRR mismatch")
    grid = []
    for weight in WEIGHTS:
        ranked = np.asarray([np.mean([row["rank_rr"][str(weight)] for row in group]) for group in groups.values()])
        delta = ranked - gl
        grid.append({"weight": weight, "rank_mrr": float(ranked.mean()),
                     "delta_vs_glacier": float(delta.mean()), "bootstrap95": bootstrap(delta),
                     "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum())})
    winner = max(grid, key=lambda x: (x["rank_mrr"], -x["weight"]))
    selected = winner["weight"] if winner["delta_vs_glacier"] > 0 else None
    result = {"partition": "confirm", "molecules": len(groups),
              "truth_coverage": float(np.mean([group[0]["truth_in_candidates"] for group in groups.values()])),
              "gl_mrr": float(gl.mean()), "grid": grid, "selected_weight": selected,
              "selection_rule": "highest confirm MRR; tie: lower weight; advance only if confirm delta positive",
              "source_sha256": sha(Path(__file__)),
              "inputs_sha256": {name: sha(path) for name, path in inputs.items()},
              "scope": "Frozen pool-control top25; Borda-style within-formula rank fusion only when all GLACIER scores exist."}
    OUT.mkdir(exist_ok=False)
    (OUT / "confirm.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
