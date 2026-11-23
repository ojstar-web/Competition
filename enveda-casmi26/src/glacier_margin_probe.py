"""Screen a base-score margin gate for frozen GLACIER same-formula reranking."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, bootstrap, rr, sha
from glacier_local_probe import PKG, candidate_meta, frozen, paths, rerank_multi


OUT = ROOT / "outputs/glacier_margin_20261119_01"
THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def gated_order(keys: list[str], base: list[float], formulas: list[str],
                gl_scores: dict, smiles: list[str], threshold: float) -> list[int]:
    glacier = rerank_multi(smiles, keys, base, formulas, [gl_scores], [0.5], top_n=25)
    order = list(range(len(keys)))
    groups = defaultdict(list)
    for pos, formula in enumerate(formulas[:25]):
        groups[formula].append(pos)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        scores = np.asarray([base[pos] for pos in positions], dtype=float)
        sd = scores.std(ddof=1)
        margin = (np.sort(scores)[-1] - np.sort(scores)[-2]) / sd if sd > 0 else 0.0
        if margin <= threshold:
            for pos in positions:
                order[pos] = glacier[pos]
    return order


def evaluate(partition: str) -> tuple[list[dict], dict[str, str]]:
    cand, _ = paths(partition)
    gl_dir = PKG / f"{partition}_probe"
    source_report = json.loads((gl_dir / "report.json").read_text())
    inputs = {"candidates": cand, "gl_scores": gl_dir / "gl_out.json",
              "gl_report": gl_dir / "report.json", "gl_rows": gl_dir / "rows.json"}
    if sha(cand) != source_report["sha256"]["candidates"] or sha(inputs["gl_scores"]) != source_report["sha256"]["scores"]:
        raise ValueError("Frozen candidate or GLACIER score SHA changed")
    controls, labels = frozen(cand)
    meta = candidate_meta({item[0] for row in controls for item in row["a_top25"]})
    gl_scores = json.loads(inputs["gl_scores"].read_text())
    gl_rows = {int(row["query_id"]): row for row in json.loads(inputs["gl_rows"].read_text())}
    rows = []
    for row in controls:
        qid = int(row["query_id"])
        keys = [item[0] for item in row["a_top25"]]
        base = [float(item[1]) for item in row["a_top25"]]
        formulas = [meta[key]["formula"] for key in keys]
        smiles = [meta[key]["smiles"] for key in keys]
        truth = labels[qid]["truth_key"]
        if qid not in gl_rows or gl_rows[qid]["identity_group"] != labels[qid]["identity_group"]:
            raise ValueError(f"Frozen row identity mismatch: {qid}")
        gl_order = rerank_multi(smiles, keys, base, formulas, [gl_scores[str(qid)]], [0.5], top_n=25)
        gl_rr = rr([keys[pos] for pos in gl_order], truth)
        if not np.isclose(gl_rr, gl_rows[qid]["gl_rr"]):
            raise ValueError(f"Frozen GLACIER replay mismatch: {qid}")
        gated = {str(threshold): rr([keys[pos] for pos in gated_order(
            keys, base, formulas, gl_scores[str(qid)], smiles, threshold)], truth)
                 for threshold in THRESHOLDS}
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_candidates": truth in keys, "base_rr": rr(keys, truth),
                     "gl_rr": gl_rr, "gated_rr": gated})
    if not np.isclose(np.mean([row["gl_rr"] for row in rows]), source_report["gl_mrr"]):
        raise ValueError("Frozen GLACIER MRR mismatch")
    return rows, {name: sha(path) for name, path in inputs.items()}


def summarize(rows: list[dict], threshold: float) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["identity_group"]].append(row)
    base = np.array([np.mean([row["base_rr"] for row in group]) for group in grouped.values()])
    gl = np.array([np.mean([row["gl_rr"] for row in group]) for group in grouped.values()])
    gated = np.array([np.mean([row["gated_rr"][str(threshold)] for row in group]) for group in grouped.values()])
    covered = np.array([group[0]["truth_in_candidates"] for group in grouped.values()])
    delta = gated - gl
    return {"molecules": len(grouped), "truth_coverage": float(covered.mean()),
            "base_mrr": float(base.mean()), "gl_mrr": float(gl.mean()),
            "gated_mrr": float(gated.mean()), "delta_vs_glacier": float(delta.mean()),
            "bootstrap95": bootstrap(delta), "improved": int((delta > 0).sum()),
            "worsened": int((delta < 0).sum()),
            "conditional_mrr": {"glacier": float(gl[covered].mean()),
                                "gated": float(gated[covered].mean())}}


def main(partition: str) -> None:
    rows, inputs = evaluate(partition)
    if partition == "confirm":
        grid = [{"threshold": threshold, **summarize(rows, threshold)} for threshold in THRESHOLDS]
        winner = max(grid, key=lambda x: (x["gated_mrr"], x["threshold"]))
        selected = winner["threshold"] if winner["delta_vs_glacier"] > 0 else None
    else:
        prior = json.loads((OUT / "confirm.json").read_text())
        if prior["source_sha256"] != sha(Path(__file__)):
            raise ValueError("Confirm source SHA changed")
        selected = prior["selected_threshold"]
        if selected not in THRESHOLDS:
            raise ValueError("No confirm-positive threshold selected")
        grid = None
    report = {"partition": partition, "selected_threshold": selected,
              "selection_rule": "highest confirm MRR; tie: larger threshold; advance only on positive confirm delta",
              "grid": grid, "comparison": summarize(rows, selected) if selected is not None else None,
              "inputs_sha256": inputs, "source_sha256": sha(Path(__file__)),
              "scope": "Frozen pool-control top25; gate same-formula GLACIER reranking by standardized base top-two score margin."}
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{partition}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"partition": partition, "selected_threshold": selected,
                      "comparison": report["comparison"], "grid": grid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("confirm", "final"))
    main(parser.parse_args().partition)
