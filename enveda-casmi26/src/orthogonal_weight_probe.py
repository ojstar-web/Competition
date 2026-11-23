"""Select ICEBERG/GLACIER fusion weights on confirm, then replay on final."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, bootstrap, rr, sha
from glacier_local_probe import PKG, candidate_meta, frozen, paths, rerank_multi


OUT = ROOT / "outputs/orthogonal_weight_20261119_01"
GRID = [(g, i) for g in (0.25, 0.5, 0.75, 1.0) for i in (0.0, 0.25, 0.5)]
REFERENCE = (0.5, 0.0)


def load_rows(partition: str) -> tuple[list[dict], dict[str, str]]:
    cand, ice_dir = paths(partition)
    gl_dir = PKG / f"{partition}_probe"
    ice_report = json.loads((ice_dir / "report.json").read_text())
    gl_report = json.loads((gl_dir / "report.json").read_text())
    inputs = {
        "candidates": cand,
        "ice_scores": ice_dir / "ice_out.json",
        "gl_scores": gl_dir / "gl_out.json",
        "ice_report": ice_dir / "report.json",
        "gl_report": gl_dir / "report.json",
    }
    expected = {
        "candidates": gl_report["sha256"]["candidates"],
        "ice_scores": gl_report["sha256"]["ice_scores"],
        "gl_scores": gl_report["sha256"]["scores"],
    }
    for name, digest in expected.items():
        if sha(inputs[name]) != digest:
            raise ValueError(f"Frozen {name} changed")
    controls, labels = frozen(cand)
    meta = candidate_meta({x[0] for row in controls for x in row["a_top25"]})
    ice_scores = json.loads(inputs["ice_scores"].read_text())
    gl_scores = json.loads(inputs["gl_scores"].read_text())
    replay = {int(x["query_id"]): x for x in json.loads((gl_dir / "rows.json").read_text())}
    rows = []
    for row in controls:
        qid = int(row["query_id"])
        keys = [x[0] for x in row["a_top25"]]
        scores = [float(x[1]) for x in row["a_top25"]]
        smis = [meta[k]["smiles"] for k in keys]
        formulas = [meta[k]["formula"] for k in keys]
        truth = labels[qid]["truth_key"]
        if qid not in replay or replay[qid]["identity_group"] != labels[qid]["identity_group"]:
            raise ValueError(f"Frozen row identity mismatch: {qid}")
        results = {}
        for gl_weight, ice_weight in GRID:
            order = rerank_multi(smis, keys, scores, formulas,
                                 [gl_scores[str(qid)], ice_scores[str(qid)]],
                                 [gl_weight, ice_weight], top_n=25)
            results[f"{gl_weight:g}:{ice_weight:g}"] = rr([keys[i] for i in order], truth)
        if not np.isclose(results["0.5:0"], replay[qid]["gl_rr"]):
            raise ValueError(f"Frozen GLACIER replay mismatch: {qid}")
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_candidates": truth in keys, "base_rr": rr(keys, truth),
                     "results": results})
    if not np.isclose(np.mean([r["base_rr"] for r in rows]), gl_report["base_mrr"]):
        raise ValueError("Frozen baseline mismatch")
    if not np.isclose(np.mean([r["results"]["0.5:0"] for r in rows]), gl_report["gl_mrr"]):
        raise ValueError("Frozen GLACIER MRR mismatch")
    return rows, {name: sha(path) for name, path in inputs.items()}


def summarize(rows: list[dict], selected: tuple[float, float]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    def values(weight: tuple[float, float]) -> np.ndarray:
        key = f"{weight[0]:g}:{weight[1]:g}"
        return np.array([np.mean([r["results"][key] for r in group]) for group in groups.values()])
    selected_mrr = values(selected)
    reference_mrr = values(REFERENCE)
    delta = selected_mrr - reference_mrr
    covered = np.array([g[0]["truth_in_candidates"] for g in groups.values()])
    return {"molecules": len(groups), "truth_coverage": float(covered.mean()),
            "reference_mrr": float(reference_mrr.mean()), "selected_mrr": float(selected_mrr.mean()),
            "delta": float(delta.mean()), "bootstrap95": bootstrap(delta),
            "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum()),
            "conditional_mrr": {"reference": float(reference_mrr[covered].mean()),
                                "selected": float(selected_mrr[covered].mean())}}


def main(partition: str) -> None:
    OUT.mkdir(exist_ok=True)
    rows, inputs = load_rows(partition)
    if partition == "confirm":
        grid = []
        for weight in GRID:
            result = summarize(rows, weight)
            grid.append({"weights": list(weight), "mrr": result["selected_mrr"],
                         "delta_vs_glacier": result["delta"]})
        # Predeclared selection: highest confirm MRR; prefer less ICEBERG, then less GLACIER on ties.
        winner = max(grid, key=lambda x: (x["mrr"], -x["weights"][1], -x["weights"][0]))
        selected = tuple(winner["weights"])
    else:
        selection = json.loads((OUT / "confirm.json").read_text())
        if selection["source_sha256"] != sha(Path(__file__)):
            raise ValueError("Selection source changed")
        selected = tuple(selection["selected_weights"])
        if selected not in GRID:
            raise ValueError("Selected weights outside frozen grid")
        grid = None
    report = {"partition": partition, "selected_weights": list(selected),
              "reference_weights": list(REFERENCE), "comparison": summarize(rows, selected),
              "grid": grid, "selection_rule": "max confirm MRR; tie: smaller ICEBERG then GLACIER weight",
              "inputs_sha256": inputs, "source_sha256": sha(Path(__file__)),
              "scope": "Same frozen pool-control candidate top25; final weights selected on confirm only."}
    path = OUT / f"{partition}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"partition": partition, "weights": report["selected_weights"], **report["comparison"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("confirm", "final"))
    main(parser.parse_args().partition)
