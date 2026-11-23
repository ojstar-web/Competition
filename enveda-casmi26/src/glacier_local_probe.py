"""Evaluate fixed GLACIER same-formula fusion on frozen pool controls."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, FOLD, bootstrap, rr, sha
from iceberg_local_probe import candidate_meta, frozen, paths


PKG = ROOT / "outputs/orthogonal_glacier_20261119_01"
sys.path.insert(0, str(PKG))
from gl_fuse import rerank_multi  # noqa: E402


def main(partition: str) -> None:
    cand, ice_dir = paths(partition)
    out = PKG / f"{partition}_probe"
    ice_prep = json.loads((ice_dir / "prepare.json").read_text())
    if sha(cand) != ice_prep["sha256"]["candidates"] or sha(ice_dir / "ice_in.json") != ice_prep["sha256"]["ice_input"]:
        raise ValueError("Frozen candidates or GLACIER input changed")
    run = json.loads((out / "gl_out.json.meta.json").read_text())
    items = json.loads((ice_dir / "ice_in.json").read_text())
    expected = sum(bool(x["spectra"]) and bool(x["cands"]) for x in items)
    if run["status"] != "ok" or run["n_mols_scored"] != expected or run["errors"] or run["n_pred_fail"]:
        raise ValueError(f"GLACIER runner incomplete: {run['status']}, {run['n_mols_scored']}/{expected}, errors={run['errors'][:3]}")
    scores = json.loads((out / "gl_out.json").read_text())
    ice_report = json.loads((ice_dir / "report.json").read_text())
    if sha(ice_dir / "ice_out.json") != ice_report["sha256"]["scores"]:
        raise ValueError("Frozen ICEBERG scores changed")
    ice_scores = json.loads((ice_dir / "ice_out.json").read_text())
    ice_rows = {int(r["query_id"]): r for r in json.loads((ice_dir / "rows.json").read_text())}
    controls, labels = frozen(cand)
    keys = {x[0] for row in controls for x in row["a_top25"]}
    meta = candidate_meta(keys)
    rows = []
    for row in controls:
        qid = int(row["query_id"])
        original = [x[0] for x in row["a_top25"]]
        if str(qid) not in scores:
            raise ValueError(f"Missing GLACIER scores for {qid}")
        if str(qid) not in ice_scores:
            raise ValueError(f"Missing ICEBERG scores for {qid}")
        if qid not in ice_rows or int(ice_rows[qid]["identity_group"]) != int(labels[qid]["identity_group"]):
            raise ValueError(f"ICEBERG row identity mismatch for {qid}")
        smis = [meta[k]["smiles"] for k in original]
        formulas = [meta[k]["formula"] for k in original]
        base = [float(x[1]) for x in row["a_top25"]]
        order = rerank_multi(smis, original, base, formulas, [scores[str(qid)]], [0.5], top_n=25)
        ice_order = rerank_multi(smis, original, base, formulas, [ice_scores[str(qid)]], [0.5], top_n=25)
        joint_order = rerank_multi(smis, original, base, formulas,
                                   [ice_scores[str(qid)], scores[str(qid)]], [0.5, 0.5], top_n=25)
        truth = labels[qid]["truth_key"]
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_msgym_pool": bool(ice_rows[qid]["truth_in_msgym_pool"]),
                     "truth_in_candidates": truth in original,
                     "base_rr": rr(original, truth), "gl_rr": rr([original[i] for i in order], truth),
                     "ice_rr": rr([original[i] for i in ice_order], truth),
                     "joint_rr": rr([original[i] for i in joint_order], truth),
                     "rank_changed": order != list(range(len(order)))})
    groups = defaultdict(list)
    for row in rows:
        groups[row["identity_group"]].append(row)
    base = np.array([np.mean([r["base_rr"] for r in g]) for g in groups.values()])
    gl = np.array([np.mean([r["gl_rr"] for r in g]) for g in groups.values()])
    joint = np.array([np.mean([r["joint_rr"] for r in g]) for g in groups.values()])
    ice = np.array([np.mean([r["ice_rr"] for r in g]) for g in groups.values()])
    if not np.isclose(base.mean(), ice_report["base_mrr"]) or not np.isclose(ice.mean(), ice_report["ice_mrr"]):
        raise ValueError("ICEBERG replay disagrees with frozen report")
    covered = np.array([g[0]["truth_in_candidates"] for g in groups.values()])
    msgym = np.array([g[0]["truth_in_msgym_pool"] for g in groups.values()])
    delta = gl - base
    strata = {}
    for name, mask in (("present_in_msgym_pool", msgym), ("absent_from_msgym_pool", ~msgym)):
        d = delta[mask]
        strata[name] = {"molecules": int(mask.sum()), "base_mrr": float(base[mask].mean()),
                        "gl_mrr": float(gl[mask].mean()), "delta": float(d.mean()),
                        "bootstrap95": bootstrap(d), "improved": int((d > 0).sum()),
                        "worsened": int((d < 0).sum())}
    report = {"partition": partition, "molecules": len(groups), "queries": len(rows), "weight": 0.5,
              "truth_coverage": float(covered.mean()), "base_mrr": float(base.mean()), "gl_mrr": float(gl.mean()),
              "ice_mrr": float(ice.mean()), "joint_mrr": float(joint.mean()),
              "delta": float(delta.mean()), "bootstrap95": bootstrap(delta),
              "joint_vs_ice": {"delta": float((joint - ice).mean()), "bootstrap95": bootstrap(joint - ice),
                               "improved": int((joint > ice).sum()), "worsened": int((joint < ice).sum())},
              "joint_vs_glacier": {"delta": float((joint - gl).mean()), "bootstrap95": bootstrap(joint - gl),
                                   "improved": int((joint > gl).sum()), "worsened": int((joint < gl).sum())},
              "conditional_mrr": {"baseline": float(base[covered].mean()), "glacier": float(gl[covered].mean())},
              "msgym_pool_strata": strata,
              "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum()),
              "queries_with_rank_change": sum(r["rank_changed"] for r in rows),
              "runner": {k: run[k] for k in ("status", "n_mols_covered", "n_mols_scored", "n_cands", "n_scored", "n_pred", "n_pred_fail", "seconds", "device", "rdkit")},
              "sha256": {"candidates": sha(cand), "input": sha(ice_dir / "ice_in.json"),
                           "ice_scores": sha(ice_dir / "ice_out.json"), "ice_report": sha(ice_dir / "report.json"),
                           "ice_rows": sha(ice_dir / "rows.json"),
                           "scores": sha(out / "gl_out.json"), "runner_meta": sha(out / "gl_out.json.meta.json"),
                           "checkpoint": sha(PKG / "ckpt/glacier.pt"), "source": sha(Path(__file__)),
                           "labels": sha(FOLD / "scoring_labels_corrected.parquet")},
              "scope": "Frozen in-pool controls; fixed weight and no tuning on final.",
              "caveat": "MassSpecGym compound-pool membership does not establish training-spectrum overlap."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("molecules", "base_mrr", "gl_mrr", "ice_mrr", "joint_mrr", "delta", "bootstrap95", "joint_vs_ice", "joint_vs_glacier", "improved", "worsened")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("confirm", "final"))
    main(parser.parse_args().partition)
