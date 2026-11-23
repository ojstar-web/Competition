"""Prepare and score the public ICEBERG model on frozen fold1 pool controls."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bond_fragment_probe import ROOT, FOLD, CAMPAIGN, bootstrap, rr, sha


PKG = ROOT / "outputs/orthogonal_simulator_20261119_01"
LAMBDA = 0.5


def paths(partition: str) -> tuple[Path, Path]:
    if partition == "confirm":
        return CAMPAIGN / "confirm_pool20_pc5" / "candidates.jsonl", PKG / "confirm_probe"
    return ROOT / "outputs/bond_fragment_final_candidates_20261119_01/candidates.jsonl", PKG / "final_probe"


def frozen(cand: Path) -> tuple[list[dict], dict[int, dict]]:
    receipt = json.loads((cand.parent / "generation.json").read_text())
    if sha(cand) != receipt["candidate_sha256"]:
        raise ValueError("Confirm candidate SHA differs from generation receipt")
    labels = {int(r["query_id"]): r for r in pq.read_table(FOLD / "scoring_labels_corrected.parquet").to_pylist()}
    rows = [json.loads(s) for s in cand.read_text().splitlines()]
    controls = [r for r in rows if labels[int(r["query_id"])]["arm"] == "control"]
    if len(controls) != 128 or len({r["query_id"] for r in controls}) != 128:
        raise ValueError("Expected 128 distinct frozen control queries")
    return controls, labels


def candidate_meta(keys: set[str]) -> dict[str, dict]:
    table = pq.read_table(FOLD / "reduced_pool/pool_meta.parquet", columns=["key", "smiles", "formula"])
    meta = {r["key"]: r for r in table.to_pylist() if r["key"] in keys}
    if keys - meta.keys():
        raise ValueError("Candidate-to-structure mapping incomplete")
    return meta


def prepare(partition: str) -> None:
    cand, out = paths(partition)
    controls, labels = frozen(cand)
    keys = {x[0] for r in controls for x in r["a_top25"]}
    meta = candidate_meta(keys)
    needed = {int(r["query_id"]) for r in controls}
    spectra = defaultdict(list)
    pf = pq.ParquetFile(FOLD / "query_features.parquet")
    for batch in pf.iter_batches(batch_size=512, columns=["query_id", "ms2_mzs", "ms2_normalized_intensities",
                                                    "precursor_mz", "adduct", "ionization_mode", "collision_energy_ev"]):
        for row in batch.to_pylist():
            qid = int(row["query_id"])
            if qid not in needed:
                continue
            if len(row["ms2_mzs"]) != len(row["ms2_normalized_intensities"]):
                raise ValueError(f"Spectrum peak arrays differ for query {qid}")
            if row["ionization_mode"] == "positive" and row["adduct"] in ("[M+H]+", "[M+Na]+"):
                spectra[qid].append({"mz": row["ms2_mzs"], "it": row["ms2_normalized_intensities"],
                                     "prec": row["precursor_mz"], "adduct": row["adduct"],
                                     "mode": row["ionization_mode"], "instrument": "timsTOF",
                                     "ce_ev": row["collision_energy_ev"]})
    items = []
    for r in controls:
        qid = int(r["query_id"])
        order = [x[0] for x in r["a_top25"]]
        count = Counter(meta[key]["formula"] for key in order)
        smis = list(dict.fromkeys(meta[key]["smiles"] for key in order if count[meta[key]["formula"]] >= 2))
        items.append({"mid": str(qid), "cands": smis, "spectra": spectra[qid]})
    out.mkdir(exist_ok=False)
    (out / "ice_in.json").write_text(json.dumps(items, separators=(",", ":")) + "\n")
    manifest = {"partition": f"fold1 {partition} pool controls", "queries": len(items),
                "queries_with_covered_spectra": sum(bool(x["spectra"]) for x in items),
                "queries_with_multi_candidate_formula": sum(bool(x["cands"]) for x in items),
                "covered_spectra": sum(len(x["spectra"]) for x in items),
                "candidates": sum(len(x["cands"]) for x in items),
                "instrument_assumption": "timsTOF per source audit; mapped to QTOF by runner",
                "missing_collision_energy": "runner uses its fixed CE summation path",
                "candidate_rule": "same-formula groups of size >=2 in frozen pool top25",
                "planned_fusion_lambda": LAMBDA,
                "sha256": {"candidates": sha(cand), "features": sha(FOLD / "query_features.parquet"),
                           "labels": sha(FOLD / "scoring_labels_corrected.parquet"),
                           "pool_meta": sha(FOLD / "reduced_pool/pool_meta.parquet"),
                           "ice_input": sha(out / "ice_in.json"), "prepare_source": sha(Path(__file__))}}
    (out / "prepare.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("queries", "queries_with_covered_spectra", "queries_with_multi_candidate_formula", "covered_spectra", "candidates")}))


def z(values: list[float | None]) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in values])
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return np.zeros(len(arr))
    sd = arr[ok].std(ddof=1)
    if sd <= 0:
        return np.zeros(len(arr))
    return np.where(ok, (arr - arr[ok].mean()) / sd, 0.0)


def main_score(partition: str) -> None:
    cand, out = paths(partition)
    scores_path = out / "ice_out.json"
    prep = json.loads((out / "prepare.json").read_text())
    if sha(out / "ice_in.json") != prep["sha256"]["ice_input"] or sha(cand) != prep["sha256"]["candidates"]:
        raise ValueError("ICEBERG input or candidates changed after preparation")
    run = json.loads((scores_path.with_name(scores_path.name + ".meta.json")).read_text())
    prepared = json.loads((out / "ice_in.json").read_text())
    eligible = sum(bool(item["spectra"]) and bool(item["cands"]) for item in prepared)
    if run["status"] != "ok" or run["n_mols_scored"] != eligible or run["errors"]:
        raise ValueError(f"ICEBERG runner incomplete: status={run['status']}, scored={run['n_mols_scored']}, errors={run['errors'][:3]}")
    scores = json.loads(scores_path.read_text())
    controls, labels = frozen(cand)
    keys = {x[0] for r in controls for x in r["a_top25"]}
    meta = candidate_meta(keys)
    msgym = set(pq.read_table(PKG / "msgym_pool.parquet", columns=["ik14_std"])["ik14_std"].to_pylist())
    rows = []
    for r in controls:
        qid = int(r["query_id"])
        original = [x[0] for x in r["a_top25"]]
        if str(qid) not in scores:
            raise ValueError(f"Missing ICEBERG scores for {qid}")
        sc = scores[str(qid)]
        order = list(range(len(original)))
        groups = defaultdict(list)
        for i, key in enumerate(original):
            groups[meta[key]["formula"]].append(i)
        moved = 0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            ice = [sc.get(meta[original[i]]["smiles"]) for i in indices]
            if sum(v is not None for v in ice) < 2:
                continue
            base_z = z([float(r["a_top25"][i][1]) for i in indices])
            ice_z = z(ice)
            fused = base_z + LAMBDA * ice_z
            chosen = [indices[j] for j in np.argsort(-fused, kind="stable")]
            for slot, src in zip(indices, chosen):
                order[slot] = src
                moved += slot != src
        ranked = [original[i] for i in order]
        truth = labels[qid]["truth_key"]
        rows.append({"query_id": qid, "identity_group": int(labels[qid]["identity_group"]),
                     "truth_in_msgym_pool": truth in msgym, "truth_in_candidates": truth in original,
                     "base_rr": rr(original, truth), "ice_rr": rr(ranked, truth), "moved_slots": moved,
                     "n_ice_scored": sum(v is not None for v in sc.values())})
    base = np.array([r["base_rr"] for r in rows])
    ice = np.array([r["ice_rr"] for r in rows])
    delta = ice - base
    covered = np.array([r["truth_in_candidates"] for r in rows])
    strata = {}
    for name, mask in (("present_in_msgym_pool", np.array([r["truth_in_msgym_pool"] for r in rows])),
                       ("absent_from_msgym_pool", np.array([not r["truth_in_msgym_pool"] for r in rows]))):
        d = delta[mask]
        strata[name] = {"molecules": int(mask.sum()), "base_mrr": float(base[mask].mean()),
                        "ice_mrr": float(ice[mask].mean()), "delta": float(d.mean()),
                        "bootstrap95": bootstrap(d), "improved": int((d > 0).sum()), "worsened": int((d < 0).sum())}
    report = {"scope": f"fold1 {partition} pool controls, frozen top25 candidates; fixed fusion weight {LAMBDA}",
              "molecules": len(rows), "base_mrr": float(base.mean()), "ice_mrr": float(ice.mean()),
              "delta": float(delta.mean()), "bootstrap95": bootstrap(delta),
              "improved": int((delta > 0).sum()), "worsened": int((delta < 0).sum()),
              "queries_with_rank_change": sum(r["moved_slots"] > 0 for r in rows),
              "truth_coverage": float(np.mean([r["truth_in_candidates"] for r in rows])),
              "conditional_mrr": {"baseline": float(base[covered].mean()),
                                  "ice": float(ice[covered].mean())},
              "msgym_pool_strata": strata,
              "runner": {k: run[k] for k in ("status", "n_mols_covered", "n_mols_scored", "n_cands", "n_scored", "n_pred", "n_pred_fail", "seconds", "device", "rdkit")},
              "sha256": {"scores": sha(scores_path), "runner_meta": sha(scores_path.with_name(scores_path.name + ".meta.json")),
                           "input": sha(out / "ice_in.json"), "source": sha(Path(__file__)),
                           "msgym_pool": sha(PKG / "msgym_pool.parquet")},
              "caveat": "MassSpecGym compound-pool membership does not prove a spectrum was or was not in model training."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("molecules", "base_mrr", "ice_mrr", "delta", "bootstrap95", "improved", "worsened")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "score"))
    p.add_argument("--partition", choices=("confirm", "final"), default="confirm")
    args = p.parse_args()
    prepare(args.partition) if args.stage == "prepare" else main_score(args.partition)
