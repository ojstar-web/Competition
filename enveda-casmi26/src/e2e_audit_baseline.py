"""End-to-end audit baseline combining pool20_pc5 (b_top25) and GLACIER 0.5 reranking."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bond_fragment_probe import CAMPAIGN, FOLD, ROOT, bootstrap, rr, sha

GL_PKG = ROOT / "outputs/orthogonal_glacier_20261119_01"
ICE_PKG = ROOT / "outputs/orthogonal_simulator_20261119_01"
OUT_ROOT = ROOT / "outputs/e2e_audit_20261124_01"
PC_META = OUT_ROOT / "pubchem_meta.json"

sys.path.insert(0, str(GL_PKG))
from gl_fuse import rerank_multi, run_gl  # noqa: E402


def cand_path(partition: str) -> Path:
    if partition == "confirm":
        return CAMPAIGN / "confirm_pool20_pc5/candidates.jsonl"
    if partition == "final":
        return ROOT / "outputs/bond_fragment_final_candidates_20261119_01/candidates.jsonl"
    raise ValueError(f"Unsupported partition: {partition}")


def load_partition(partition: str) -> tuple[list[dict], dict[int, dict]]:
    cand = cand_path(partition)
    receipt = json.loads((cand.parent / "generation.json").read_text())
    if sha(cand) != receipt["candidate_sha256"]:
        raise ValueError(f"Candidate SHA mismatch for {partition}")
    labels = {int(r["query_id"]): r for r in pq.read_table(FOLD / "scoring_labels_corrected.parquet").to_pylist()}
    rows = [json.loads(s) for s in cand.read_text().splitlines()]
    return rows, labels


def load_all_meta(keys: set[str]) -> dict[str, dict]:
    table = pq.read_table(FOLD / "reduced_pool/pool_meta.parquet", columns=["key", "smiles", "formula"])
    meta = {r["key"]: {"smiles": r["smiles"], "formula": r["formula"]} for r in table.to_pylist() if r["key"] in keys}
    if PC_META.exists():
        pc = json.loads(PC_META.read_text())
        for k in keys - meta.keys():
            if k in pc:
                meta[k] = pc[k]
    return meta


def load_query_metadata(needed: set[int]) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    spectra = defaultdict(list)
    qmeta: dict[int, dict] = {}
    pf = pq.ParquetFile(FOLD / "query_features.parquet")
    cols = ["query_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", "ionization_mode", "collision_energy_ev"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        for row in batch.to_pylist():
            qid = int(row["query_id"])
            if qid not in needed:
                continue
            if len(row["ms2_mzs"]) != len(row["ms2_normalized_intensities"]):
                raise ValueError(f"Spectrum peak arrays differ for query {qid}")
            if qid not in qmeta:
                qmeta[qid] = {
                    "ionization_mode": str(row["ionization_mode"]),
                    "adduct": str(row["adduct"]),
                    "covered_glacier": False,
                }
            if row["ionization_mode"] == "positive" and row["adduct"] in ("[M+H]+", "[M+Na]+"):
                qmeta[qid]["covered_glacier"] = True
                spectra[qid].append({
                    "mz": row["ms2_mzs"],
                    "it": row["ms2_normalized_intensities"],
                    "prec": row["precursor_mz"],
                    "adduct": row["adduct"],
                    "mode": row["ionization_mode"],
                    "instrument": "timsTOF",
                    "ce_ev": row["collision_energy_ev"],
                })
    return spectra, qmeta


def load_cached_gl(partition: str) -> dict[str, dict[str, float | None]]:
    cached = {}
    probe_out = GL_PKG / f"{partition}_probe/gl_out.json"
    if probe_out.exists():
        cached.update(json.loads(probe_out.read_text()))
    inc_out = OUT_ROOT / f"{partition}_missing/gl_out.json"
    if inc_out.exists():
        for qid, smap in json.loads(inc_out.read_text()).items():
            cached.setdefault(qid, {}).update(smap)
    return cached


def prepare_missing(partition: str) -> Path:
    rows, _ = load_partition(partition)
    cached = load_cached_gl(partition)
    keys = {x[0] for r in rows for x in r["b_top25"]}
    meta = load_all_meta(keys)
    needed = {int(r["query_id"]) for r in rows}
    spectra, _ = load_query_metadata(needed)
    items = []
    for r in rows:
        qid = int(r["query_id"])
        order = [x[0] for x in r["b_top25"]]
        formulas = [meta[k]["formula"] if k in meta else f"__UNK_{k}" for k in order]
        count = Counter(formulas)
        q_cache = cached.get(str(qid), {})
        smis = list(dict.fromkeys(
            meta[k]["smiles"]
            for k, f in zip(order, formulas)
            if k in meta and count[f] >= 2 and meta[k]["smiles"] not in q_cache
        ))
        if smis and spectra[qid]:
            items.append({"mid": str(qid), "cands": smis, "spectra": spectra[qid]})
    out_dir = OUT_ROOT / f"{partition}_missing"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = out_dir / "gl_in.json"
    in_path.write_text(json.dumps(items, separators=(",", ":")) + "\n")
    manifest = {
        "partition": f"fold1 {partition} b_top25 incremental",
        "queries_needing_inference": len(items),
        "covered_spectra": sum(len(x["spectra"]) for x in items),
        "candidates": sum(len(x["cands"]) for x in items),
        "sha256": {"candidates": sha(cand_path(partition)), "gl_input": sha(in_path), "source": sha(Path(__file__))},
    }
    (out_dir / "prepare.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{partition} missing] prepared:", json.dumps({k: manifest[k] for k in ("queries_needing_inference", "covered_spectra", "candidates")}))
    return out_dir


def run_gl_missing(partition: str, budget_s: float = 1800.0) -> None:
    out_dir = OUT_ROOT / f"{partition}_missing"
    in_path = out_dir / "gl_in.json"
    if not in_path.exists():
        prepare_missing(partition)
    items = json.loads(in_path.read_text())
    if not items:
        (out_dir / "gl_out.json").write_text("{}\n")
        return
    run_gl(
        str(GL_PKG),
        items,
        workdir=str(out_dir),
        device="cuda",
        budget_s=budget_s,
        site=str(ICE_PKG / "ice_site"),
        wheels=str(ICE_PKG / "wheels"),
        python="/home/yuanyn/anaconda3/envs/mypoltype2/bin/python",
    )


def summarize_slice(mol_rows: list[dict]) -> dict:
    if not mol_rows:
        return {"molecules": 0}
    base = np.array([r["base_rr"] for r in mol_rows], dtype=np.float64)
    gl = np.array([r["gl_rr"] for r in mol_rows], dtype=np.float64)
    cov = np.array([r["truth_in_candidates"] for r in mol_rows], dtype=bool)
    delta = gl - base
    res = {
        "molecules": len(mol_rows),
        "candidate_coverage": float(cov.mean()),
        "base_mrr": float(base.mean()),
        "gl_mrr": float(gl.mean()),
        "delta": float(delta.mean()),
        "bootstrap95": bootstrap(delta),
        "improved": int((delta > 0).sum()),
        "worsened": int((delta < 0).sum()),
    }
    if cov.any():
        res["cond_base_mrr"] = float(base[cov].mean())
        res["cond_gl_mrr"] = float(gl[cov].mean())
        res["cond_delta"] = float(delta[cov].mean())
    return res


def audit_partition(partition: str) -> dict:
    rows, labels = load_partition(partition)
    scores = load_cached_gl(partition)
    keys = {x[0] for r in rows for x in r["b_top25"]}
    meta = load_all_meta(keys)
    needed = {int(r["query_id"]) for r in rows}
    _, qmeta = load_query_metadata(needed)

    q_records = []
    groups = defaultdict(list)
    for r in rows:
        qid = int(r["query_id"])
        lbl = labels[qid]
        original = [x[0] for x in r["b_top25"]]
        smis = [meta[k]["smiles"] if k in meta else f"__UNK_{k}" for k in original]
        formulas = [meta[k]["formula"] if k in meta else f"__UNK_{k}" for k in original]
        base = [float(x[1]) for x in r["b_top25"]]
        gl_map = scores.get(str(qid), {})
        order = rerank_multi(smis, original, base, formulas, [gl_map], [0.5], top_n=25)
        reranked = [original[i] for i in order]
        truth = lbl["truth_key"]
        rec = {
            "partition": partition,
            "query_id": qid,
            "identity_group": int(lbl["identity_group"]),
            "arm": str(lbl["arm"]),
            "ionization_mode": qmeta[qid]["ionization_mode"],
            "adduct": qmeta[qid]["adduct"],
            "covered_glacier": bool(qmeta[qid]["covered_glacier"]),
            "truth_in_candidates": truth in original,
            "base_rr": rr(original, truth),
            "gl_rr": rr(reranked, truth),
            "rank_changed": order != list(range(len(order))),
        }
        q_records.append(rec)
        groups[rec["identity_group"]].append(rec)

    mol_rows = []
    for gid, g in groups.items():
        mol_rows.append({
            "partition": partition,
            "identity_group": gid,
            "arm": g[0]["arm"],
            "ionization_mode": g[0]["ionization_mode"],
            "adduct": g[0]["adduct"],
            "covered_glacier": any(x["covered_glacier"] for x in g),
            "truth_in_candidates": any(x["truth_in_candidates"] for x in g),
            "base_rr": float(np.mean([x["base_rr"] for x in g])),
            "gl_rr": float(np.mean([x["gl_rr"] for x in g])),
        })

    slices = {"overall": summarize_slice(mol_rows)}
    for field in ("arm", "ionization_mode", "adduct", "covered_glacier"):
        vals = sorted({str(m[field]) for m in mol_rows})
        slices[field] = {v: summarize_slice([m for m in mol_rows if str(m[field]) == v]) for v in vals}

    return {"partition": partition, "queries": len(q_records), "molecules": len(mol_rows), "slices": slices, "mol_rows": mol_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare-missing", "run-gl-missing", "audit"])
    parser.add_argument("--partition", choices=["confirm", "final", "all"], default="all")
    parser.add_argument("--budget", type=float, default=1800.0)
    args = parser.parse_args()

    parts = ["confirm", "final"] if args.partition == "all" else [args.partition]
    if args.step == "prepare-missing":
        for p in parts:
            prepare_missing(p)
    elif args.step == "run-gl-missing":
        for p in parts:
            run_gl_missing(p, budget_s=args.budget)
    elif args.step == "audit":
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        all_mols = []
        report = {}
        for p in parts:
            res = audit_partition(p)
            all_mols.extend(res.pop("mol_rows"))
            report[p] = res
        (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        pq.write_table(pa.Table.from_pylist(all_mols), OUT_ROOT / "per_molecule.parquet")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()