"""Evaluate expanded PubChem rescue depths (pool15_pc10, pool12_pc13) with dual-model gate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bond_fragment_probe import FOLD, ROOT, bootstrap, rr, sha
from dual_gate_rerank import dual_gate_rerank
from e2e_audit_baseline import (
    GL_PKG,
    ICE_PKG,
    cand_path,
    load_cached_gl,
    load_partition,
    load_query_metadata,
    summarize_slice,
)

sys.path.insert(0, str(GL_PKG))
from gl_fuse import run_gl  # noqa: E402

OUT_DIR = ROOT / "outputs/pool12_pc13_20261125_01"
PC_META_PATH = OUT_DIR / "pubchem_meta.json"
CUDA_PY = "/home/yuanyn/anaconda3/envs/mypoltype2/bin/python"


def make_candidates(row: dict, pool_k: int, pc_k: int) -> list[list]:
    arm_a = row["a_top25"]
    pc = row["pubchem_shortlist"]
    res = arm_a[:pool_k] + pc[:pc_k]
    if len(res) < 25:
        res += arm_a[pool_k : pool_k + (25 - len(res))]
    return res[:25]


def load_meta_with_pc13(keys: set[str]) -> dict[str, dict]:
    table = pq.read_table(FOLD / "reduced_pool/pool_meta.parquet", columns=["key", "smiles", "formula"])
    meta = {r["key"]: {"smiles": r["smiles"], "formula": r["formula"]} for r in table.to_pylist() if r["key"] in keys}
    pc = json.loads(PC_META_PATH.read_text())
    for k in keys - meta.keys():
        if k in pc:
            meta[k] = pc[k]
    return meta


def load_all_gl(partition: str) -> dict[str, dict[str, float | None]]:
    cached = load_cached_gl(partition)
    extra = OUT_DIR / f"{partition}_gl/gl_out.json"
    if extra.exists():
        for qid, smap in json.loads(extra.read_text()).items():
            cached.setdefault(qid, {}).update(smap)
    return cached


def prepare_gl(partition: str) -> None:
    rows, _ = load_partition(partition)
    cached = load_all_gl(partition)
    cands_by_q = {}
    keys = set()
    for r in rows:
        # Union of pool15_pc10 and pool12_pc13 candidates covers both
        c15 = make_candidates(r, 15, 10)
        c12 = make_candidates(r, 12, 13)
        cands_by_q[int(r["query_id"])] = (c15, c12)
        for x in c15 + c12:
            keys.add(x[0])
    meta = load_meta_with_pc13(keys)
    spectra, _ = load_query_metadata({int(r["query_id"]) for r in rows})
    items = []
    for r in rows:
        qid = int(r["query_id"])
        if not spectra[qid]:
            continue
        q_cache = cached.get(str(qid), {})
        needed_smis = []
        for cand_list in cands_by_q[qid]:
            order = [x[0] for x in cand_list]
            formulas = [meta[k]["formula"] if k in meta else f"__UNK_{k}" for k in order]
            cnt = Counter(formulas)
            for k, f in zip(order, formulas):
                if k in meta and cnt[f] >= 2 and meta[k]["smiles"] not in q_cache:
                    needed_smis.append(meta[k]["smiles"])
        smis = list(dict.fromkeys(needed_smis))
        if smis:
            items.append({"mid": str(qid), "cands": smis, "spectra": spectra[qid]})
    out_dir = OUT_DIR / f"{partition}_gl"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gl_in.json").write_text(json.dumps(items, separators=(",", ":")) + "\n")
    print(f"[{partition}] incremental GLACIER items: {len(items)}, cands: {sum(len(x['cands']) for x in items)}, spectra: {sum(len(x['spectra']) for x in items)}")


def run_gl_part(partition: str, budget_s: float = 1800.0) -> None:
    out_dir = OUT_DIR / f"{partition}_gl"
    items = json.loads((out_dir / "gl_in.json").read_text())
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
        python=CUDA_PY,
    )


def eval_config(partition: str, pool_k: int, pc_k: int, tau_gl: float = 0.05, tau_base: float = 10.0) -> dict:
    rows, labels = load_partition(partition)
    scores = load_all_gl(partition)
    cands_map = {int(r["query_id"]): make_candidates(r, pool_k, pc_k) for r in rows}
    keys = {x[0] for c in cands_map.values() for x in c}
    meta = load_meta_with_pc13(keys)

    groups = defaultdict(list)
    for r in rows:
        qid = int(r["query_id"])
        lbl = labels[qid]
        cand = cands_map[qid]
        orig = [x[0] for x in cand]
        base_orig = [x[0] for x in r["b_top25"]]
        ranked, _ = dual_gate_rerank(cand, meta, scores.get(str(qid), {}), tau_gl=tau_gl, tau_base=tau_base)
        truth = lbl["truth_key"]
        groups[int(lbl["identity_group"])].append({
            "arm": str(lbl["arm"]),
            "truth_in_candidates": truth in orig,
            "base_rr": rr(base_orig, truth),
            "gl_rr": rr(ranked, truth),
        })

    mol_rows = []
    for gid, g in groups.items():
        mol_rows.append({
            "identity_group": gid,
            "arm": g[0]["arm"],
            "truth_in_candidates": any(x["truth_in_candidates"] for x in g),
            "base_rr": float(np.mean([x["base_rr"] for x in g])),
            "gl_rr": float(np.mean([x["gl_rr"] for x in g])),
        })
    return {
        "config": f"pool{pool_k}_pc{pc_k}_tgl{tau_gl}_tb{tau_base}",
        "overall": summarize_slice(mol_rows),
        "control": summarize_slice([m for m in mol_rows if m["arm"] == "control"]),
        "ood": summarize_slice([m for m in mol_rows if m["arm"] == "ood"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "run-gl", "eval-confirm", "eval-final"])
    ap.add_argument("--pool-k", type=int, default=12)
    ap.add_argument("--pc-k", type=int, default=13)
    ap.add_argument("--tau-gl", type=float, default=0.05)
    ap.add_argument("--tau-base", type=float, default=10.0)
    args = ap.parse_args()

    if args.step == "prepare":
        for p in ("confirm", "final"):
            prepare_gl(p)
    elif args.step == "run-gl":
        for p in ("confirm", "final"):
            run_gl_part(p)
    elif args.step == "eval-confirm":
        res = []
        for pk, pck in [(20, 5), (15, 10), (12, 13)]:
            for tgl in (0.05, 0.10):
                r = eval_config("confirm", pk, pck, tau_gl=tgl, tau_base=10.0)
                res.append(r)
                o, c, d = r["overall"], r["control"], r["ood"]
                print(
                    f"{r['config']:28s} | ALL={o['gl_mrr']:.4f} (cov={o['candidate_coverage']:.4f}) "
                    f"| CTRL={c['gl_mrr']:.4f} (cov={c['candidate_coverage']:.4f}) "
                    f"| OOD={d['gl_mrr']:.4f} (cov={d['candidate_coverage']:.4f})"
                )
        (OUT_DIR / "confirm_grid.json").write_text(json.dumps(res, indent=2) + "\n")
    elif args.step == "eval-final":
        r = eval_config("final", args.pool_k, args.pc_k, tau_gl=args.tau_gl, tau_base=args.tau_base)
        (OUT_DIR / "final_report.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()