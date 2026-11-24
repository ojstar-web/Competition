"""Dual-model (FPNet + GLACIER) consistency gate for cross-source same-formula reranking."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bond_fragment_probe import ROOT, bootstrap, rr, sha
from e2e_audit_baseline import (
    GL_PKG,
    OUT_ROOT as BASE_AUDIT_ROOT,
    cand_path,
    load_all_meta,
    load_cached_gl,
    load_partition,
    load_query_metadata,
    summarize_slice,
)

sys.path.insert(0, str(GL_PKG))
from gl_fuse import rerank_multi  # noqa: E402

OUT_DIR = ROOT / "outputs/dual_gate_rerank_20261124_01"
TAU_GL = 0.05
TAU_BASE = 10.0


def dual_gate_rerank(
    b_top25: list[list],
    meta: dict[str, dict],
    gl_map: dict[str, float | None],
    tau_gl: float = TAU_GL,
    tau_base: float = TAU_BASE,
    lam: float = 0.5,
) -> tuple[list[str], bool]:
    orig = [x[0] for x in b_top25]
    smis = [meta[k]["smiles"] if k in meta else f"__UNK_{k}" for k in orig]
    base = [float(x[1]) for x in b_top25]

    pool_gl = [gl_map.get(meta[x[0]]["smiles"]) for x in b_top25 if x[2] == "pool" and x[0] in meta]
    pc_gl = [gl_map.get(meta[x[0]]["smiles"]) for x in b_top25 if x[2] == "pubchem" and x[0] in meta]
    pool_gl_ok = [float(v) for v in pool_gl if v is not None]
    pc_gl_ok = [float(v) for v in pc_gl if v is not None]
    pool_b = [float(x[1]) for x in b_top25 if x[2] == "pool"]
    pc_b = [float(x[1]) for x in b_top25 if x[2] == "pubchem"]

    unlocked = (
        bool(pool_gl_ok and pc_gl_ok and pool_b and pc_b)
        and (max(pc_gl_ok) - max(pool_gl_ok) > tau_gl)
        and (max(pc_b) - max(pool_b) > tau_base)
    )
    if unlocked:
        forms = [meta[k]["formula"] if k in meta else f"__UNK_{k}" for k in orig]
    else:
        forms = [f"{x[2]}|{meta[k]['formula'] if k in meta else k}" for x, k in zip(b_top25, orig)]

    order = rerank_multi(smis, orig, base, forms, [gl_map], [lam], top_n=25)
    return [orig[i] for i in order], unlocked


def eval_partition(partition: str) -> dict:
    rows, labels = load_partition(partition)
    scores = load_cached_gl(partition)
    keys = {x[0] for r in rows for x in r["b_top25"]}
    meta = load_all_meta(keys)
    needed = {int(r["query_id"]) for r in rows}
    _, qmeta = load_query_metadata(needed)

    groups = defaultdict(list)
    unlock_count = 0
    for r in rows:
        qid = int(r["query_id"])
        lbl = labels[qid]
        orig = [x[0] for x in r["b_top25"]]
        gl_map = scores.get(str(qid), {})
        ranked, unlocked = dual_gate_rerank(r["b_top25"], meta, gl_map)
        if unlocked:
            unlock_count += 1
        truth = lbl["truth_key"]
        groups[int(lbl["identity_group"])].append({
            "arm": str(lbl["arm"]),
            "ionization_mode": qmeta[qid]["ionization_mode"],
            "covered_glacier": bool(qmeta[qid]["covered_glacier"]),
            "truth_in_candidates": truth in orig,
            "base_rr": rr(orig, truth),
            "gl_rr": rr(ranked, truth),
        })

    mol_rows = []
    for gid, g in groups.items():
        mol_rows.append({
            "identity_group": gid,
            "arm": g[0]["arm"],
            "ionization_mode": g[0]["ionization_mode"],
            "covered_glacier": any(x["covered_glacier"] for x in g),
            "truth_in_candidates": any(x["truth_in_candidates"] for x in g),
            "base_rr": float(np.mean([x["base_rr"] for x in g])),
            "gl_rr": float(np.mean([x["gl_rr"] for x in g])),
        })

    slices = {"overall": summarize_slice(mol_rows)}
    for field in ("arm", "ionization_mode", "covered_glacier"):
        vals = sorted({str(m[field]) for m in mol_rows})
        slices[field] = {v: summarize_slice([m for m in mol_rows if str(m[field]) == v]) for v in vals}

    return {
        "partition": partition,
        "molecules": len(mol_rows),
        "queries": len(rows),
        "unlocked_queries": unlock_count,
        "params": {"tau_gl": TAU_GL, "tau_base": TAU_BASE, "lambda": 0.5},
        "sha256": {"candidates": sha(cand_path(partition)), "source": sha(Path(__file__))},
        "slices": slices,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {p: eval_partition(p) for p in ("confirm", "final")}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for p, res in report.items():
        s = res["slices"]
        print(f"=== {p} (unlocked={res['unlocked_queries']}/{res['queries']}) ===")
        for label, d in [("overall", s["overall"]), ("control", s["arm"]["control"]), ("ood", s["arm"]["ood"])]:
            print(
                f"  {label:8s}: cov={d['candidate_coverage']:.4f} base={d['base_mrr']:.4f} "
                f"dual_gate={d['gl_mrr']:.4f} delta={d['delta']:+.4f} "
                f"CI={[round(x, 4) for x in d['bootstrap95']]} (+{d['improved']}/-{d['worsened']})"
            )


if __name__ == "__main__":
    main()