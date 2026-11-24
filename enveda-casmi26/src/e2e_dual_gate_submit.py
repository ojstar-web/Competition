"""End-to-end submission pipeline: fold-safe FPNet pool20_pc5 + GLACIER + dual-model consistency gate."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLD1 = ROOT / "outputs/fold1_ood_retrieval_20261110_01"
PREV = ROOT / "outputs/fold0_fp_prior_validation_20261110_01"
CODE = PREV / "inputs/casmi26-v3-models/code"
GL_PKG = ROOT / "outputs/orthogonal_glacier_20261119_01"
ICE_PKG = ROOT / "outputs/orthogonal_simulator_20261119_01"
VENV_PY = str(ROOT / ".venv/bin/python")
CUDA_PY = "/home/yuanyn/anaconda3/envs/mypoltype2/bin/python"
REF_477 = ROOT / "outputs/public477_exact_reproduction_20261105_01/cloud_output_v1/submission.csv"


def cmd_prep(workdir: Path, max_mols: int | None) -> None:
    import pyarrow.parquet as pq

    workdir.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data/sample_submission.csv") as f:
        sample_ids = [r["molecule_id"] for r in csv.DictReader(f)]
    selected_ids = sample_ids[:max_mols] if max_mols else sample_ids
    selected_set = set(selected_ids)

    pf = pq.ParquetFile(ROOT / "data/test.parquet")
    cols = [
        "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
        "adduct", "ionization_mode", "precursor_mz", "collision_energy_ev",
    ]
    by_mol = defaultdict(list)
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        for row in batch.to_pylist():
            mid = row["molecule_id"]
            if mid not in selected_set:
                continue
            ce = row["collision_energy_ev"]
            by_mol[mid].append({
                "mz": [float(x) for x in row["ms2_mzs"]],
                "it": [float(x) for x in row["ms2_normalized_intensities"]],
                "mode": 1 if row["ionization_mode"] == "positive" else -1,
                "ionization_mode": str(row["ionization_mode"]),
                "adduct": str(row["adduct"]),
                "prec": float(row["precursor_mz"]),
                "ce": float(np.mean(ce)) if ce is not None and len(ce) else None,
                "ce_n": len(ce) if ce is not None else 0,
                "ce_ev": [float(x) for x in ce] if ce is not None else None,
            })

    queries = [{"molecule_id": mid, "spectra": by_mol[mid]} for mid in selected_ids]
    (workdir / "test_queries.json").write_text(json.dumps(queries))

    pool_pkl = workdir / "pool_meta_slim.pkl"
    if not pool_pkl.exists():
        t = pq.read_table(FOLD1 / "reduced_pool/pool_meta.parquet", columns=["key", "mass", "smiles", "formula", "pid"])
        with pool_pkl.open("wb") as fh:
            pickle.dump(t.to_pylist(), fh)
    print(f"[prep] prepared {len(queries)} molecules in {workdir}")


def cmd_generate(workdir: Path) -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    sys.path[:0] = [str(FOLD1 / "source"), str(CODE)]
    from casmi import chem, fpnet  # noqa: E402
    import probe_core2 as pc  # noqa: E402

    queries = json.loads((workdir / "test_queries.json").read_text())
    with (workdir / "pool_meta_slim.pkl").open("rb") as fh:
        pool_rows = pickle.load(fh)

    keys = [r["key"] for r in pool_rows]
    mass = np.array([r["mass"] for r in pool_rows], dtype=np.float64)
    meta = {r["key"]: {"smiles": r["smiles"], "formula": r["formula"]} for r in pool_rows}

    packed = np.load(FOLD1 / "reduced_pool/pool_fp.npy", mmap_mode="r")
    bits = np.load(FOLD1 / "reduced_pool/fp_bits.npy")
    bank = fpnet.ModelBank([str(FOLD1 / "inputs/fold_safe/fpnet_fold1.pt")], device="cuda")

    # Configure probe_core2 worker state without pyarrow dependency
    r = chem._rdkit()
    pc._W["chem"] = chem
    pc._W["bits"] = bits
    pc._W["sel_e"] = np.where(bits < 4096)[0]
    pc._W["raw_e"] = bits[pc._W["sel_e"]]
    pc._W["pool_keys"] = set(keys)
    pc._W["ecfp4"] = r["gen"].GetMorganGenerator(radius=2, fpSize=4096)
    pc._W["Chem"] = r["Chem"]
    stores = []
    pc_base = FOLD1 / "inputs/pubchem"
    for stem in ("", "delta", "growth"):
        prefix = f"{stem}_" if stem else ""
        stores.append(tuple(np.load(pc_base / f"{prefix}pc_{p}.npy", mmap_mode="r") for p in ("mass", "off", "smiles")))

    def windows(target: float):
        lo_m, hi_m = target * (1 - 10.0 * 1e-6), target * (1 + 10.0 * 1e-6)
        return [(int(np.searchsorted(m, lo_m, "left")), int(np.searchsorted(m, hi_m, "right"))) for m, _, _ in stores]

    def decode_windows(ranges):
        out = []
        for (_, offsets, buf), (a, b) in zip(stores, ranges):
            if b <= a:
                continue
            raw = bytes(buf[int(offsets[a]):int(offsets[b])])
            base = int(offsets[a])
            out.extend(raw[int(offsets[i] - base):int(offsets[i + 1] - base)].decode("ascii") for i in range(a, b))
        return out

    pc._window = lambda target: decode_windows(windows(target))

    cand_rows = []
    gl_items = []
    t0 = time.monotonic()
    for idx, q in enumerate(queries, 1):
        mid = q["molecule_id"]
        spectra = [
            {
                "mz": np.asarray(s["mz"], dtype=np.float32),
                "it": np.asarray(s["it"], dtype=np.float32),
                "mode": s["mode"],
                "adduct": s["adduct"],
                "prec": s["prec"],
                "ce": s["ce"],
                "ce_n": s["ce_n"],
            }
            for s in q["spectra"]
        ]
        neutral = chem.neutral_mass([s["prec"] for s in spectra], [s["adduct"] for s in spectra])
        neutral = neutral[np.isfinite(neutral)]
        target = float(np.median(neutral))
        z = pc.molecule_logits(bank, spectra)

        tol = 10.0
        a = int(np.searchsorted(mass, target * (1 - tol * 1e-6), "left"))
        b = int(np.searchsorted(mass, target * (1 + tol * 1e-6), "right"))
        if b == a:
            tol = 30.0
            a = int(np.searchsorted(mass, target * (1 - tol * 1e-6), "left"))
            b = int(np.searchsorted(mass, target * (1 + tol * 1e-6), "right"))
        pool_scores = np.unpackbits(packed[a:b], axis=1)[:, :len(bits)].astype(np.float32) @ z
        pool_cands = sorted([(keys[i], float(pool_scores[i - a]), "pool") for i in range(a, b)], key=lambda x: (-x[1], x[0]))
        arm_a = pool_cands[:25]

        # PubChem shortlist
        _, _, pc_scores, pc_keys, _ = pc.probe_one((idx, target, z))
        pubchem = [(str(k), float(v), "pubchem") for k, v in zip(pc_keys, pc_scores)]
        b_top25 = arm_a[:20] + pubchem[:5]
        if len(b_top25) < 25:
            b_top25 += arm_a[20:20 + (25 - len(b_top25))]

        # Resolve any missing pubchem SMILES/formula directly from windows
        missing_pc = {x[0] for x in b_top25 if x[0] not in meta}
        if missing_pc:
            for smi in pc._window(target):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                ik = Chem.MolToInchiKey(mol)[:14]
                if ik in missing_pc and ik not in meta:
                    meta[ik] = {
                        "smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
                        "formula": rdMolDescriptors.CalcMolFormula(mol),
                    }
                    missing_pc.remove(ik)
                    if not missing_pc:
                        break

        cand_rows.append({"molecule_id": mid, "b_top25": b_top25})
        order = [x[0] for x in b_top25]
        formulas = [meta[k]["formula"] if k in meta else f"__UNK_{k}" for k in order]
        count = Counter(formulas)
        gl_smis = list(dict.fromkeys(meta[k]["smiles"] for k, f in zip(order, formulas) if k in meta and count[f] >= 2))
        gl_specs = [
            {
                "mz": s["mz"], "it": s["it"], "prec": s["prec"], "adduct": s["adduct"],
                "mode": s["ionization_mode"], "instrument": "timsTOF", "ce_ev": s["ce_ev"],
            }
            for s in q["spectra"]
            if s["ionization_mode"] == "positive" and s["adduct"] in ("[M+H]+", "[M+Na]+")
        ]
        if gl_smis and gl_specs:
            gl_items.append({"mid": mid, "cands": gl_smis, "spectra": gl_specs})
        if idx % 10 == 0 or idx == len(queries):
            print(f"[generate] {idx}/{len(queries)} elapsed={time.monotonic()-t0:.1f}s", flush=True)

    (workdir / "candidates.json").write_text(json.dumps(cand_rows))
    (workdir / "cand_meta.json").write_text(json.dumps({k: meta[k] for r in cand_rows for x in r["b_top25"] for k in [x[0]] if k in meta}))
    (workdir / "gl_in.json").write_text(json.dumps(gl_items, separators=(",", ":")))
    print(f"[generate] done: {len(cand_rows)} mols, {len(gl_items)} GLACIER items")


def cmd_run_gl(workdir: Path, budget_s: float) -> None:
    sys.path.insert(0, str(GL_PKG))
    from gl_fuse import run_gl  # noqa: E402

    items = json.loads((workdir / "gl_in.json").read_text())
    if not items:
        (workdir / "gl_out.json").write_text("{}\n")
        return
    run_gl(
        str(GL_PKG),
        items,
        workdir=str(workdir),
        device="cuda",
        budget_s=budget_s,
        site=str(ICE_PKG / "ice_site"),
        wheels=str(ICE_PKG / "wheels"),
        python=CUDA_PY,
    )


def cmd_submit(workdir: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from dual_gate_rerank import dual_gate_rerank  # noqa: E402

    cand_rows = json.loads((workdir / "candidates.json").read_text())
    meta = json.loads((workdir / "cand_meta.json").read_text())
    gl_scores = json.loads((workdir / "gl_out.json").read_text()) if (workdir / "gl_out.json").exists() else {}

    with open(ROOT / "data/sample_submission.csv") as f:
        sample_ids = [r["molecule_id"] for r in csv.DictReader(f)]

    ref_map = {}
    if REF_477.exists():
        with open(REF_477) as f:
            ref_map = {r["molecule_id"]: r["smiles"].split(";") for r in csv.DictReader(f)}

    sub_rows = []
    unlocked_mols = []
    top1_agree_477 = 0
    cand_map = {r["molecule_id"]: r["b_top25"] for r in cand_rows}

    for mid in sample_ids:
        if mid in cand_map:
            b_top25 = cand_map[mid]
            ranked_keys, unlocked = dual_gate_rerank(b_top25, meta, gl_scores.get(mid, {}))
            if unlocked:
                unlocked_mols.append(mid)
            smis = [meta[k]["smiles"] if k in meta else "CCO" for k in ranked_keys]
            while len(smis) < 25:
                smis.append("CCO")
            smis = smis[:25]
            if mid in ref_map and ref_map[mid][0] == smis[0]:
                top1_agree_477 += 1
        else:
            smis = ["CCO"] * 25
        sub_rows.append({"molecule_id": mid, "smiles": ";".join(smis)})

    out_csv = workdir / "submission.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["molecule_id", "smiles"])
        w.writeheader()
        w.writerows(sub_rows)

    summary = {
        "total_submission_rows": len(sub_rows),
        "processed_molecules": len(cand_rows),
        "unlocked_molecules": len(unlocked_mols),
        "unlocked_ids": unlocked_mols,
        "top1_agree_with_0477": top1_agree_477,
        "schema_valid": len(sub_rows) == len(sample_ids) and all(len(r["smiles"].split(";")) == 25 for r in sub_rows),
    }
    (workdir / "submit_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "generate", "run-gl", "submit", "all"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "outputs/e2e_submit_smoke_20261124_01")
    ap.add_argument("--max-mols", type=int, default=16)
    ap.add_argument("--budget", type=float, default=1200.0)
    args = ap.parse_args()

    if args.step == "prep":
        cmd_prep(args.workdir, args.max_mols)
    elif args.step == "generate":
        cmd_generate(args.workdir)
    elif args.step == "run-gl":
        cmd_run_gl(args.workdir, args.budget)
    elif args.step == "submit":
        cmd_submit(args.workdir)
    elif args.step == "all":
        subprocess.run([VENV_PY, __file__, "prep", "--workdir", str(args.workdir), "--max-mols", str(args.max_mols)], check=True)
        subprocess.run([CUDA_PY, __file__, "generate", "--workdir", str(args.workdir)], check=True)
        subprocess.run([VENV_PY, __file__, "run-gl", "--workdir", str(args.workdir), "--budget", str(args.budget)], check=True)
        subprocess.run([VENV_PY, __file__, "submit", "--workdir", str(args.workdir)], check=True)


if __name__ == "__main__":
    main()