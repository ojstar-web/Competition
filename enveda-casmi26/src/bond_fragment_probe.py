"""Probe a one-bond structural fragment signal on frozen fold1 pool candidates."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
FOLD = ROOT / "outputs/fold1_ood_retrieval_20261110_01"
CAMPAIGN = ROOT / "outputs/local_strategy_campaign_20261119_01"
OUT = ROOT / "outputs/bond_fragment_20261119_02"
H = 1.00782503223
PROTON = 1.007276466621
POLICIES = {"baseline": 0.0, "bond_01": 0.1, "bond_025": 0.25, "bond_05": 0.5}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_bond_masses(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid frozen candidate SMILES: {smiles}")
    periodic = Chem.GetPeriodicTable()
    atom_masses = np.array([a.GetMass() if a.GetIsotope() else periodic.GetMostCommonIsotopeMass(a.GetAtomicNum())
                            for a in mol.GetAtoms()])
    atom_masses += H * np.array([a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms()])
    masses = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        cut = Chem.RWMol(mol)
        cut.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        parts = Chem.GetMolFrags(cut.GetMol())
        if len(parts) != 2:
            continue
        masses.extend(sum(atom_masses[list(part)]) for part in parts)
    return np.unique(np.asarray(masses, dtype=np.float64))


def query_rows(needed: set[int]) -> dict[int, list[tuple[np.ndarray, np.ndarray, int]]]:
    result = defaultdict(list)
    pf = pq.ParquetFile(FOLD / "query_features.parquet")
    for batch in pf.iter_batches(batch_size=512, columns=["query_id", "ms2_mzs", "ms2_normalized_intensities", "ionization_mode", "precursor_mz"]):
        for row in batch.to_pylist():
            qid = int(row["query_id"])
            if qid not in needed:
                continue
            mz = np.asarray(row["ms2_mzs"], dtype=float)
            inten = np.asarray(row["ms2_normalized_intensities"], dtype=float)
            precursor = float(row["precursor_mz"])
            valid = (mz > 30) & (mz < precursor - 1.5) & (inten >= 0.005)
            if valid.any():
                order = np.argsort(inten[valid])[-64:]
                mode = 1 if row["ionization_mode"] == "positive" else -1
                result[qid].append((mz[valid][order], inten[valid][order], mode))
    return result


def fragment_score(masses: np.ndarray, rows: list[tuple[np.ndarray, np.ndarray, int]]) -> float:
    if not len(masses) or not rows:
        return 0.0
    scores = []
    for mz, intensity, mode in rows:
        # A cut fragment can retain or transfer H; the four offsets bracket that
        # local uncertainty without assuming a full fragmentation tree.
        shifts = np.array([-H, 0.0, H, 2 * H]) + (PROTON if mode == 1 else -PROTON)
        expected = (masses[:, None] + shifts).ravel()
        nearest = np.searchsorted(expected := np.sort(expected), mz)
        left = expected[np.clip(nearest - 1, 0, len(expected) - 1)]
        right = expected[np.clip(nearest, 0, len(expected) - 1)]
        error = np.minimum(abs(mz - left), abs(mz - right))
        hit = error <= np.maximum(0.01, mz * 20e-6)
        scores.append(float(np.sqrt(intensity[hit]).sum() / np.sqrt(intensity).sum()))
    return float(np.mean(scores))


def rr(keys: list[str], truth: str) -> float:
    return 1.0 / (keys.index(truth) + 1) if truth in keys else 0.0


def bootstrap(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    return np.quantile(diff[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    start = monotonic()
    labels = {int(r["query_id"]): r for r in pq.read_table(FOLD / "scoring_labels_corrected.parquet").to_pylist()}
    entries = []
    for split, path in (("pilot", FOLD / "pilot_candidates.json"),
                        ("confirm", CAMPAIGN / "confirm_pool20_pc5/candidates.jsonl")):
        records = json.loads(path.read_text()) if split == "pilot" else [json.loads(s) for s in path.read_text().splitlines()]
        entries.extend((split, r) for r in records if labels[int(r["query_id"])]["arm"] == "control")
    needed = {int(r["query_id"]) for _, r in entries}
    spectra = query_rows(needed)
    missing = needed - spectra.keys()
    if missing:
        raise ValueError(f"No usable spectra for {len(missing)} evaluation query IDs")
    candidate_keys = {item[0] for _, r in entries for item in r["a_top25"]}
    pool = pq.read_table(FOLD / "reduced_pool/pool_meta.parquet", columns=["key", "smiles"])
    smiles = {r["key"]: r["smiles"] for r in pool.to_pylist() if r["key"] in candidate_keys}
    if candidate_keys - smiles.keys():
        raise ValueError("Frozen pool candidates lack structure mappings")
    fragments = {key: one_bond_masses(smiles[key]) for key in sorted(candidate_keys)}
    rows_out = []
    for split, entry in entries:
        qid = int(entry["query_id"])
        truth = labels[qid]["truth_key"]
        original = [x[0] for x in entry["a_top25"]]
        frag = np.asarray([fragment_score(fragments[key], spectra[qid]) for key in original])
        # Rank fusion avoids comparing FPNet logits with bounded peak fractions.
        base = 1.0 / (np.arange(len(original)) + 1)
        frag_rank = np.empty(len(original), dtype=int)
        frag_rank[np.argsort(-frag, kind="stable")] = np.arange(len(original))
        fscore = 1.0 / (frag_rank + 1)
        ranks = {policy: [original[i] for i in np.argsort(-(base + weight * fscore), kind="stable")]
                 for policy, weight in POLICIES.items()}
        if ranks["baseline"] != original:
            raise AssertionError("Baseline did not replay frozen candidate order")
        rows_out.append({"split": split, "query_id": qid,
                         "identity_group": int(labels[qid]["identity_group"]),
                         "truth_in_candidates": truth in original,
                         "rr": {policy: rr(ranked, truth) for policy, ranked in ranks.items()},
                         "truth_fragment_score": float(frag[original.index(truth)]) if truth in original else None})
    report = {"experiment": "one-bond fragment rank fusion on frozen fold1 pool-only control candidates",
              "policies": POLICIES, "rows": {}, "input_sha256": {
                  "source": sha(Path(__file__)),
                  "pool_meta": sha(FOLD / "reduced_pool/pool_meta.parquet"),
                  "query_features": sha(FOLD / "query_features.parquet"),
                  "labels": sha(FOLD / "scoring_labels_corrected.parquet"),
                  "pilot_candidates": sha(FOLD / "pilot_candidates.json"),
                  "confirm_candidates": sha(CAMPAIGN / "confirm_pool20_pc5/candidates.jsonl")},
              "unique_candidate_structures": len(fragments), "elapsed_seconds": monotonic() - start,
              "limitation": "Control structures only; this does not test PubChem OOD rescue or model-pretraining overlap."}
    for split in ("pilot", "confirm"):
        subset = [r for r in rows_out if r["split"] == split]
        groups = defaultdict(list)
        for r in subset:
            groups[r["identity_group"]].append(r)
        values = {policy: np.array([np.mean([r["rr"][policy] for r in group]) for group in groups.values()])
                  for policy in POLICIES}
        report["rows"][split] = {"molecules": len(groups), "queries": len(subset),
                                 "truth_in_candidates": float(np.mean([r["truth_in_candidates"] for r in subset])),
                                 "mrr": {p: float(x.mean()) for p, x in values.items()},
                                 "delta_vs_baseline": {p: {"mean": float((x - values["baseline"]).mean()),
                                                           "bootstrap95": bootstrap(x - values["baseline"])}
                                                       for p, x in values.items() if p != "baseline"}}
    OUT.mkdir(exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "rows.json").write_text(json.dumps(rows_out, indent=2) + "\n")
    print(json.dumps(report["rows"]))


if __name__ == "__main__":
    main()
