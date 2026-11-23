"""Compare query precursor-exclusion windows on frozen fresh S1 structures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

import baseline as b
import phase1_eval as p


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/phase1_20261023_01/split_manifest.json"
CONTROL = ROOT / "outputs/peak_filter_ab_20261119_fresh_control_01/predictions.json"
WINDOWS = (0.0, 0.5, 1.5, 2.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_vector(row: dict, exclusion_da: float) -> dict[int, float]:
    mzs = np.asarray(row["ms2_mzs"], dtype=float)
    intensities = np.asarray(row["ms2_normalized_intensities"], dtype=float)
    precursor = float(row["precursor_mz"])
    if mzs.shape != intensities.shape or mzs.ndim != 1:
        raise ValueError("Peak arrays must be aligned one-dimensional arrays")
    valid = (np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0) & (mzs <= 2000)
             & (mzs <= precursor + 2) & (intensities > 0)
             & (np.abs(mzs - precursor) > exclusion_da))
    mzs, intensities = mzs[valid], intensities[valid]
    if not len(mzs):
        return {}
    keep = intensities >= .005 * intensities.max()
    mzs, intensities = mzs[keep], intensities[keep]
    order = np.argsort(-intensities, kind="stable")[:64]
    mzs, weights = mzs[order], np.sqrt(intensities[order])
    result = {}
    for positions, offset, channel_weight in ((mzs, 0, .7), (precursor - mzs, b.N_BINS, .3)):
        channel = defaultdict(float)
        for position, weight in zip(positions, weights):
            if 0 < position <= 2000:
                channel[int(np.floor(position / b.BIN_WIDTH + .5))] += float(weight)
        norm = np.sqrt(sum(value * value for value in channel.values()))
        if norm:
            result.update({index + offset: np.sqrt(channel_weight) * value / norm
                           for index, value in channel.items()})
    norm = np.sqrt(sum(value * value for value in result.values()))
    return {index: value / norm for index, value in result.items()} if norm else {}


def main(partition: str, output: Path) -> None:
    if sha(MANIFEST) != (MANIFEST.parent / "split_manifest.sha256").read_text().split()[0]:
        raise ValueError("Frozen split manifest SHA mismatch")
    if b.RDKIT_VERSION != "2026.03.3":
        raise RuntimeError("Official RDKit 2026.03.3 required")
    manifest = json.loads(MANIFEST.read_text())
    primary = manifest["primary_fresh_dev_keys"][:64]
    keys = (primary if partition == "development" else
            [key for key in manifest["partitions"]["fresh_dev"] if key not in set(primary)])
    if set(keys) & (set(manifest["partitions"]["legacy_dev"]) | set(manifest["partitions"]["sealed"])):
        raise ValueError("Fresh split overlaps exposed or sealed structures")
    groups, truth, spectrum_count = p.load_fresh_queries(ROOT / "data", keys)
    query_masses = [b.neutral_mass(row["precursor_mz"], row["adduct"])
                    for rows in groups.values() for row in rows]
    metadata, matrix, _, signatures, reference_audit = p.build_reference(
        ROOT / "data", b.read_metadata(ROOT / "data"), query_masses)
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([index for index, signature in enumerate(signatures)
                          if signature not in query_signatures], dtype=int)
    reference_masses = np.asarray([metadata[index][2] for index in allowed])
    stored = json.loads(CONTROL.read_text())["A_binned"] if partition == "development" else None
    names = {window: f"exclusion_{window:g}" for window in WINDOWS}
    predictions = {name: {} for name in names.values()}
    group_scores = {}
    changed_vectors = {names[window]: 0 for window in WINDOWS if window != 1.5}
    for query_id, rows in sorted(groups.items()):
        masses = np.asarray([b.neutral_mass(row["precursor_mz"], row["adduct"]) for row in rows])
        mass = float(np.median(masses))
        tolerance = max(b.CONFIG["mass_floor_da"], mass * b.CONFIG["mass_ppm"] * 1e-6)
        eligible = allowed[np.abs(reference_masses - mass) <= tolerance]
        reference_keys = [metadata[index][0] for index in eligible]
        smiles = {metadata[index][0]: metadata[index][1] for index in eligible}
        vectors = {window: [] for window in WINDOWS}
        for row in rows:
            baseline = query_vector(row, 1.5)
            if baseline != b.peak_vector(row["ms2_mzs"], row["ms2_normalized_intensities"],
                                         row["precursor_mz"]):
                raise AssertionError(f"Default vector replay differs for {query_id}")
            for window in WINDOWS:
                value = baseline if window == 1.5 else query_vector(row, window)
                vectors[window].append(value)
                if window != 1.5:
                    changed_vectors[names[window]] += value != baseline
        group_scores[query_id] = {}
        for window, name in names.items():
            scores = (b.sparse_vectors(vectors[window]) @ matrix[eligible].T).toarray()
            for row_index, row in enumerate(rows):
                scores[row_index, [metadata[index][3] != row["ionization_mode"] for index in eligible]] = 0.
            ranked = b.rank_candidates(scores, reference_keys, set())[:25]
            predictions[name][query_id] = [smiles[key] for key, _ in ranked]
            group_scores[query_id][name] = b.reciprocal_rank(truth[query_id], predictions[name][query_id])
        if stored is not None and predictions[names[1.5]][query_id] != stored[query_id]:
            raise AssertionError(f"Baseline top25 replay differs for {query_id}")
    metrics = {name: b.evaluate(truth, pred) for name, pred in predictions.items()}
    baseline_name = names[1.5]
    paired = {}
    for window, name in names.items():
        if window == 1.5:
            continue
        diff = np.asarray([group_scores[query_id][name] - group_scores[query_id][baseline_name]
                           for query_id in sorted(groups)])
        draws = np.random.default_rng(20261119).integers(0, len(diff), size=(10000, len(diff)))
        paired[name] = {"mrr_delta": float(diff.mean()),
                        "bootstrap95": np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                        "improved": int((diff > 0).sum()), "worsened": int((diff < 0).sum()),
                        "changed_top25_lists": sum(predictions[name][query_id] != predictions[baseline_name][query_id]
                                                   for query_id in groups),
                        "changed_query_vectors": changed_vectors[name]}
    report = {"experiment": "S1 fixed-reference query precursor-exclusion window",
              "partition": partition, "molecules": len(keys), "query_spectra": spectrum_count,
              "reference_spectra": len(allowed), "reference_audit": reference_audit,
              "windows_da": list(WINDOWS), "metrics": metrics, "paired_vs_1p5_da": paired,
              "baseline_replay": f"{len(keys)}/{len(keys)} exact top25 lists" if stored is not None else None,
              "input_sha256": {"manifest": sha(MANIFEST), "train_parquet": sha(ROOT / "data/train.parquet"),
                               "control_predictions": sha(CONTROL) if stored is not None else None,
                               "source": sha(Path(__file__)), "baseline_source": sha(Path(b.__file__)),
                               "phase1_eval_source": sha(Path(p.__file__))},
              "limitation": "S1 reference structures are available; this does not test unseen structures."}
    output.mkdir(exist_ok=False)
    b.write_json(output / "report.json", report)
    b.write_json(output / "group_scores.json", group_scores)
    print(json.dumps({"metrics": metrics, "paired": paired}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("development", "independent"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.partition, args.output)
