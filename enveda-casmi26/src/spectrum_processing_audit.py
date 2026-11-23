"""Measure information removed by the repository's MS/MS peak filters.

Run with the repository's existing pyarrow environment. This audit never
changes spectra or uses hidden test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "test": ROOT / "data/test.parquet",
    "fold1": ROOT / "outputs/fold1_ood_retrieval_20261110_01/query_features.parquet",
}
COLUMNS = [
    "ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
    "ionization_mode", "adduct", "collision_energy_ev",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_metrics(row: dict) -> dict:
    mz = np.asarray(row["ms2_mzs"], dtype=np.float64)
    intensity = np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
    if len(mz) != len(intensity):
        raise ValueError("m/z and intensity arrays differ in length")
    if len(mz) == 0 or not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity)):
        raise ValueError("empty or nonfinite spectrum")
    if np.any(intensity < 0) or np.max(intensity) <= 0:
        raise ValueError("negative or zero-only intensity")
    precursor = float(row["precursor_mz"])
    if not np.isfinite(precursor) or precursor <= 0:
        raise ValueError("invalid precursor m/z")

    positive = (mz > 0) & (intensity > 0)
    # clean_peaks(): precursor + 2 Da, 0.1% relative floor, top 512.
    library_window = positive & (mz <= precursor + 2)
    library_floor = library_window & (intensity >= .001 * np.max(intensity[library_window])) if np.any(library_window) else library_window
    library_idx = np.flatnonzero(library_floor)
    if len(library_idx) > 512:
        library_idx = library_idx[np.argpartition(intensity[library_idx], -512)[-512:]]
    # baseline.py: excludes the precursor neighbourhood, 0.5% floor, top 64.
    baseline_window = positive & (mz <= 2000) & (mz <= precursor + 2) & (np.abs(mz - precursor) > 1.5)
    baseline_floor = baseline_window & (intensity >= .005 * np.max(intensity[baseline_window])) if np.any(baseline_window) else baseline_window
    baseline_idx = np.flatnonzero(baseline_floor)
    if len(baseline_idx) > 64:
        baseline_idx = baseline_idx[np.argpartition(intensity[baseline_idx], -64)[-64:]]
    signal = float(np.sum(intensity[positive]))
    return {
        "raw_peaks": int(np.sum(positive)),
        "library_peaks": len(library_idx),
        "baseline_peaks": len(baseline_idx),
        "library_signal_fraction": float(np.sum(intensity[library_idx]) / signal),
        "baseline_signal_fraction": float(np.sum(intensity[baseline_idx]) / signal),
        "precursor_neighborhood_signal_fraction": float(np.sum(intensity[positive & (np.abs(mz - precursor) <= 1.5)]) / signal),
        "above_precursor_plus_2_signal_fraction": float(np.sum(intensity[positive & (mz > precursor + 2)]) / signal),
    }


def summarize(rows: list[dict]) -> dict:
    fields = ("raw_peaks", "library_peaks", "baseline_peaks", "library_signal_fraction",
              "baseline_signal_fraction", "precursor_neighborhood_signal_fraction",
              "above_precursor_plus_2_signal_fraction")
    output = {"spectra": len(rows)}
    for field in fields:
        values = np.asarray([row[field] for row in rows])
        output[field] = {"median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
                         "p90": float(np.quantile(values, .9))}
    output["library_fewer_than_5_peaks"] = sum(row["library_peaks"] < 5 for row in rows)
    output["baseline_fewer_than_5_peaks"] = sum(row["baseline_peaks"] < 5 for row in rows)
    output["library_loses_over_20pct_signal"] = sum(row["library_signal_fraction"] < .8 for row in rows)
    output["baseline_loses_over_20pct_signal"] = sum(row["baseline_signal_fraction"] < .8 for row in rows)
    return output


def audit(name: str, path: Path) -> dict:
    group_column = "molecule_id" if name == "test" else "query_id"
    table = pq.read_table(path, columns=COLUMNS + [group_column])
    metrics = []
    by_mode = defaultdict(list)
    by_adduct = defaultdict(list)
    group_ce = defaultdict(list)
    missing_ce = 0
    multi_ce_row = 0
    for row in table.to_pylist():
        result = peak_metrics(row)
        metrics.append(result)
        by_mode[row["ionization_mode"]].append(result)
        by_adduct[row["adduct"]].append(result)
        ce = row["collision_energy_ev"] or []
        if not ce:
            missing_ce += 1
        if len(ce) > 1:
            multi_ce_row += 1
        group_ce[row[group_column]].extend(float(value) for value in ce if value is not None)
    ce_ranges = [max(values) - min(values) for values in group_ce.values() if values]
    return {
        "input": str(path.relative_to(ROOT)), "input_sha256": sha256(path),
        "groups": len(group_ce), "overall": summarize(metrics),
        "by_mode": {key: summarize(value) for key, value in sorted(by_mode.items())},
        "by_adduct": {key: summarize(value) for key, value in sorted(by_adduct.items())},
        "collision_energy": {"missing_rows": missing_ce, "multiple_values_per_row": multi_ce_row,
                             "groups_with_ce_range_over_10ev": sum(value > 10 for value in ce_ranges),
                             "groups_with_any_ce": len(ce_ranges)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spectrum_processing_audit_20261119_01/report.json")
    args = parser.parse_args()
    report = {name: audit(name, path) for name, path in INPUTS.items()}
    report["script_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
