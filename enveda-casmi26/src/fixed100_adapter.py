"""Adapt pinned fixed100 spectra to the official V17 test schema, keeping truth local."""

import argparse
import bisect
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rank_aggregation_fixed100_adapter_20261030_01"
SELECTED = ROOT / "outputs/own_ranker_isolation_20261028_01/selected_structures.json"
SPECTRA = ROOT / "outputs/rank_aggregation_fixed100_peaks_20261030_01/fixed100_spectra.json"
TRAIN = ROOT / "data/train.parquet"
OFFICIAL_TEST = ROOT / "data/test.parquet"
EXPECTED_SELECTED = "367d17d16a1afd45ad5950bea6a33f8d39b54e3d52eea05b67d81a397dc42278"
EXPECTED_SPECTRA = "e5a231be263293b1f14ac44643cc2817451e790c4a42e0ee3ec453fa473b81e7"
EXPECTED_TRAIN_BYTES = 3033286496
EXPECTED_TRAIN_ROWS = 2539608
EXPECTED_FOOTER_SHA = "fce7ae8a9b6403d579218b36663f892853d5a7b28483c95b444923e84faa2acb"
FIELDS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
          "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
          "precursor_mz", "collision_energy_orig", "collision_energy_ev",
          "collision_energy_orig_units"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def row_for_test(spectrum, auxiliary, pos):
    qid = f"f100_{pos:03d}"
    row = {"molecule_id": qid, "spectrum_id": qid + "_s00",
           "ms2_mzs": spectrum["ms2_mzs"],
           "ms2_normalized_intensities": spectrum["ms2_normalized_intensities"],
           "base_peak_intensity": auxiliary["base_peak_intensity"],
           "adduct": spectrum["adduct"], "ionization_mode": spectrum["ionization_mode"],
           "instrument_type": auxiliary["instrument_type"],
           "precursor_mz": spectrum["precursor_mz"],
           "collision_energy_orig": spectrum["collision_energy_orig"],
           "collision_energy_ev": spectrum["collision_energy_ev"],
           "collision_energy_orig_units": spectrum["collision_energy_orig_units"]}
    if list(row) != FIELDS:
        raise AssertionError("Official input field order changed")
    return row


def fixture():
    sample = {"key": "SECRET", "physical_row_index": 123, "normalized_smiles": "CCO",
              "ms2_mzs": [10.0, 11.0], "ms2_normalized_intensities": [1.0, 0.5],
              "adduct": "[M+H]+", "ionization_mode": "positive", "precursor_mz": 50.0,
              "collision_energy_orig": "20", "collision_energy_ev": [20.0],
              "collision_energy_orig_units": "eV"}
    row = row_for_test(sample, {"instrument_type": "qTOF", "base_peak_intensity": 100.0}, 7)
    if (list(row) != FIELDS or row["molecule_id"] != "f100_007" or
            row["spectrum_id"] != "f100_007_s00" or
            any(k in json.dumps(row) for k in ("SECRET", "CCO", "123"))):
        raise AssertionError("Truth leaked from adapter fixture")


def footer_sha(path, footer_bytes):
    with path.open("rb") as f:
        f.seek(-8-footer_bytes, 2)
        return hashlib.sha256(f.read(footer_bytes)).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    fixture()
    if args.fixture:
        print(json.dumps({"status": "fixture_pass", "fields": len(FIELDS)}))
        return
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError("Adapter run ID already exists")
    if sha(SELECTED) != EXPECTED_SELECTED or sha(SPECTRA) != EXPECTED_SPECTRA:
        raise ValueError("Fixed100 manifest or spectra changed")
    selected = json.loads(SELECTED.read_text())["internal_validation"]
    spectra = json.loads(SPECTRA.read_text())["spectra"]
    if len(selected) != 100 or len(spectra) != 100:
        raise ValueError("Fixed100 row count changed")
    for pos, (item, s) in enumerate(zip(selected, spectra)):
        if (s["manifest_position"] != pos or s["physical_row_index"] != item["row_index"] or
                s["key"] != item["key"] or len(s["ms2_mzs"]) != s["num_peaks"] or
                len(s["ms2_normalized_intensities"]) != s["num_peaks"]):
            raise ValueError("Fixed100 physical row or peak identity changed")
    pf = pq.ParquetFile(TRAIN)
    md = pf.metadata
    if (TRAIN.stat().st_size != EXPECTED_TRAIN_BYTES or md.num_rows != EXPECTED_TRAIN_ROWS or
            footer_sha(TRAIN, md.serialized_size) != EXPECTED_FOOTER_SHA):
        raise ValueError("Official train identity changed")
    starts = []
    count = 0
    for g in range(md.num_row_groups):
        starts.append(count)
        count += md.row_group(g).num_rows
    if count != EXPECTED_TRAIN_ROWS:
        raise ValueError("Physical row offsets changed")
    groups = collections.defaultdict(list)
    for pos, s in enumerate(spectra):
        g = bisect.bisect_right(starts, s["physical_row_index"]) - 1
        groups[g].append((s["physical_row_index"]-starts[g], pos))
    auxiliary = [None] * 100
    for g in sorted(groups):
        pending = sorted(groups[g])
        cursor = base = 0
        for batch in pf.iter_batches(row_groups=[g], batch_size=1024,
                                     columns=["instrument_type", "base_peak_intensity"],
                                     use_threads=False):
            end = base + batch.num_rows
            while cursor < len(pending) and pending[cursor][0] < end:
                local, pos = pending[cursor]
                auxiliary[pos] = {name: batch.column(batch.schema.get_field_index(name))[local-base].as_py()
                                  for name in ("instrument_type", "base_peak_intensity")}
                cursor += 1
            base = end
        if cursor != len(pending) or base != md.row_group(g).num_rows:
            raise ValueError("Auxiliary metadata row group incomplete")
    if any(x is None for x in auxiliary):
        raise ValueError("Auxiliary metadata incomplete")
    rows = [row_for_test(s, auxiliary[i], i) for i, s in enumerate(spectra)]
    official_schema = pq.ParquetFile(OFFICIAL_TEST).schema_arrow
    if official_schema.names != FIELDS:
        raise ValueError("Official test schema field order changed")
    table = pa.Table.from_pylist(rows, schema=official_schema)
    if not table.schema.equals(official_schema, check_metadata=True):
        raise ValueError("Official test schema differs")
    truth = [{"molecule_id": rows[i]["molecule_id"], "manifest_position": i,
              "physical_row_index": s["physical_row_index"], "canonical_key": s["key"]}
             for i, s in enumerate(spectra)]
    OUT.mkdir()
    test_path, sample_path, truth_path = (OUT / "test.parquet", OUT / "sample_submission.csv",
                                          OUT / "truth_local.json")
    pq.write_table(table, test_path, compression="snappy")
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id"])
        w.writerows((row["molecule_id"],) for row in rows)
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    if pq.read_table(test_path).to_pylist() != rows:
        raise ValueError("Roundtrip test rows changed")
    if list(csv.DictReader(sample_path.open(newline="", encoding="utf-8"))) != [
            {"molecule_id": row["molecule_id"]} for row in rows]:
        raise ValueError("Sample qid order changed")
    record = {"run_id": OUT.name, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "project_realpath": str(ROOT), "server": "ubuntu/yuanyn", "entry_sha256": sha(Path(__file__)),
              "environment": {"python": sys.version.split()[0], "pyarrow": pa.__version__},
              "source_selected_sha256": EXPECTED_SELECTED, "source_spectra_sha256": EXPECTED_SPECTRA,
              "train_footer_sha256": EXPECTED_FOOTER_SHA, "test_schema_sha256": hashlib.sha256(
                  official_schema.serialize().to_pybytes()).hexdigest(),
              "selected_rowgroups": sorted(groups), "auxiliary_columns": ["instrument_type", "base_peak_intensity"],
              "test_rows": 100, "sample_qids": 100, "total_peaks": sum(s["num_peaks"] for s in spectra),
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in (test_path, sample_path, truth_path)},
              "truth_mapping_local_only": True, "test_fields": FIELDS,
              "elapsed_seconds": time.perf_counter()-started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "sealed_manifest_opened": False}
    (OUT / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "complete", "test_rows": 100,
                      "selected_rowgroups": len(groups), "total_peaks": record["total_peaks"],
                      "elapsed_seconds": record["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
