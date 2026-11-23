"""Build a truth-free dev87 V17 input adapter from frozen safe spectra."""
import csv
import hashlib
import json
import math
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "outputs/phase5_20261023_02/queries.json"
SOURCE = ROOT / "outputs/p5_mass2_20261027_01/full_input"
MGF = SOURCE / "model.mgf"
INPUT_RECORD = SOURCE / "record.json"
OFFICIAL_TEST = ROOT / "data/test.parquet"
OUT = ROOT / "outputs/o1_dev87_v17_adapter_20261029_01"
EXPECTED_QUERIES_SHA = "365524355aacedcedf7ba16f8a218eb16a00f8a251526ae8c0e0a87954a7b3bc"
EXPECTED_FIELDS = [
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
    "precursor_mz", "collision_energy_orig", "collision_energy_ev",
    "collision_energy_orig_units",
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def parse_blocks(raw):
    text = raw.decode("ascii")
    pieces = text.split("BEGIN IONS\n")
    if pieces[0] or len(pieces) != 501:
        raise ValueError("MGF block count or prefix changed")
    blocks = {}
    for piece in pieces[1:]:
        if piece.count("END IONS\n") != 1:
            raise ValueError("Malformed MGF terminator")
        body, terminator, tail = piece.partition("END IONS\n")
        if terminator != "END IONS\n" or tail != "\n":
            raise ValueError("MGF block boundary changed")
        block = "BEGIN IONS\n" + body + terminator + tail
        names = [line.split("=", 1)[1] for line in block.splitlines()
                 if line.startswith("FEATURE_ID=")]
        if len(names) != 1 or names[0] in blocks:
            raise ValueError("MGF spectrum ID missing or repeated")
        blocks[names[0]] = block
    if "".join(blocks.values()) != text:
        raise ValueError("MGF block reconstruction changed bytes")
    return blocks


def parse_original_block(block, spec, query):
    lines = block.splitlines()
    if lines[0] != "BEGIN IONS" or lines[-2:] != ["END IONS", ""]:
        raise ValueError("MGF structure changed")
    header = {}
    mzs, intensities = [], []
    peaks_started = False
    for line in lines[1:-2]:
        if not peaks_started and "=" in line:
            key, value = line.split("=", 1)
            if key in header:
                raise ValueError("Repeated MGF metadata field")
            header[key] = value
            continue
        peaks_started = True
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Unexpected MGF peak encoding")
        mz, intensity = map(float, parts)
        if not (math.isfinite(mz) and mz > 0 and math.isfinite(intensity) and intensity >= 0):
            raise ValueError("Invalid MGF peak")
        mzs.append(mz)
        intensities.append(intensity)
    if set(header) != {"FEATURE_ID", "PEPMASS", "CHARGE", "MSLEVEL"}:
        raise ValueError("Unexpected MGF metadata fields")
    if header["FEATURE_ID"] != spec or header["CHARGE"] != "1+" or header["MSLEVEL"] != "2":
        raise ValueError("MGF metadata identity changed")
    if header["PEPMASS"] != format(float(query["precursor_mz"]), ".12g"):
        raise ValueError("MGF precursor differs from frozen query")
    if len(mzs) != query["input_peaks"] or not mzs:
        raise ValueError("MGF peak count differs from frozen query")
    instrument = query["instrument"]
    if not isinstance(instrument, str) or not instrument.strip():
        raise ValueError("Missing frozen instrument")
    return mzs, intensities, float(header["PEPMASS"]), instrument


def main():
    if OUT.exists():
        raise FileExistsError("Adapter run ID already exists")
    if sha(QUERIES) != EXPECTED_QUERIES_SHA:
        raise ValueError("Frozen safe queries changed")
    input_record = json.loads(INPUT_RECORD.read_text(encoding="utf-8"))
    qids = input_record["qids"]
    if (len(qids) != 87 or len(set(qids)) != 87 or
            input_record["model_qids"] != qids or input_record["failures"] or
            input_record["spectra"] != 500 or input_record["truth_used"] is not False):
        raise ValueError("Frozen dev87 input record changed")
    if sha(MGF) != input_record["mgf_sha256"]:
        raise ValueError("Frozen paired MGF changed")
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    blocks = parse_blocks(MGF.read_bytes())
    official_schema = pq.ParquetFile(OFFICIAL_TEST).schema_arrow
    if official_schema.names != EXPECTED_FIELDS or not all(f.nullable for f in official_schema):
        raise ValueError("Official test schema changed")
    rows = []
    expected_pair_names = set()
    pair_checks = 0
    for qid in qids:
        if not isinstance(qid, str) or not re.fullmatch(r"q\d{3}", qid) or qid not in queries:
            raise ValueError("Invalid or unmapped dev qid")
        specs = queries[qid]["hplus_spectra"]
        if not specs or any(s.get("invalid") for s in specs):
            raise ValueError("Selected qid has missing or invalid H+ spectrum")
        for query in specs:
            spec = query["spec"]
            if not re.fullmatch(re.escape(qid) + r"_s\d{3}", spec):
                raise ValueError("Spectrum ID is not bound to its qid")
            n0, n1 = spec + "__f0", spec + "__f1"
            expected_pair_names.update((n0, n1))
            if n0 not in blocks or n1 not in blocks:
                raise ValueError("Missing formula pair")
            b0 = blocks[n0].replace("FEATURE_ID=" + n0 + "\n",
                                    "FEATURE_ID=" + spec + "\n", 1)
            b1 = blocks[n1].replace("FEATURE_ID=" + n1 + "\n",
                                    "FEATURE_ID=" + spec + "\n", 1)
            if b0 != b1 or sha_bytes(b0.encode("ascii")) != query["raw_block_sha256"]:
                raise ValueError("Paired peak or metadata bytes disagree")
            mzs, intensities, precursor, instrument = parse_original_block(b0, spec, query)
            rows.append({
                "molecule_id": qid,
                "spectrum_id": spec,
                "ms2_mzs": mzs,
                "ms2_normalized_intensities": intensities,
                "base_peak_intensity": None,
                "adduct": "[M+H]+",
                "ionization_mode": "positive",
                "instrument_type": instrument,
                "precursor_mz": precursor,
                "collision_energy_orig": None,
                "collision_energy_ev": None,
                "collision_energy_orig_units": None,
            })
            pair_checks += 1
    if len(rows) != 250 or pair_checks != 250 or set(blocks) != expected_pair_names:
        raise ValueError("Unexpected paired spectrum membership")
    if len({r["spectrum_id"] for r in rows}) != 250:
        raise ValueError("Duplicate original spectrum ID")
    if set(r["molecule_id"] for r in rows) != set(qids):
        raise ValueError("Missing dev qid")
    table = pa.Table.from_pylist(rows, schema=official_schema)
    if not table.schema.equals(official_schema, check_metadata=True):
        raise ValueError("Adapter schema is not the official test schema")
    row_order_sha = digest_json([(r["molecule_id"], r["spectrum_id"]) for r in rows])
    peak_array_sha = digest_json([(r["ms2_mzs"], r["ms2_normalized_intensities"]) for r in rows])
    full_row_sha = digest_json(rows)
    OUT.mkdir()
    test_path = OUT / "test.parquet"
    sample_path = OUT / "sample_submission.csv"
    pq.write_table(table, test_path, compression="snappy")
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule_id"])
        writer.writerows((qid,) for qid in qids)
    reread = pq.read_table(test_path)
    if not reread.schema.equals(official_schema, check_metadata=True):
        raise ValueError("Written Parquet schema changed")
    back = reread.to_pylist()
    if (len(back) != 250 or digest_json(back) != full_row_sha or
            digest_json([(r["molecule_id"], r["spectrum_id"]) for r in back]) != row_order_sha or
            digest_json([(r["ms2_mzs"], r["ms2_normalized_intensities"]) for r in back]) != peak_array_sha):
        raise ValueError("Written Parquet changed rows or peak order")
    with sample_path.open(newline="", encoding="utf-8") as f:
        sample_rows = list(csv.DictReader(f))
    if len(sample_rows) != 87 or list(sample_rows[0]) != ["molecule_id"] or [r["molecule_id"] for r in sample_rows] != qids:
        raise ValueError("Private sample submission changed qid order")
    if any(r["collision_energy_ev"] is not None for r in back):
        raise ValueError("Collision energy must remain missing")
    record = {
        "run_id": OUT.name,
        "status": "complete_local_adapter",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_id": "net-4090d-jyx",
        "server_id": "net-4090d",
        "hostname": "ubuntu",
        "user": "yuanyn",
        "project_realpath": str(ROOT),
        "entry": str(Path(__file__).resolve()),
        "entry_sha256": sha(Path(__file__)),
        "environment": {"python": __import__("sys").version.split()[0], "pyarrow": pa.__version__},
        "input": {
            "queries_path": str(QUERIES), "queries_sha256": sha(QUERIES),
            "mgf_path": str(MGF), "mgf_sha256": sha(MGF),
            "input_record_path": str(INPUT_RECORD), "input_record_sha256": sha(INPUT_RECORD),
            "official_test_schema_source": str(OFFICIAL_TEST),
            "official_test_schema_sha256": sha_bytes(official_schema.serialize().to_pybytes()),
        },
        "output": {
            "test_path": str(test_path), "test_sha256": sha(test_path),
            "test_bytes": test_path.stat().st_size,
            "sample_path": str(sample_path), "sample_sha256": sha(sample_path),
            "sample_bytes": sample_path.stat().st_size,
        },
        "qa": {
            "qid_count": len(qids), "original_spectra": len(rows),
            "formula_pair_blocks": len(blocks), "byte_identical_pairs": pair_checks,
            "total_peaks": sum(len(r["ms2_mzs"]) for r in rows),
            "row_order_sha256": row_order_sha,
            "peak_arrays_order_sha256": peak_array_sha,
            "full_rows_order_sha256": full_row_sha,
            "collision_energy_ev_null_rows": len(rows),
            "collision_energy_orig_null_rows": len(rows),
            "base_peak_intensity_null_rows": len(rows),
            "sample_columns": ["molecule_id"],
            "inference_contains_labels_target_keys_or_smiles": False,
        },
        "limitations": [
            "Collision energy is unavailable in safe source inputs; V17 single-spectrum branch uses its existing 25.0 default for null collision_energy_ev.",
            "MGF peaks were serialized with 12 significant digits upstream; this adapter preserves their recorded order and values without renormalization.",
            "V17 cell 2 binds TRAIN, TEST, and SAMPLE to one discovered directory; cloud use needs a separate explicit path-binding fix and verification before execution.",
            "This local adapter did not run V17 inference, evaluate truth, access sealed70, or submit to Kaggle.",
        ],
    }
    (OUT / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "qid_count": 87, "spectra": 250,
                      "pair_blocks": 500, "test_bytes": test_path.stat().st_size,
                      "sample_bytes": sample_path.stat().st_size}, ensure_ascii=False))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()

