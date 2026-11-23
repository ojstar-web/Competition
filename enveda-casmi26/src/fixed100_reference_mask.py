"""Exclude the pinned fixed100 canonical structures from V17 reference spectra."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rank_aggregation_fixed100_mask_20261030_01"
SELECTED = ROOT / "outputs/own_ranker_isolation_20261028_01/selected_structures.json"
SELECTED_SHA = "367d17d16a1afd45ad5950bea6a33f8d39b54e3d52eea05b67d81a397dc42278"
RUN_STARTED = False
sys.path.insert(0, str(ROOT / "src"))
import o1_dev87_reference_mask as ref  # noqa: E402


def targets():
    if ref.sha(SELECTED) != SELECTED_SHA:
        raise ValueError("Fixed100 selection SHA changed")
    manifest = json.loads(SELECTED.read_text())
    fixed = manifest["internal_validation"]
    if (len(fixed) != 100 or len(manifest["train"]) != 200 or
            len({x["key"] for x in fixed}) != 100 or
            len({x["row_index"] for x in fixed}) != 100):
        raise ValueError("Fixed100 cohort identity changed")
    keys = {x["key"] for x in fixed}
    if any(not isinstance(k, str) or len(k) != 14 or not k.isalpha() or not k.isupper() for k in keys):
        raise ValueError("Invalid canonical key")
    return keys


def fixture(keys, chem, key_func, np):
    et = key_func("CCO")
    cases = [(None, "missing", set()), ("not_a_smiles", "invalid", set()),
             ("C[N+](=O)[O-]", "retain", set()),
             ("[NH3+]CC(=O)[O-]", "retain", set()),
             ("[NH4+]", "charged", set()), ("[Na+].[Cl-]", "multifragment", set()),
             ("CCO", "target_delete", {et})]
    for value, expected, target in cases:
        if ref.classify(value, target, chem, key_func)[0] != expected:
            raise AssertionError("Classification fixture: " + expected)
    if ref.bit_roundtrip(bytearray([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]), np) != bytes([0b01001101, 0b00010011]):
        raise AssertionError("Bit order fixture")
    if len(keys) != 100:
        raise AssertionError("Target fixture")


def main():
    global RUN_STARTED
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        raise RuntimeError("Two CPU cores unavailable")
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem, __version__ as rdkit_version
    import baseline
    pa.set_cpu_count(2)
    if rdkit_version != "2026.03.3" or ref.sha(ref.BASELINE) != ref.EXPECTED["baseline_sha256"]:
        raise ValueError("Pinned canonicalizer changed")
    keys = targets()
    fixture(keys, Chem, baseline.canonical_key, np)
    if args.fixture:
        print(json.dumps({"status": "fixture_pass", "target_count": len(keys)}))
        return
    if OUT.exists():
        raise FileExistsError("Mask run ID already exists")
    OUT.mkdir()
    RUN_STARTED = True
    (OUT / "start.json").write_text(json.dumps({
        "run_id": OUT.name, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_sha256": ref.sha(Path(__file__)), "selected_sha256": SELECTED_SHA,
        "server": "ubuntu/yuanyn", "environment": ".venv/bin/python",
        "limits": {"cpu_cores": 2, "virtual_memory_bytes": 4294967296,
                   "rss_stop_kib": int(3.2 * 1024 * 1024), "wall_seconds": 2400,
                   "gpu": False}, "fixture_pass": True,
    }, indent=2) + "\n")
    pf, starts = ref.check_footer(pq)
    counts = Counter()
    memo = {}
    bits = bytearray()
    digest = hashlib.sha256()
    retained_keys = set()
    target_seen = set()
    baseline.canonical_key.cache_clear()
    for position, value in ref.projection_rows(pf, starts, True):
        ref.digest_value(digest, value)
        if value not in memo:
            memo[value] = ref.classify(value, keys, Chem, baseline.canonical_key)
        cls, key = memo[value]
        counts[cls] += 1
        bits.append(int(cls == "retain"))
        if cls == "retain":
            retained_keys.add(key)
        elif cls == "target_delete":
            target_seen.add(key)
        if position % 4096 == 0 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= int(3.2 * 1024 * 1024):
            raise MemoryError("RSS crossed 3.2 GiB stop threshold")
        if position and position % 250000 == 0:
            print(json.dumps({"rows": position, "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
    n = len(bits)
    reject = sum(counts[k] for k in ref.REJECT_CLASSES)
    if (n != ref.EXPECTED["train_rows"] or
            digest.hexdigest() != ref.EXPECTED["ordered_projection_sha256"] or
            len(target_seen) != 100 or retained_keys & keys or
            counts["retain"] + counts["target_delete"] + reject != n):
        raise ValueError("Full mask identity, coverage or exclusion check failed")
    packed = ref.bit_roundtrip(bits, np)
    if len(packed) != (n+7)//8:
        raise ValueError("Packed mask length changed")
    report = {
        "run_id": OUT.name, "status": "full_verified", "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_id": "net-4090d-jyx", "server_id": "net-4090d", "hostname": os.uname().nodename,
        "user": os.environ.get("USER"), "project_realpath": str(ROOT),
        "entry": str(Path(__file__).resolve()), "entry_sha256": ref.sha(Path(__file__)),
        "source_helper_sha256": ref.sha(Path(ref.__file__)),
        "environment": {"python": sys.version.split()[0], "rdkit": rdkit_version,
                        "pyarrow": pa.__version__, "cpu_affinity_count": len(os.sched_getaffinity(0)),
                        "virtual_memory_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0]},
        "input": {"selected_sha256": SELECTED_SHA, "train_bytes": ref.TRAIN.stat().st_size,
                  "train_rows": n, "train_row_groups": pf.metadata.num_row_groups,
                  "train_footer_sha256": ref.EXPECTED["footer_sha256"],
                  "baseline_sha256": ref.EXPECTED["baseline_sha256"]},
        "projection": {"column": "normalized_smiles", "physical_row_order": True,
                       "rows_scanned": n, "ordered_sha256": digest.hexdigest()},
        "classification": {"target_key_count": len(keys), "target_keys_seen_count": len(target_seen),
                           "exact_unique_input_values_including_null": len(memo),
                           "canonical_key_cache_misses": baseline.canonical_key.cache_info().misses,
                           "retain": counts["retain"], "target_delete": counts["target_delete"],
                           "reject": reject, "reject_breakdown": {k: counts[k] for k in ref.REJECT_CLASSES},
                           "target_retain_canonical_overlap": len(retained_keys & keys),
                           "charge_rule": "net molecular formal charge != 0; no neutralization"},
        "mask": {"path": str(OUT / "mask_full.bin"), "bit_semantics": "1=retain in reference; 0=target_delete or reject",
                 "packbits_bitorder": "little", "bit_index": "physical train row",
                 "bit_count": n, "byte_count": len(packed), "sha256": hashlib.sha256(packed).hexdigest()},
        "elapsed_seconds": time.perf_counter()-started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sealed_manifest_opened": False,
    }
    (OUT / "mask_full.bin").write_bytes(packed)
    (OUT / "record.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "record.sha256").write_text(ref.sha(OUT / "record.json") + "  record.json\n")
    print(json.dumps({"status": report["status"], "rows": n, "target_seen": len(target_seen),
                      "retain": counts["retain"], "reject": reject,
                      "elapsed_seconds": report["elapsed_seconds"], "max_rss_kib": report["max_rss_kib"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN_STARTED:
            (OUT / "failure.json").write_text(json.dumps({
                "status": "failed", "exception": repr(exc),
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }, indent=2) + "\n")
        raise
