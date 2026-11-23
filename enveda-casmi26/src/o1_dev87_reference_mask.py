"""Build a version-bound, truth-free train reference mask for the dev87 proxy.

Default action validates a small fixed sample. Full mode requires explicit --full;
neither mode reads peaks, labels.tsv, or the sealed split manifest.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/train.parquet"
BASELINE = ROOT / "src/baseline.py"
MEMBERS = ROOT / "outputs/phase5_20261023_01/members.json"
DEV_RECORD = ROOT / "outputs/p5_mass2_20261027_01/full_input/record.json"
OUT_PARENT = ROOT / "outputs/mass_error_calibration_20261029_01"
EXPECTED = {
    "train_bytes": 3033286496,
    "train_rows": 2539608,
    "footer_bytes": 50891,
    "footer_sha256": "fce7ae8a9b6403d579218b36663f892853d5a7b28483c95b444923e84faa2acb",
    "ordered_projection_sha256": "5db4bea535fdb82aa1b24cd3791a08dad727a0a835bbb2be86d175bb092ffc1c",
    "baseline_sha256": "6ca77ff438fba8804bcf966cecd48189ad3ee908418bfd6e367a4480d3564ea9",
    "members_sha256": "577fb395cdb8122b783b3f143f7af640d0871d89a01103fb865580fa909de943",
    "dev_record_sha256": "c36b5c00ff4a2e54d68a26588ecc13e139eba24f3c34fdc8e5a76f88edc71946",
}
SAMPLE_GROUPS = (0, 6, 13, 20)
SAMPLE_ROWS_PER_GROUP = 250
REJECT_CLASSES = ("missing", "invalid", "multifragment", "charged")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def digest_value(h, value):
    """Match the earlier local probe: N, or S + uint64 BE UTF-8 byte length + bytes."""
    if value is None:
        h.update(b"N")
    elif isinstance(value, str):
        data = value.encode("utf-8")
        h.update(b"S")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    else:
        raise TypeError("normalized_smiles is not nullable string")


def classify(value, target_keys, chem, key_func, charge_rule="net"):
    if value is None or not isinstance(value, str) or not value.strip():
        return "missing", None
    mol = chem.MolFromSmiles(value)
    if mol is None:
        return "invalid", None
    parts = chem.GetMolFrags(mol)
    if len(parts) != 1:
        return "multifragment", None
    if charge_rule == "net":
        charged = chem.GetFormalCharge(mol) != 0
    elif charge_rule == "any_atom":  # Comparison with the preserved superseded sample only.
        charged = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    else:
        raise ValueError("Unknown charge rule")
    if charged:
        return "charged", None
    key = key_func(value)
    if key is None:
        return "invalid", None
    return ("target_delete" if key in target_keys else "retain"), key


def load_targets():
    if sha(MEMBERS) != EXPECTED["members_sha256"] or sha(DEV_RECORD) != EXPECTED["dev_record_sha256"]:
        raise ValueError("Frozen safe dev identity changed")
    members = json.loads(MEMBERS.read_text(encoding="utf-8"))
    dev = json.loads(DEV_RECORD.read_text(encoding="utf-8"))
    qids = dev["qids"]
    if (len(qids) != 87 or len(set(qids)) != 87 or
            dev["model_qids"] != qids or dev["failures"] or
            dev["spectra"] != 500 or dev["truth_used"] is not False or
            members.get("sealed_scored") is not False or len(members["members"]) != 140):
        raise ValueError("Safe dev87 member selection changed")
    by_qid = {m["qid"]: m["canonical_target_key"] for m in members["members"]}
    if len(by_qid) != 140 or any(q not in by_qid for q in qids):
        raise ValueError("Safe dev member mapping is not one to one")
    keys = [by_qid[q] for q in qids]
    if (len(set(keys)) != 87 or
            any(not isinstance(k, str) or len(k) != 14 or not k.isalpha() or not k.isupper() for k in keys)):
        raise ValueError("Safe dev canonical keys changed")
    return set(keys), members["parents"]["split_manifest_sha256"]


def check_footer(pq):
    if TRAIN.stat().st_size != EXPECTED["train_bytes"]:
        raise ValueError("Train byte size changed")
    pf = pq.ParquetFile(TRAIN)
    meta = pf.metadata
    n = meta.serialized_size
    with TRAIN.open("rb") as f:
        f.seek(TRAIN.stat().st_size - 8 - n)
        footer_sha = hashlib.sha256(f.read(n)).hexdigest()
    if (n != EXPECTED["footer_bytes"] or footer_sha != EXPECTED["footer_sha256"] or
            meta.num_rows != EXPECTED["train_rows"] or
            str(pf.schema_arrow.field("normalized_smiles").type) != "string"):
        raise ValueError("Train footer/schema identity changed")
    starts = []
    total = 0
    for i in range(meta.num_row_groups):
        starts.append(total)
        total += meta.row_group(i).num_rows
    if total != meta.num_rows:
        raise ValueError("Row-group offsets do not cover train")
    return pf, starts


def projection_rows(pf, starts, full):
    if full:
        position = 0
        for batch in pf.iter_batches(batch_size=4096, columns=["normalized_smiles"], use_threads=False):
            for value in batch.column(0).to_pylist():
                yield position, value
                position += 1
    else:
        for group in SAMPLE_GROUPS:
            iterator = pf.iter_batches(batch_size=SAMPLE_ROWS_PER_GROUP,
                                       row_groups=[group], columns=["normalized_smiles"],
                                       use_threads=False)
            batch = next(iterator)
            values = batch.column(0).to_pylist()
            if len(values) != SAMPLE_ROWS_PER_GROUP:
                raise ValueError("Sample row group has fewer than 250 rows")
            for i, value in enumerate(values):
                yield starts[group] + i, value


def bit_roundtrip(bits, np):
    packed = np.packbits(np.frombuffer(bits, dtype=np.uint8), bitorder="little").tobytes()
    unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if unpacked[:len(bits)].tobytes() != bytes(bits) or unpacked[len(bits):].any():
        raise ValueError("Bitmask pack/unpack or zero tail padding failed")
    if len(packed) != (len(bits) + 7) // 8:
        raise ValueError("Bitmask byte length changed")
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Explicit future full-train run; not used for the current sample validation")
    args = parser.parse_args()
    out = OUT_PARENT / ("dev87_reference_mask_full_netcharge_20261030_01" if args.full
                        else "dev87_reference_mask_sample_netcharge_20261030_01")
    if out.exists():
        raise FileExistsError("Run ID already exists")
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
    sys.path.insert(0, str(ROOT / "src"))
    import baseline
    pa.set_cpu_count(2)
    if rdkit_version != "2026.03.3" or sha(BASELINE) != EXPECTED["baseline_sha256"]:
        raise ValueError("Pinned baseline canonical_key or RDKit changed")
    target_keys, split_sha_from_members = load_targets()
    pf, starts = check_footer(pq)
    # Test class precedence and a non-byte-aligned bit tail without real target values.
    synthetic_target = baseline.canonical_key("CCO")
    fixture = [(None, "missing", set()), ("", "missing", set()),
               ("not_a_smiles", "invalid", set()), ("CCO", "retain", set()),
               ("C[N+](=O)[O-]", "retain", set()),
               ("[NH3+]CC(=O)[O-]", "retain", set()),
               ("[NH4+]", "charged", set()), ("[Cl-]", "charged", set()),
               ("[Na+].[Cl-]", "multifragment", set()),
               ("CCO", "target_delete", {synthetic_target})]
    for value, expected, fixture_targets in fixture:
        if classify(value, fixture_targets, Chem, baseline.canonical_key)[0] != expected:
            raise ValueError("Classification fixture failed: " + expected)
    if bit_roundtrip(bytearray([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]), np) != bytes([0b01001101, 0b00010011]):
        raise ValueError("Bit-order fixture failed")
    baseline.canonical_key.cache_clear()
    counts = Counter()
    memo = {}
    bits = bytearray()
    digest = hashlib.sha256()
    retained_keys = set()
    target_seen = set()
    sample_direct = []
    ranges = []
    last_group = None
    for position, value in projection_rows(pf, starts, args.full):
        digest_value(digest, value)
        if value not in memo:
            memo[value] = classify(value, target_keys, Chem, baseline.canonical_key)
        cls, key = memo[value]
        counts[cls] += 1
        bits.append(int(cls == "retain"))
        if cls == "retain":
            retained_keys.add(key)
        elif cls == "target_delete":
            target_seen.add(key)
        if not args.full:
            sample_direct.append((position, value, cls, key))
            group = next(g for g in SAMPLE_GROUPS if starts[g] <= position < starts[g] + SAMPLE_ROWS_PER_GROUP)
            if group != last_group:
                ranges.append({"row_group": group, "physical_row_start": position,
                               "sampled_rows": SAMPLE_ROWS_PER_GROUP})
                last_group = group
        if len(bits) % (4096 if args.full else SAMPLE_ROWS_PER_GROUP) == 0:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= int(3.2 * 1024 * 1024):
                raise MemoryError("RSS crossed 3.2 GiB stop threshold")
            if not args.full and time.perf_counter() - started >= 110:
                raise TimeoutError("Sample exceeded 110 seconds")
    n = len(bits)
    if args.full:
        ranges = [{"physical_row_start": 0, "sampled_rows": n}]
        if n != EXPECTED["train_rows"] or digest.hexdigest() != EXPECTED["ordered_projection_sha256"]:
            raise ValueError("Full ordered train projection changed")
        if len(target_seen) != 87:
            raise ValueError("Not all 87 target keys were found for exclusion")
    elif n != len(SAMPLE_GROUPS) * SAMPLE_ROWS_PER_GROUP:
        raise ValueError("Sample size changed")
    reject = sum(counts[k] for k in REJECT_CLASSES)
    if n != counts["retain"] + counts["target_delete"] + reject:
        raise ValueError("Mask class counts do not sum to row count")
    if retained_keys & target_seen or retained_keys & target_keys:
        raise ValueError("Target canonical key leaked into retained reference")
    direct_checked = 0
    if not args.full:
        for _, value, cls, key in sample_direct:
            direct_cls, direct_key = classify(value, target_keys, Chem, baseline.canonical_key.__wrapped__)
            if (direct_cls, direct_key) != (cls, key):
                raise ValueError("Memoized and direct per-row classifications differ")
            direct_checked += 1
    legacy_comparison = None
    if not args.full:
        old_out = OUT_PARENT / "dev87_reference_mask_sample_20261029_01"
        old_record_path = old_out / "record.json"
        old_mask_path = old_out / "mask_sample_not_for_inference.bin"
        if sha(old_record_path) != "3b45e09943b29c910578b9caa3fe0a3da44051fd600642b862ac3953c0d1a9e8":
            raise ValueError("Superseded sample record changed")
        old_record = json.loads(old_record_path.read_text(encoding="utf-8"))
        if (old_record["projection"]["ranges"] != ranges or
                old_record["classification"]["target_key_count"] != 87 or
                sha(old_mask_path) != old_record["mask"]["sha256"]):
            raise ValueError("Superseded sample identity changed")
        old_bits = np.unpackbits(np.frombuffer(old_mask_path.read_bytes(), dtype=np.uint8),
                                 bitorder="little")[:n]
        old_counts = Counter()
        changes = Counter()
        for i, (_, value, cls, key) in enumerate(sample_direct):
            old_cls, old_key = classify(value, target_keys, Chem,
                                        baseline.canonical_key.__wrapped__,
                                        charge_rule="any_atom")
            old_counts[old_cls] += 1
            if int(old_cls == "retain") != int(old_bits[i]):
                raise ValueError("Superseded mask disagrees with old rule")
            if old_cls == cls and old_key != key:
                raise ValueError("Unchanged class has a changed canonical key")
            if old_cls != cls:
                mol = Chem.MolFromSmiles(value)
                if (old_cls != "charged" or mol is None or
                        Chem.GetFormalCharge(mol) != 0 or
                        not any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())):
                    raise ValueError("Classification changed outside net-neutral charged atoms")
                changes[cls] += 1
        if (old_counts["retain"] != old_record["classification"]["retain"] or
                old_counts["target_delete"] != old_record["classification"]["target_delete"] or
                sum(old_counts[k] for k in REJECT_CLASSES) != old_record["classification"]["reject"]):
            raise ValueError("Superseded sample class counts disagree")
        legacy_comparison = {
            "old_record_sha256": sha(old_record_path),
            "old_mask_sha256": sha(old_mask_path),
            "old_rule": "any atom formal charge nonzero",
            "new_rule": "net molecular formal charge nonzero",
            "changed_rows": sum(changes.values()),
            "changed_to_retain": changes["retain"],
            "changed_to_target_delete": changes["target_delete"],
            "changed_to_invalid": changes["invalid"],
            "mask_bit_changes": changes["retain"],
            "all_changes_single_fragment_net_zero_with_charged_atoms": True,
        }
    packed = bit_roundtrip(bits, np)
    if any((packed[i // 8] >> (i % 8)) & 1 != bits[i] for i in range(n)):
        raise ValueError("Packed bit-to-row mapping changed")
    report = {
        "run_id": out.name,
        "status": "full_verified" if args.full else "sample_validation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_id": "net-4090d-jyx", "server_id": "net-4090d",
        "hostname": os.uname().nodename, "user": os.environ.get("USER"),
        "project_realpath": str(ROOT),
        "entry": str(Path(__file__).resolve()), "entry_sha256": sha(Path(__file__)),
        "environment": {"python": sys.version.split()[0], "rdkit": rdkit_version,
                        "pyarrow": pa.__version__, "cpu_affinity_count": len(os.sched_getaffinity(0)),
                        "virtual_memory_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0]},
        "input": {"train_path": str(TRAIN), "train_bytes": TRAIN.stat().st_size,
                  "train_rows": pf.metadata.num_rows, "train_row_groups": pf.metadata.num_row_groups,
                  "train_footer_bytes": pf.metadata.serialized_size,
                  "train_footer_sha256": EXPECTED["footer_sha256"],
                  "members_path": str(MEMBERS), "members_sha256": EXPECTED["members_sha256"],
                  "dev_record_path": str(DEV_RECORD), "dev_record_sha256": EXPECTED["dev_record_sha256"],
                  "split_manifest_sha256_from_members_only": split_sha_from_members,
                  "baseline_source_sha256": EXPECTED["baseline_sha256"]},
        "projection": {"column": "normalized_smiles", "physical_row_order": True,
                       "ranges": ranges, "rows_scanned": n,
                       "ordered_sha256": digest.hexdigest(),
                       "digest_encoding": "SHA256: null=N; string=S + uint64 big-endian UTF-8 byte length + UTF-8 bytes; no normalization",
                       "prior_full_ordered_sha256": EXPECTED["ordered_projection_sha256"],
                       "prior_full_sha_role": "required comparison for full mode; sample digest does not establish full version identity"},
        "classification": {"target_key_count": len(target_keys), "target_keys_seen_count": len(target_seen),
                           "target_key_provenance": "Selected from pinned safe members; members contain keys but no SMILES. Phase5 source matched train normalized_smiles via baseline.canonical_key; this sample run does not independently recompute all 87 target keys.",
                           "exact_unique_input_values_including_null": len(memo),
                           "canonical_key_cache_misses": baseline.canonical_key.cache_info().misses,
                           "retain": counts["retain"], "target_delete": counts["target_delete"],
                           "reject": reject,
                           "reject_breakdown": {k: counts[k] for k in REJECT_CLASSES},
                           "charge_rule": "reject only if Chem.GetFormalCharge(mol) != 0; net-neutral nitro and zwitterions are eligible; no neutralization",
                           "class_precedence": ["missing", "invalid", "multifragment", "charged", "target_delete", "retain"],
                           "target_retain_canonical_overlap": 0,
                           "direct_raw_row_checks": direct_checked},
        "comparison_to_superseded_sample": legacy_comparison,
        "consumer_gate": {"required_status": "full_verified",
                          "required_train_rows": EXPECTED["train_rows"],
                          "required_ordered_projection_sha256": EXPECTED["ordered_projection_sha256"],
                          "reject_sample_or_incomplete_mask": True},
        "mask": {"path": str(out / ("mask_full.bin" if args.full else "mask_sample_not_for_inference.bin")),
                 "bit_semantics": "1=retain in reference; 0=target_delete or reject",
                 "packbits_bitorder": "little", "bit_index": "physical row for full; ordered sample range position for sample",
                 "bit_count": n, "byte_count": len(packed),
                 "tail_padding_bits": (8 - n % 8) % 8, "tail_padding_value": 0,
                 "non_aligned_13_bit_fixture_passed": True,
                 "sha256": hashlib.sha256(packed).hexdigest()},
        "scope_limit": [
            "This 87-target exclusion is solely for private dev87 paired reference-library evaluation; both sqrt and linear arms share one filtered library.",
            "Formal Kaggle inference must use the complete official train reference library and must never carry this dev87 mask.",
            "Cloud mask consumer must verify the full ordered normalized_smiles digest, then stream batches and apply corresponding mask slices before accumulating peak arrays; do not read the full train table then Arrow.filter.",
            "The candidate pool retains original train inchikey14, smiles, and all other source columns; only direct and analog reference spectra are filtered.",
            "Member target keys and labels stay local and are absent from mask bytes and this metadata; sealed70 was not opened.",
            "Sample mask is not a full-train mask and must not be used for inference.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    out.mkdir()
    mask_path = Path(report["mask"]["path"])
    mask_path.write_bytes(packed)
    record_path = out / "record.json"
    record_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "record.sha256").write_text(sha(record_path) + "  record.json\n", encoding="ascii")
    print(json.dumps({"status": report["status"], "rows_scanned": n,
                      "retain": counts["retain"], "target_delete": counts["target_delete"],
                      "reject": reject, "direct_checks": direct_checked,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "max_rss_kib": report["max_rss_kib"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

