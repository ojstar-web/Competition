"""P5: fixed two-formula ranking experiment on exposed fresh_dev only.

Reuses the frozen candidate pool and official MIST preprocessing/inference functions.
Target keys are opened only by the separate evaluate action.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
P = {
    "members": "outputs/phase5_20261023_01/members.json",
    "queries": "outputs/phase5_20261023_02/queries.json",
    "pools": "outputs/phase5_20261023_02/pools.json",
    "mgf": "outputs/phase5_20261023_02/hplus.mgf",
    "primary_forms": "outputs/phase5_20261023_02/formulas_primary.json",
    "remaining_forms": "outputs/phase3_20261023_03/formula_audit.json",
    "primary_meta": "outputs/phase5_20261023_03/metadata_primary.json",
    "remaining_meta": "outputs/phase4_20261023_02/formula_metadata.json",
    "selection": "outputs/phase5_20261023_06/selection.json",
    "old_predictions": "outputs/phase5_20261023_12/predictions.json",
    "old_diagnostics": "outputs/phase5_20261023_12/diagnostics.json",
    "old_metrics": "outputs/phase5_20261023_13/metrics.json",
    "checkpoint": "references/mist_fp_canopus_pretrain.ckpt",
    "fingerprint_source": "outputs/phase2_20261023_03/fingerprint_compare.py",
    "mist_source": "src/mist_phase2.py",
}
PIN = {
    "members": "577fb395cdb8122b783b3f143f7af640d0871d89a01103fb865580fa909de943",
    "queries": "365524355aacedcedf7ba16f8a218eb16a00f8a251526ae8c0e0a87954a7b3bc",
    "pools": "ecbcfd92102c36c45618f48852eee6f0e24e7a8d3f7a4ecbf52c0b68941b17d3",
    "mgf": "73f68b49ada5129daed08673f4828b64a526e20fe7319f2011f55c21aaf07a89",
    "primary_forms": "b6f9cd728aa4171dae8e993d90b9a82438bc0137699c0253493c1d0727e9cde1",
    "remaining_forms": "bac9c92f503e8d440090ece04c18fc496b34f06d240af2479de43f8c1cdbf940",
    "old_predictions": "5ccbb209ee95ee964aadd652e0ff7208297865e661fa82223965e4851b1c4ce4",
    "old_diagnostics": "1999427927d7620a25cadde32f50fa84a73e8f04fc9915d01bf545ecb98eab88",
    "selection": "282de6dee33e7869c08b8635cc8c96d7a09f860d66cf8798d9bc6c13e6b313e9",
    "checkpoint": "a5a37e2b269f9a5ec483bd7fd2d3402257ce5f223270d534610fe41c7c4c5dea",
}
QIDS = [f"q{i:03}" for i in range(140)]


def file(label):
    return ROOT / P[label]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def input_hashes():
    hashes = {name: sha(file(name)) for name in P}
    for name, expected in PIN.items():
        if hashes[name] != expected:
            raise ValueError(f"Frozen P5 input changed: {name}")
    return hashes


def formula_maps():
    primary = read(file("primary_forms"))
    remaining = read(file("remaining_forms"))["candidate_formula_by_key"]
    return primary, remaining


def freeze(out):
    """Freeze all 87 choices without opening member target keys."""
    started = time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    hashes = input_hashes()
    queries, pools = read(file("queries")), read(file("pools"))
    selection = read(file("selection"))["queries"]
    old = read(file("old_predictions"))
    old_detail = read(file("old_diagnostics"))
    metadata = {
        **read(file("primary_meta"))["queries"],
        **read(file("remaining_meta"))["queries"],
    }
    primary, remaining = formula_maps()
    if list(queries) != QIDS or set(pools) != set(QIDS) or set(old_detail) != set(QIDS):
        raise ValueError("The frozen 140-query identity changed")
    if set(old) != {"mass_only", "single_formula_hybrid", "fm"} or any(set(old[k]) != set(QIDS) for k in old):
        raise ValueError("Original complete prediction arms changed")
    selected = [q for q in QIDS if selection.get(q, {}).get("status") == "formula_ambiguous_explanation"]
    if len(selected) != 87 or len(set(selected)) != 87:
        raise ValueError("Expected exactly 87 frozen EI-ambiguous H+ queries")
    records, subset = {}, {}
    for qid in selected:
        row, meta, pool = queries[qid], metadata[qid], pools[qid]
        fm = primary if int(qid[1:]) < 64 else remaining
        if not pool or len(pool) != row["candidate_count"] or len(row["formulas"]) < 2:
            raise ValueError("Frozen pool or formula count changed: " + qid)
        if sorted(c["key"] for c in pool) != row["candidate_keys"]:
            raise ValueError("Frozen complete pool changed: " + qid)
        if abs(meta["neutral_mass_da"] - row["neutral_mass_da"]) > 1e-9:
            raise ValueError("Frozen all-adduct neutral mass changed: " + qid)
        ids = row["hplus_positive_spec_ids"]
        if not ids or ids != meta["hplus_positive_spec_ids"] or ids != selection[qid]["all_hplus_spec_ids"]:
            raise ValueError("Incomplete original H+ spectrum IDs: " + qid)
        if row["hplus_input_failure"] or any(s.get("invalid") for s in row["hplus_spectra"]):
            raise ValueError("Frozen 87 includes invalid H+ input: " + qid)
        complete = {c["key"]: fm[c["key"]]["formula"] for c in pool}
        if sorted(set(complete.values())) != row["formulas"]:
            raise ValueError("Full-pool formula grouping changed: " + qid)
        infos = meta["formulas"]
        if {x["formula"] for x in infos} != set(row["formulas"]) or len(infos) != len(row["formulas"]):
            raise ValueError("Frozen full-formula metadata changed: " + qid)
        counts = {f: sum(v == f for v in complete.values()) for f in row["formulas"]}
        for info in infos:
            if info["candidate_count"] != counts[info["formula"]] or abs(info["absolute_mass_error_da"] - abs(info["neutral_exact_mass_da"] - row["neutral_mass_da"])) > 1e-9:
                raise ValueError("Formula population or mass error changed: " + qid)
        top = sorted(infos, key=lambda f: (
            abs(f["neutral_exact_mass_da"] - row["neutral_mass_da"]), f["formula"]))[:2]
        pair_count = len(infos) * len(ids)
        w = len(ids) * sum(x["cartesian_upper_bound"] for x in infos)
        if pair_count > 200 or w > 20_000_000:
            raise ValueError("Original full-formula support budget changed: " + qid)
        choices = []
        for info in top:
            f = info["formula"]
            if not info["upstream_supported"] or info["isotopes"] or info["cartesian_upper_bound"] > 2_000_000:
                raise ValueError("Frozen eligible formula is unsupported: " + qid)
            keys = [c["key"] for c in pool if complete[c["key"]] == f]
            if len(keys) != counts[f]:
                raise ValueError("Incomplete selected formula group: " + qid)
            choices.append({"formula": f, "keys": keys,
                "neutral_exact_mass_da": info["neutral_exact_mass_da"],
                "absolute_mass_error_da": abs(info["neutral_exact_mass_da"] - row["neutral_mass_da"]),
                "cartesian_upper_bound": info["cartesian_upper_bound"]})
            for c in pool:
                if c["key"] in keys:
                    prior = subset.get(c["key"])
                    value = {"key": c["key"], "smiles": c["smiles"]}
                    if prior is not None and prior != value:
                        raise ValueError("Representative SMILES changed across pools: " + c["key"])
                    subset[c["key"]] = value
        original = old_detail[qid]["full_fm_key_order"]
        if len(original) != len(pool) or len(set(original)) != len(pool) or set(original) != set(complete):
            raise ValueError("Original complete FM order changed: " + qid)
        if [next(c["smiles"] for c in pool if c["key"] == k) for k in original[:25]] != old["fm"][qid]:
            raise ValueError("Original FM top25 changed: " + qid)
        records[qid] = {"neutral_mass_da": row["neutral_mass_da"], "hplus_spec_ids": ids,
            "choices": choices, "full_formula_pair_count": pair_count, "full_formula_w": w,
            "chosen_pair_count": 2 * len(ids),
            "chosen_w": len(ids) * sum(c["cartesian_upper_bound"] for c in choices),
            "candidate_count": len(pool)}
    smoke = [min(selected, key=lambda q: (records[q]["chosen_w"], q)),
             max(selected, key=lambda q: (records[q]["chosen_w"], q))]
    if smoke[0] == smoke[1]:
        raise ValueError("Cost extremes must be distinct")
    out.mkdir(parents=True)
    save(out / "protocol.json", {
        "status": "frozen_before_p5_model_or_truth", "source_sha256": sha(__file__),
        "input_sha256": hashes, "qid_order": selected, "unchanged_qids": [q for q in QIDS if q not in selected],
        "smoke_qids": smoke, "queries": records,
        "method": "two complete nearest-mass formulas; native MIST 20ppm/top60+CLS/fixed61; big-packed Morgan4096; mass/F-MIST formula-internal 1:1 alternating then original FM tail",
        "failure_rule": "Either arm's required input failure => both arms whole-qid original FM fallback; no dropped formula or H+ spectrum",
        "evaluation": "All predictions frozen before target-key access; full140 MRR@25 with 87 and original-hit58/miss29 diagnostics",
        "limits": {"threads": 2, "virtual_memory_gib": 12, "job_seconds": 900,
                   "cumulative_cpu_job_wall_seconds": 3600, "formula_cartesian": 2000000,
                   "full_multi_pairs": 200, "full_multi_w": 20000000},
        "truth_in_prediction": False, "sealed_scored": False,
    })
    save(out / "candidate_subset.json", {
        "candidate_sha256": hashes["pools"], "input_manifest_sha256": sha(out / "protocol.json"),
        "candidates": [subset[k] for k in sorted(subset)],
        "pool_sizes": {q: len(pools[q]) for q in selected},
    })
    save(out / "freeze_record.json", {"status": "P5_inputs_frozen",
        "protocol_sha256": sha(out / "protocol.json"),
        "candidate_subset_sha256": sha(out / "candidate_subset.json"),
        "source_sha256": sha(__file__), "eligible": len(selected), "smoke_qids": smoke,
        "selected_formula_spectrum_pairs": sum(records[q]["chosen_pair_count"] for q in selected),
        "selected_cost_proxy_w": sum(records[q]["chosen_w"] for q in selected),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "truth_used": False, "sealed_scored": False})
    print(json.dumps(read(out / "freeze_record.json")), flush=True)


def official_fingerprint(out, mode):
    """Reuse the unchanged official cross-RDKit Morgan4096 audit."""
    import importlib.util
    protocol = read(out / "protocol.json")
    if protocol["source_sha256"] != sha(__file__) or protocol["input_sha256"] != input_hashes():
        raise ValueError("P5 frozen protocol changed")
    source = file("fingerprint_source")
    spec = importlib.util.spec_from_file_location("p5_frozen_morgan_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = out
    module.PROJECT = ROOT
    if mode in ("model", "baseline"):
        module.compute(mode)
    elif mode == "compare":
        module.compare()
        a = read(out / "fingerprints_model.json")["vectors_hex_by_key"]
        b = read(out / "fingerprints_baseline.json")["vectors_hex_by_key"]
        mismatches = sorted(k for k in a if a[k] != b[k])
        save(out / "fingerprint_qid_audit.json", {
            "model_sha256": sha(out / "fingerprints_model.json"),
            "baseline_sha256": sha(out / "fingerprints_baseline.json"),
            "comparison_sha256": sha(out / "fingerprint_comparison.json"),
            "mismatched_keys": mismatches,
            "candidate_count": len(a), "exact_equal_count": len(a) - len(mismatches),
        })
    else:
        raise ValueError(mode)


def raw_blocks():
    """Read only the frozen H+ MGF and retain its original peak bytes."""
    blocks = {}
    for chunk in file("mgf").read_text().split("BEGIN IONS\n")[1:]:
        block = "BEGIN IONS\n" + chunk.split("END IONS\n")[0] + "END IONS\n\n"
        names = [line.split("=", 1)[1] for line in block.splitlines()
                 if line.startswith("FEATURE_ID=")]
        if len(names) != 1 or names[0] in blocks:
            raise ValueError("Invalid original H+ MGF identity")
        blocks[names[0]] = block
    queries = read(file("queries"))
    expected = {spec["spec"] for q in QIDS for spec in queries[q]["hplus_spectra"]
                if not spec.get("invalid")}
    if set(blocks) != expected:
        raise ValueError("Original H+ MGF spectrum set changed")
    for q in QIDS:
        for spec in queries[q]["hplus_spectra"]:
            if not spec.get("invalid") and hashlib.sha256(blocks[spec["spec"]].encode()).hexdigest() != spec["raw_block_sha256"]:
                raise ValueError("Original raw H+ peak block changed: " + spec["spec"])
    return blocks


def pair_name(spec, fi):
    return f"{spec}__f{fi}"


def assign(out, first, last):
    """Create official 20ppm subformula files by complete qid, with atomic qid records."""
    from mist import utils
    started = time.monotonic()
    protocol = read(out / "protocol.json")
    if protocol["source_sha256"] != sha(__file__) or protocol["input_sha256"] != input_hashes():
        raise ValueError("P5 assignment parent changed")
    selected = protocol["qid_order"]
    if not 0 <= first < last <= len(selected):
        raise ValueError("Invalid frozen qid range")
    blocks = raw_blocks()
    parsed = {meta["FEATURE_ID"]: (meta, peaks)
              for meta, peaks in utils.parse_spectra_mgf(file("mgf"))}
    if set(parsed) != set(blocks):
        raise ValueError("Official MGF parse lost a spectrum")
    status_dir = out / "qid_status"
    sub_dir = out / "subforms"
    status_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)
    summary = []
    for qid in selected[first:last]:
        t = time.monotonic()
        status_path = status_dir / f"{qid}.json"
        if status_path.exists():
            prior = read(status_path)
            if prior["protocol_sha256"] != sha(out / "protocol.json") or prior["qid"] != qid:
                raise ValueError("Existing qid checkpoint belongs to another input")
            summary.append({"qid": qid, "status": prior["status"], "reused": True})
            continue
        row = protocol["queries"][qid]
        complete = []
        failure = None
        for fi, choice in enumerate(row["choices"]):
            for original in row["hplus_spec_ids"]:
                name = pair_name(original, fi)
                meta, peaks = parsed[original]
                try:
                    processed = utils.process_spec_file(meta, peaks)
                    if processed is None or len(processed) == 0:
                        raise ValueError("empty_processed_spectrum")
                    result = utils.get_output_dict(spec_name=name, spec=processed.copy(),
                                form=choice["formula"], mass_diff_type="ppm",
                                mass_diff_thresh=20, ion_type="[M+H]+")
                    table = result["output_tbl"]
                    if result["cand_form"] != choice["formula"] or result["cand_ion"] != "[M+H]+" or table is None or not len(table["formula"]):
                        raise ValueError("empty_or_incorrect_subformula_assignment")
                except (ValueError, KeyError) as exc:
                    failure = f"official_subformula_input_failure:{type(exc).__name__}:{exc}"
                    break
                path = sub_dir / f"{name}.json"
                temp = path.with_suffix(".json.partial")
                save(temp, result)
                os.replace(temp, path)
                complete.append({"name": name, "spec": original, "formula": choice["formula"],
                    "processed_peaks": len(processed), "matched_subformulas": len(table["formula"]),
                    "subformula_sha256": sha(path)})
            if failure:
                break
        status = {"qid": qid, "status": "complete" if failure is None else "whole_qid_fallback",
                  "reason": failure, "protocol_sha256": sha(out / "protocol.json"),
                  "expected_pairs": row["chosen_pair_count"], "completed_pairs": complete,
                  "elapsed_seconds": time.monotonic() - t,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        temp = status_path.with_suffix(".json.partial")
        save(temp, status)
        os.replace(temp, status_path)
        summary.append({"qid": qid, "status": status["status"], "pairs": len(complete),
                        "seconds": status["elapsed_seconds"]})
        print(json.dumps(summary[-1]), flush=True)
    record = {"status": "p5_official_subformula_range_complete",
              "protocol_sha256": sha(out / "protocol.json"), "source_sha256": sha(__file__),
              "range": [first, last], "items": summary, "elapsed_seconds": time.monotonic() - started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save(out / f"assign_{first:02}_{last:02}.json", record)


def build_inputs(out, scope):
    """Write MIST labels and MGF with unique formula/spec names; original peaks unchanged."""
    protocol = read(out / "protocol.json")
    if scope not in ("smoke", "full"):
        raise ValueError(scope)
    qids = protocol["smoke_qids"] if scope == "smoke" else protocol["qid_order"]
    queries = read(file("queries"))
    blocks = raw_blocks()
    rows = []
    output_blocks = []
    failures = {}
    for qid in qids:
        path = out / "qid_status" / f"{qid}.json"
        if not path.exists():
            raise ValueError("Missing complete qid assignment status: " + qid)
        status = read(path)
        if status["protocol_sha256"] != sha(out / "protocol.json"):
            raise ValueError("Subformula checkpoint changed")
        if status["status"] != "complete":
            failures[qid] = status["reason"]
            continue
        choices = protocol["queries"][qid]["choices"]
        ids = protocol["queries"][qid]["hplus_spec_ids"]
        expected = [pair_name(spec, fi) for fi in range(2) for spec in ids]
        actual = [r["name"] for r in status["completed_pairs"]]
        if actual != expected or len(actual) != protocol["queries"][qid]["chosen_pair_count"]:
            raise ValueError("Incomplete selected formula x complete H+ set: " + qid)
        by_spec = {r["spec"]: r for r in queries[qid]["hplus_spectra"]}
        for fi, choice in enumerate(choices):
            for spec in ids:
                name = pair_name(spec, fi)
                sub = out / "subforms" / f"{name}.json"
                observed = next(r for r in status["completed_pairs"] if r["name"] == name)
                if sha(sub) != observed["subformula_sha256"]:
                    raise ValueError("Official subformula output changed: " + name)
                x = by_spec[spec]
                rows.append({"spec": name, "formula": choice["formula"], "ionization": "[M+H]+",
                             "dataset": "phase5_mass2_fresh_dev", "compound": qid,
                             "parentmass": x["precursor_mz"], "instrument": x["instrument"]})
                changed = blocks[spec].replace("FEATURE_ID=" + spec + "\n",
                                               "FEATURE_ID=" + name + "\n", 1)
                if changed == blocks[spec] or changed.replace("FEATURE_ID=" + name + "\n",
                                                               "FEATURE_ID=" + spec + "\n", 1) != blocks[spec]:
                    raise ValueError("MGF peak block changed while renaming formula pair")
                output_blocks.append(changed)
    dest = out / f"{scope}_input"
    dest.mkdir(exist_ok=False)
    with (dest / "labels.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["spec", "formula", "ionization",
                     "dataset", "compound", "parentmass", "instrument"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (dest / "model.mgf").write_text("".join(output_blocks))
    save(dest / "record.json", {"status": "p5_model_inputs_frozen",
         "scope": scope, "qids": qids, "model_qids": [q for q in qids if q not in failures],
         "failures": failures, "labels_sha256": sha(dest / "labels.tsv"),
         "mgf_sha256": sha(dest / "model.mgf"), "protocol_sha256": sha(out / "protocol.json"),
         "source_sha256": sha(__file__), "spectra": len(rows), "truth_used": False})
    print(json.dumps({"scope": scope, "qids": len(qids), "model_spectra": len(rows),
                      "failures": failures}), flush=True)


def infer(out, scope, label_file=None, name=None):
    """Use the unchanged official strict checkpoint loader and fixed61 wrapper."""
    from mist_phase2 import infer_fixed61
    if sha(file("checkpoint")) != PIN["checkpoint"]:
        raise ValueError("MIST checkpoint changed")
    labels = Path(label_file) if label_file else out / f"{scope}_input" / "labels.tsv"
    if not labels.exists():
        raise FileNotFoundError(labels)
    dataset = name or f"p5_{scope}"
    save_dir = out / "preds" / dataset
    if save_dir.exists():
        raise FileExistsError(save_dir)
    save_dir.mkdir(parents=True)
    infer_fixed61(SimpleNamespace(model=file("checkpoint"), labels=labels,
                    subforms=out / "subforms", name=dataset, save_dir=save_dir))
    pred = save_dir / f"fp_preds_{dataset}.p"
    if not pred.exists():
        raise FileNotFoundError(pred)
    save(save_dir / "record.json", {"status": "official_MIST_fixed61_predicted",
         "prediction_sha256": sha(pred), "labels_sha256": sha(labels),
         "checkpoint_sha256": PIN["checkpoint"], "mist_source_sha256": sha(file("mist_source")),
         "scope": scope, "dataset": dataset})
    print(json.dumps({"scope": scope, "dataset": dataset, "prediction_sha256": sha(pred)}), flush=True)


def smoke_variants(out):
    """Run unchanged model on every smoke pair alone, together, and in reverse order."""
    import subprocess
    import sys
    source = out / "smoke_input" / "labels.tsv"
    record = read(out / "smoke_input" / "record.json")
    if record["failures"] or record["spectra"] != sum(
            read(out / "protocol.json")["queries"][q]["chosen_pair_count"] for q in record["qids"]):
        raise ValueError("Smoke must include both complete qids")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len({r["spec"] for r in rows}):
        raise ValueError("Duplicate smoke formula/spectrum name")
    variants = [("smoke_reverse", list(reversed(rows)))]
    variants += [(f"smoke_solo_{i:02}", [row]) for i, row in enumerate(rows)]
    paths = []
    for name, variant in variants:
        path = out / "smoke_input" / f"{name}.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(variant)
        paths.append((name, path))
    for name, path in paths:
        cmd = [sys.executable, __file__, "infer", "--out", str(out),
               "--scope", "smoke", "--labels", str(path), "--name", name]
        subprocess.run(cmd, check=True, timeout=120)
        print(json.dumps({"smoke_variant": name, "complete": True}), flush=True)


def smoke_check(out):
    """Check all pair vectors and within-formula orders at the predeclared tolerance."""
    import numpy as np
    from mist_phase2 import rank_cosine
    protocol = read(out / "protocol.json")
    qids = protocol["smoke_qids"]
    labels = []
    with (out / "smoke_input" / "labels.tsv").open(newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    names = [r["spec"] for r in labels]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate smoke pair")
    def vectors(name):
        path = out / "preds" / name / f"fp_preds_{name}.p"
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        x = np.asarray(obj["preds"])
        ns = list(obj["names"])
        if x.shape != (len(ns), 4096) or len(ns) != len(set(ns)) or not np.isfinite(x).all():
            raise ValueError("Invalid smoke model vector")
        return dict(zip(ns, x))
    baseline = vectors("p5_smoke")
    reverse = vectors("smoke_reverse")
    if set(baseline) != set(names) or set(reverse) != set(names):
        raise ValueError("Incomplete forward/reverse smoke")
    solo = {}
    for i, name in enumerate(names):
        item = vectors(f"smoke_solo_{i:02}")
        if set(item) != {name}:
            raise ValueError("Incomplete solo smoke: " + name)
        solo.update(item)
    max_abs = 0.0
    for name in names:
        for other in (reverse, solo):
            max_abs = max(max_abs, float(np.max(np.abs(baseline[name] - other[name]))))
            if not np.allclose(baseline[name], other[name], atol=1e-6, rtol=1e-5):
                raise ValueError("Fixed61 smoke batch composition drift: " + name)
    from numpy import frombuffer, unpackbits, uint8
    fingerprints = read(out / "fingerprints_model.json")["vectors_hex_by_key"]
    pools = read(file("pools"))
    for qid in qids:
        for fi, choice in enumerate(protocol["queries"][qid]["choices"]):
            candidates = [c for c in pools[qid] if c["key"] in set(choice["keys"])]
            cv = [unpackbits(frombuffer(bytes.fromhex(fingerprints[c["key"]]), dtype=uint8), bitorder="big") for c in candidates]
            pair_names = [pair_name(spec, fi) for spec in protocol["queries"][qid]["hplus_spec_ids"]]
            orders = [[k for k, _ in rank_cosine([mapping[s] for s in pair_names], cv,
                      [c["key"] for c in candidates])] for mapping in (baseline, reverse, solo)]
            if not orders[0] == orders[1] == orders[2]:
                raise ValueError("Within-formula smoke rank changed: " + qid)
    save(out / "smoke_check.json", {
        "status": "predeclared_2qid_full_pair_batch_invariance_pass",
        "qids": qids, "pairs": len(names), "max_abs": max_abs,
        "atol": 1e-6, "rtol": 1e-5,
        "forward_sha256": sha(out / "preds" / "p5_smoke" / "fp_preds_p5_smoke.p"),
        "reverse_sha256": sha(out / "preds" / "smoke_reverse" / "fp_preds_smoke_reverse.p"),
        "source_sha256": sha(__file__), "protocol_sha256": sha(out / "protocol.json"),
        "truth_used": False,
    })
    print(json.dumps(read(out / "smoke_check.json")), flush=True)


def alternating(first, second, old_tail):
    """Alternate complete formula orders and append unused old FM keys once."""
    result = []
    i = j = 0
    while i < len(first) or j < len(second):
        if i < len(first):
            result.append(first[i])
            i += 1
        if j < len(second):
            result.append(second[j])
            j += 1
    used = set(result)
    if len(used) != len(result):
        raise ValueError("Selected formulas share a candidate")
    result.extend(k for k in old_tail if k not in used)
    if len(result) != len(old_tail) or len(set(result)) != len(result) or set(result) != set(old_tail):
        raise ValueError("Alternation lost or duplicated a frozen pool candidate")
    return result


def rank(out):
    """Freeze both 140-query policies before opening any target key."""
    import numpy as np
    from mist_phase2 import rank_cosine
    started = time.monotonic()
    protocol = read(out / "protocol.json")
    if protocol["source_sha256"] != sha(__file__) or protocol["input_sha256"] != input_hashes():
        raise ValueError("P5 ranking provenance changed")
    if read(out / "smoke_check.json")["status"] != "predeclared_2qid_full_pair_batch_invariance_pass":
        raise ValueError("Predeclared model smoke is incomplete")
    scope = read(out / "full_input" / "record.json")
    if scope["protocol_sha256"] != sha(out / "protocol.json") or scope["scope"] != "full":
        raise ValueError("Full input was not frozen")
    fp0 = read(out / "fingerprints_model.json")
    fp1 = read(out / "fingerprints_baseline.json")
    if fp0["candidate_subset_sha256"] != sha(out / "candidate_subset.json") or fp1["candidate_subset_sha256"] != sha(out / "candidate_subset.json"):
        raise ValueError("Fingerprint candidate subset changed")
    a, b = fp0["vectors_hex_by_key"], fp1["vectors_hex_by_key"]
    subset = read(out / "candidate_subset.json")["candidates"]
    if set(a) != set(b) or set(a) != {c["key"] for c in subset} or len(a) != len(subset):
        raise ValueError("Cross-RDKit fingerprint key set changed")
    mismatch = {k for k in a if a[k] != b[k]}
    if mismatch != set(read(out / "fingerprint_qid_audit.json")["mismatched_keys"]):
        raise ValueError("Cross-RDKit fingerprint mismatch audit changed")
    pred_path = out / "preds" / "p5_full" / "fp_preds_p5_full.p"
    if scope["spectra"]:
        with pred_path.open("rb") as handle:
            model = pickle.load(handle)
        names = list(model["names"])
        vectors = np.asarray(model["preds"])
        if len(names) != len(set(names)) or vectors.shape != (len(names), 4096):
            raise ValueError("Invalid full MIST model shape or duplicate spectrum")
        by_spec = dict(zip(names, vectors))
    else:
        by_spec = {}
    selected = protocol["qid_order"]
    expected = {pair_name(spec, fi) for q in scope["model_qids"]
                for fi in range(2) for spec in protocol["queries"][q]["hplus_spec_ids"]}
    if set(by_spec) - expected:
        raise ValueError("Unexpected full MIST model spectrum name")
    queries, pools = read(file("queries")), read(file("pools"))
    primary, remaining = formula_maps()
    old = read(file("old_predictions"))
    old_detail = read(file("old_diagnostics"))
    arms = {"mass_only": old["mass_only"], "old_fm": old["fm"],
            "F_mass2": {}, "FM_mass2": {}}
    details = {}
    for qid in QIDS:
        pool = pools[qid]
        by_key = {c["key"]: c for c in pool}
        old_order = old_detail[qid]["full_fm_key_order"]
        mass_order = [] if queries[qid]["neutral_mass_da"] is None else sorted(
            pool, key=lambda c: (abs(c["exact_mass"] - queries[qid]["neutral_mass_da"]), c["key"]))
        if [c["key"] for c in mass_order] != old_detail[qid]["full_mass_key_order"] or len(by_key) != len(pool):
            raise ValueError("Original complete mass/pool order changed: " + qid)
        if set(old_order) != set(by_key) or len(old_order) != len(by_key):
            raise ValueError("Original complete FM order changed: " + qid)
        if qid not in selected:
            arms["F_mass2"][qid] = list(old["fm"][qid])
            arms["FM_mass2"][qid] = list(old["fm"][qid])
            continue
        row = protocol["queries"][qid]
        fm = primary if int(qid[1:]) < 64 else remaining
        statuses = read(out / "qid_status" / f"{qid}.json")
        reason = statuses["reason"] if statuses["status"] != "complete" else None
        if any(k in mismatch for choice in row["choices"] for k in choice["keys"]):
            reason = "cross_rdkit_fingerprint_bit_mismatch"
        names_for_qid = [pair_name(spec, fi) for fi in range(2) for spec in row["hplus_spec_ids"]]
        if reason is None and any(name not in by_spec or not np.isfinite(by_spec[name]).all() or np.linalg.norm(by_spec[name]) <= 0 for name in names_for_qid):
            reason = "incomplete_or_invalid_model_spectrum_prediction"
        if reason is None and any(len(bytes.fromhex(a[k])) != 512 for choice in row["choices"] for k in choice["keys"]):
            reason = "invalid_candidate_fingerprint_bytes"
        if reason is None:
            mass_groups = []
            model_groups = []
            for fi, choice in enumerate(row["choices"]):
                group = [c for c in mass_order if fm[c["key"]]["formula"] == choice["formula"]]
                if [c["key"] for c in group] != [c["key"] for c in mass_order if c["key"] in set(choice["keys"])] or set(c["key"] for c in group) != set(choice["keys"]):
                    raise ValueError("Selected formula is not complete: " + qid)
                keys = [c["key"] for c in group]
                mass_groups.append(keys)
                cv = []
                for c in group:
                    packed = bytes.fromhex(a[c["key"]])
                    if len(packed) != 512:
                        raise ValueError("Candidate fingerprint length changed: " + c["key"])
                    bit = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
                    if bit.shape != (4096,):
                        raise ValueError("Candidate fingerprint vector changed: " + c["key"])
                    cv.append(bit)
                specs = [pair_name(s, fi) for s in row["hplus_spec_ids"]]
                if len(group) == 1:
                    model_groups.append(keys)
                else:
                    model_groups.append([k for k, _ in rank_cosine([by_spec[s] for s in specs], cv, keys)])
            f_order = alternating(mass_groups[0], mass_groups[1], old_order)
            fm_order = alternating(model_groups[0], model_groups[1], old_order)
        else:
            f_order = fm_order = list(old_order)
        arms["F_mass2"][qid] = [by_key[k]["smiles"] for k in f_order[:25]]
        arms["FM_mass2"][qid] = [by_key[k]["smiles"] for k in fm_order[:25]]
        details[qid] = {"choices": [c["formula"] for c in row["choices"]], "status": "ranked" if reason is None else "both_arms_old_fm_fallback",
                        "failure_reason": reason, "hplus_spectra": len(row["hplus_spec_ids"]),
                        "F_complete_order": f_order, "FM_complete_order": fm_order}
        print(json.dumps({"stage": "rank_complete", "qid": qid, "status": details[qid]["status"]}), flush=True)
    if len(details) != 87 or any(set(arms[arm]) != set(QIDS) for arm in arms):
        raise ValueError("P5 denominator changed")
    for q in set(QIDS) - set(selected):
        if arms["F_mass2"][q] != old["fm"][q] or arms["FM_mass2"][q] != old["fm"][q]:
            raise ValueError("One of the 53 unchanged queries drifted: " + q)
    path = out / "predictions.json"
    if path.exists():
        raise FileExistsError(path)
    save(path, arms)
    save(out / "ranking_diagnostics.json", details)
    save(out / "prediction_record.json", {
        "status": "P5_140_predictions_frozen_before_truth_evaluation",
        "source_sha256": sha(__file__), "protocol_sha256": sha(out / "protocol.json"),
        "model_prediction_sha256": sha(pred_path), "fingerprint_audit_sha256": sha(out / "fingerprint_qid_audit.json"),
        "predictions_sha256": sha(path), "ranking_diagnostics_sha256": sha(out / "ranking_diagnostics.json"),
        "selected_qids": len(selected), "unchanged_qids": 53,
        "both_arm_fallback_qids": [q for q in selected if details[q]["status"] != "ranked"],
        "elapsed_seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "truth_used": False, "sealed_scored": False})
    print(json.dumps(read(out / "prediction_record.json")), flush=True)


def evaluate(out):
    """Score the already frozen four arms using 140 official canonical target keys."""
    import baseline as baseline
    started = time.monotonic()
    protocol = read(out / "protocol.json")
    frozen = read(out / "prediction_record.json")
    path = out / "predictions.json"
    if (out / "evaluation.json").exists():
        raise FileExistsError("P5 truth evaluation already exists")
    if frozen["status"] != "P5_140_predictions_frozen_before_truth_evaluation" or frozen["predictions_sha256"] != sha(path):
        raise ValueError("P5 predictions were not frozen")
    if frozen["protocol_sha256"] != sha(out / "protocol.json") or frozen["source_sha256"] != sha(__file__) or protocol["input_sha256"] != input_hashes():
        raise ValueError("P5 evaluation input/source identity changed")
    arms = read(path)
    if list(arms) != ["mass_only", "old_fm", "F_mass2", "FM_mass2"]:
        raise ValueError("P5 arm identity changed")
    old = read(file("old_predictions"))
    if arms["mass_only"] != old["mass_only"] or arms["old_fm"] != old["fm"]:
        raise ValueError("Original reference predictions changed")
    pools = read(file("pools"))
    keys_by_smiles = {}
    for arm, table in arms.items():
        if list(table) != QIDS:
            raise ValueError("All-140 qid order changed: " + arm)
        for q in QIDS:
            if not isinstance(table[q], list) or len(table[q]) > 25 or len(table[q]) != min(25, len(pools[q])):
                raise ValueError("Incomplete top25 prediction: " + arm + "/" + q)
            observed = []
            for smiles in table[q]:
                key = keys_by_smiles.setdefault(smiles, baseline.canonical_key(smiles))
                if not key:
                    raise ValueError("Invalid predicted SMILES: " + q)
                observed.append(key)
            if len(observed) != len(set(observed)) or not set(observed) <= {c["key"] for c in pools[q]}:
                raise ValueError("Invalid or duplicate pool representative: " + arm + "/" + q)
    # This is the first target-key access. Prediction files and their SHA already exist.
    members = read(file("members"))["members"]
    if len(members) != 140 or [m["qid"] for m in members] != QIDS:
        raise ValueError("Frozen member identity changed")
    truth = {m["qid"]: m["canonical_target_key"] for m in members}
    if len(set(truth.values())) != 140:
        raise ValueError("Frozen target keys changed")
    ranks = {arm: {q: next((i for i, s in enumerate(table[q], 1)
                 if keys_by_smiles[s] == truth[q]), None) for q in QIDS}
             for arm, table in arms.items()}
    old_metrics = read(file("old_metrics"))["all140"]
    def rr(arm, q):
        r = ranks[arm][q]
        return 0.0 if r is None else 1.0 / r
    def metrics(arm, qids):
        values = [rr(arm, q) for q in qids]
        n = len(qids)
        top1 = sum(x == 1 for x in values)
        top5 = sum(x >= .2 for x in values)
        top25 = sum(x > 0 for x in values)
        return {"n": n, "mrr25": sum(values) / n, "top1_count": top1,
                "top5_count": top5, "top25_count": top25}
    for new, old_name in [("old_fm", "fm"), ("mass_only", "mass_only")]:
        actual = metrics(new, QIDS)
        expected = old_metrics[old_name]
        for k, v in [("mrr25", "mrr25"), ("top1_count", "top1_count"),
                     ("top5_count", "top5_count"), ("top25_count", "top25_count")]:
            if abs(actual[k] - expected[v]) > 1e-12:
                raise ValueError("Old 140 prediction score is not equivalent: " + new + "/" + k)
    selected = protocol["qid_order"]
    hit = [q for q in selected if rr("old_fm", q) > 0]
    miss = [q for q in selected if rr("old_fm", q) == 0]
    if len(hit) != 58 or len(miss) != 29:
        raise ValueError("Frozen 87 old-hit/miss denominator changed")
    groups = {"all140": QIDS, "selected87": selected, "old_hit58": hit,
              "old_miss29": miss}
    result = {"status": "P5_fixed_policy_development_evaluated_once",
              "protocol_sha256": sha(out / "protocol.json"),
              "predictions_sha256": sha(path),
              "source_sha256": sha(__file__), "groups": {},
              "sealed_scored": False, "cloud_submitted": False}
    for label, qids in groups.items():
        tables = {arm: metrics(arm, qids) for arm in arms}
        comparisons = {}
        for left, right in [("old_fm", "F_mass2"), ("old_fm", "FM_mass2"),
                            ("F_mass2", "FM_mass2")]:
            diffs = [rr(right, q) - rr(left, q) for q in qids]
            comparisons[f"{left}_to_{right}"] = {
                "win": sum(d > 0 for d in diffs),
                "tie": sum(d == 0 for d in diffs),
                "loss": sum(d < 0 for d in diffs),
                "delta_mrr25": sum(diffs) / len(qids)}
        result["groups"][label] = {"arms": tables, "comparisons": comparisons}
    main = result["groups"]["all140"]["arms"]
    oldfm = main["old_fm"]
    newfm = main["FM_mass2"]
    newf = main["F_mass2"]
    result["prespecified_gate"] = {
        "delta_mrr_at_least_0_01": newfm["mrr25"] - oldfm["mrr25"] >= .01,
        "top25_not_lower": newfm["top25_count"] >= oldfm["top25_count"],
        "fm_mass2_not_below_f_mass2": newfm["mrr25"] >= newf["mrr25"],
    }
    result["prespecified_gate"]["all_pass"] = all(result["prespecified_gate"].values())
    by_qid = {q: {"rank": {a: ranks[a][q] for a in arms},
                  "rr": {a: rr(a, q) for a in arms},
                  "delta_old_fm_to_F": rr("F_mass2", q) - rr("old_fm", q),
                  "delta_old_fm_to_FM": rr("FM_mass2", q) - rr("old_fm", q),
                  "delta_F_to_FM": rr("FM_mass2", q) - rr("F_mass2", q)}
              for q in QIDS}
    save(out / "evaluation.json", result)
    save(out / "evaluation_per_qid.json", by_qid)
    save(out / "evaluation_record.json", {
        "status": "P5_evaluation_complete", "evaluation_sha256": sha(out / "evaluation.json"),
        "per_qid_sha256": sha(out / "evaluation_per_qid.json"),
        "predictions_sha256": sha(path), "source_sha256": sha(__file__),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "official_score": "RDKit2026 canonical key first top25 rank; MRR@25"})
    print(json.dumps({"all140": result["groups"]["all140"],
                      "gate": result["prespecified_gate"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "fp", "assign", "build",
                        "infer", "smoke-variants", "smoke-check", "rank", "evaluate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["model", "baseline", "compare"])
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--scope", choices=["smoke", "full"])
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.action == "freeze":
        freeze(args.out)
    elif args.action == "fp":
        official_fingerprint(args.out, args.mode)
    elif args.action == "assign":
        assign(args.out, args.first, args.last)
    elif args.action == "build":
        build_inputs(args.out, args.scope)
    elif args.action == "infer":
        infer(args.out, args.scope, args.labels, args.name)
    elif args.action == "smoke-variants":
        smoke_variants(args.out)
    elif args.action == "smoke-check":
        smoke_check(args.out)
    elif args.action == "rank":
        rank(args.out)
    elif args.action == "evaluate":
        evaluate(args.out)
