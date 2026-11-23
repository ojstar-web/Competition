"""Build a truth-free fixed100 package and a fail-closed rank-aggregation notebook."""

import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rank_aggregation_fixed100_cloud_20261030_02"
ADAPTER = ROOT / "outputs/rank_aggregation_fixed100_adapter_20261030_01"
MASK = ROOT / "outputs/rank_aggregation_fixed100_mask_20261030_01"
BASE_NOTEBOOK = ROOT / "outputs/o1_dev87_private_notebook_20261030_01/source.ipynb"
BASE_SHA = "f2e3ecf1153ed156eb9cedbfc46bd2410a684dcf06bdf2582caadca42bc92a19"
ADAPTER_RECORD_SHA = "f0f13f9bba5a1e0c674f571c7f5f142b492500fb113e844d52a334f70749a4a1"
SELECTED_SHA = "367d17d16a1afd45ad5950bea6a33f8d39b54e3d52eea05b67d81a397dc42278"
TRAIN_FOOTER_SHA = "fce7ae8a9b6403d579218b36663f892853d5a7b28483c95b444923e84faa2acb"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Notebook source anchor missing or ambiguous: " + old[:70])
    return source.replace(old, new)


INFERENCE = '''# Fixed100 rank aggregation: one shared V17 candidate/reference/FP/feature state.
load_neural_models()
pool = build_candidate_pool()                 # Complete official TRAIN candidate pool.
L = load_library(TRAIN, MASK_BITS)            # Fixed100 mask applies to reference spectra only.
rep, rep_key, rep_nm = build_rep(L)

te = pq.read_table(TEST).to_pandas()
te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id'))
if len(te) != 100 or len(mols) != 100 or NFEAT != 31 or len(RANKERS) != 8:
    raise RuntimeError('Fixed100 cohort or ranker shape changed')
print(f"[RANKAGG] {len(te)} spectra across {len(mols)} fixed qids.", flush=True)

def average_tied_desc_rank(prob):
    """Rank 1 is best; equal probabilities receive their average occupied rank."""
    order = np.argsort(-prob, kind='mergesort')
    sorted_prob = prob[order]
    starts = np.r_[0, np.flatnonzero(sorted_prob[1:] != sorted_prob[:-1]) + 1]
    ends = np.r_[starts[1:], len(prob)]
    ranks = np.empty(len(prob), dtype=np.float64)
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + 1 + b) / 2.0
    return ranks

rows_control, rows_rankagg, diag, full_rankings = [], [], [], []
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    control_smis, rankagg_smis = [], []
    full_row = {'molecule_id': mid, 'candidate_pool_indices': [], 'candidate_smiles': [],
                'control_order_positions': [], 'rankagg_order_positions': [],
                'mean_probabilities': [], 'mean_ranks': [], 'per_model_probabilities': [],
                'features_sha256': None}
    if len(nms):
        target = float(np.median(nms))
        specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
        lib_hits = lib_sim(L, specs, target)
        analogs = analog_sim(L, specs, target, rep, rep_key, rep_nm)
        cand = pool.window(target, CFG.PPM_WIN)
        if len(cand) == 0:
            cand = pool.window(target, CFG.PPM_FALLBACK)
        if len(cand):
            cfp = pool.fps(cand)
            lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
            ids, sims = [], []
            for k, s in analogs:
                i = pool.k2i.get(k, -1)
                if i >= 0:
                    ids.append(i); sims.append(s)
            afp = pool.fps(np.array(ids)) if ids else None
            ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
            fsc = frag_scores([pool.smiles[c] for c in cand], specs, ion_mode)
            zlog = model_logits(sub)
            X = rank_features(cfp, lv, afp, np.array(sims, np.float32), zlog, fsc)[:, :NFEAT]
            probabilities = np.stack([m.predict_proba(X)[:, 1] for m in RANKERS])
            if probabilities.shape != (8, len(cand)) or not np.isfinite(probabilities).all():
                raise RuntimeError('GBM probability shape or finite-value check failed')
            mean_probability = np.mean(probabilities, axis=0)
            mean_rank = np.mean([average_tied_desc_rank(p) for p in probabilities], axis=0)
            control_order = np.argsort(-mean_probability)  # Original V17 control.
            rankagg_order = np.lexsort((cand, -mean_probability, mean_rank))
            control_smis = [pool.smiles[cand[i]] for i in control_order[:CFG.TOPN]]
            rankagg_smis = [pool.smiles[cand[i]] for i in rankagg_order[:CFG.TOPN]]
            full_row.update({
                'candidate_pool_indices': [int(x) for x in cand],
                'candidate_smiles': [str(pool.smiles[x]) for x in cand],
                'control_order_positions': [int(x) for x in control_order],
                'rankagg_order_positions': [int(x) for x in rankagg_order],
                'mean_probabilities': [float(x) for x in mean_probability],
                'mean_ranks': [float(x) for x in mean_rank],
                'per_model_probabilities': [[float(x) for x in p] for p in probabilities],
                'features_sha256': hashlib.sha256(X.tobytes()).hexdigest(),
            })
            diag.append((mid, target, len(cand), float(lv.max()),
                         float(sims[0]) if sims else 0.0, float(mean_probability[control_order[0]])))
    if not control_smis: control_smis = ['CCO']
    if not rankagg_smis: rankagg_smis = ['CCO']
    rows_control.append((mid, ';'.join(control_smis[:CFG.TOPN])))
    rows_rankagg.append((mid, ';'.join(rankagg_smis[:CFG.TOPN])))
    full_rankings.append(full_row)
    if gi % 25 == 0 or gi == len(mols)-1:
        print(f"  Fixed100 qids {gi+1}/{len(mols)} | elapsed {time.time()-T0:.1f}s", flush=True)

samp = pd.read_csv(SAMPLE)
if list(samp.columns) != ['molecule_id']:
    raise RuntimeError('Private sample must contain only qid')
def _top25(rows):
    frame = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    frame = samp[['molecule_id']].merge(frame, on='molecule_id', how='left')
    frame['smiles'] = frame['smiles'].fillna('CCO')
    def pad(s):
        tokens = s.split(';')
        if len(tokens) < CFG.TOPN: tokens += ['CCO'] * (CFG.TOPN-len(tokens))
        return ';'.join(tokens[:CFG.TOPN])
    frame['smiles'] = frame['smiles'].apply(pad)
    if (len(frame) != 100 or frame.molecule_id.duplicated().any() or
            frame.smiles.isnull().any() or
            not (frame.smiles.apply(lambda s: len(s.split(';'))) == 25).all()):
        raise RuntimeError('Private top25 output failed row/candidate checks')
    return frame

control_submission = _top25(rows_control)
rankagg_submission = _top25(rows_rankagg)
control_submission.to_csv('/kaggle/working/fixed100_control_top25.csv', index=False)
rankagg_submission.to_csv('/kaggle/working/fixed100_rankagg_top25.csv', index=False)
with open('/kaggle/working/fixed100_full_rankings.jsonl', 'w') as f:
    for row in full_rankings: f.write(json.dumps(row, separators=(',', ':')) + '\\n')
print('[RANKAGG] Frozen paired predictions saved; no truth or sealed manifest read.')
'''


def package(adapter_record, mask_record):
    files = {
        "test.parquet": ADAPTER / "test.parquet",
        "sample_submission.csv": ADAPTER / "sample_submission.csv",
        "mask_full.bin": MASK / "mask_full.bin",
        "record.json": MASK / "record.json",
        "record.sha256": MASK / "record.sha256",
    }
    if (adapter_record["status"] != "complete" or adapter_record["test_rows"] != 100 or
            adapter_record["sample_qids"] != 100 or not adapter_record["truth_mapping_local_only"] or
            adapter_record["source_selected_sha256"] != SELECTED_SHA or
            mask_record["status"] != "full_verified" or
            mask_record["input"]["selected_sha256"] != SELECTED_SHA or
            mask_record["input"]["train_footer_sha256"] != TRAIN_FOOTER_SHA or
            mask_record["projection"]["ordered_sha256"] !=
            "5db4bea535fdb82aa1b24cd3791a08dad727a0a835bbb2be86d175bb092ffc1c" or
            mask_record["classification"]["target_key_count"] != 100 or
            mask_record["classification"]["target_keys_seen_count"] != 100 or
            mask_record["classification"]["target_retain_canonical_overlap"] != 0):
        raise ValueError("Fixed100 package source gate failed")
    if sha(files["mask_full.bin"]) != mask_record["mask"]["sha256"]:
        raise ValueError("Mask bytes changed")
    if files["record.sha256"].read_text().split()[0] != sha(files["record.json"]):
        raise ValueError("Mask record sidecar changed")
    if re.search(r"\b[A-Z]{14}\b", files["record.json"].read_text()):
        raise ValueError("Mask metadata contains a target-like key")
    for name in ("test.parquet", "sample_submission.csv"):
        if sha(files[name]) != adapter_record["files"][name]["sha256"]:
            raise ValueError("Adapter bytes changed: " + name)
    t = pq.read_table(files["test.parquet"])
    if (t.num_rows != 100 or t.column("collision_energy_ev").null_count != 15 or
            not t.schema.equals(pq.ParquetFile(ROOT / "data/test.parquet").schema_arrow,
                                check_metadata=True)):
        raise ValueError("Fixed100 test row or CE count changed")
    manifest = {"run_id": OUT.name, "scope": "fixed100 private rank aggregation diagnostic",
                "created_utc": datetime.now(timezone.utc).isoformat(), "files": {
                    name: {"sha256": sha(path), "bytes": path.stat().st_size}
                    for name, path in files.items()},
                "test_rows": 100, "sample_qids": 100, "mask_bit_count": 2539608,
                "train_footer_sha256": TRAIN_FOOTER_SHA,
                "train_ordered_projection_sha256": mask_record["projection"]["ordered_sha256"],
                "no_truth_key_smiles_or_physical_row_payload": True}
    (OUT / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = OUT / "fixed100_private_inputs.bin"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in files.items():
            z.write(path, arcname=name)
        z.write(OUT / "package_manifest.json", arcname="manifest.json")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(files) | {"manifest.json"}:
            raise ValueError("Archive member set changed")
        for name, meta in manifest["files"].items():
            raw = z.read(name)
            if len(raw) != meta["bytes"] or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError("Archive member identity changed: " + name)
    return archive, manifest


def notebook(archive, dry_run=False):
    if sha(BASE_NOTEBOOK) != BASE_SHA:
        raise ValueError("O1 source notebook changed")
    base = json.loads(BASE_NOTEBOOK.read_text())
    derived = json.loads(BASE_NOTEBOOK.read_text())
    derived["cells"][0]["source"] = [
        "Fixed100 private rank aggregation diagnostic. One shared V17 candidate pool, filtered reference, FPNet, 31 features and eight GBMs. Control is mean probability; candidate is equal-weight mean of per-GBM average-tie ranks. No truth or sealed data in this notebook.\n"]
    s = "".join(derived["cells"][3]["source"])
    s = change(s, "# O1 private dev87 proxy input package. Pinned dataset path verified by private GPU IO probe version 1.",
               "# Fixed100 private input package. Bind only after the uploaded dataset path is verified.")
    s = change(s, "PACKAGE = '/kaggle/input/datasets/okkojbjj/casmi26-o1-dev87-private-inputs-20261030-01/o1_dev87_private_inputs.bin'", "PACKAGE = 'TO_BE_PINNED'")
    s = change(s, "PACKAGE_SHA256 = 'f6e924c0f2f41745753a55d278d7502524cc5182fe591a452347a4a0c168359b'", f"PACKAGE_SHA256 = '{sha(archive)}'")
    s = change(s, "PRIVATE_DIR = '/kaggle/working/o1_dev87_private_inputs'", "PRIVATE_DIR = '/kaggle/working/fixed100_private_inputs'")
    s = change(s, "_manifest['sample_qids'] != 87 or _manifest['test_rows'] != 250", "_manifest['sample_qids'] != 100 or _manifest['test_rows'] != 100")
    s = change(s, "if (_mask_record['status'] != 'full_verified' or\n        _mask_record['consumer_gate']['required_status'] != 'full_verified' or\n        not _mask_record['consumer_gate']['reject_sample_or_incomplete_mask']):", "if (_mask_record['status'] != 'full_verified' or\n        _mask_record['input']['selected_sha256'] != '" + SELECTED_SHA + "'):")
    s = change(s, "_mask_record['classification']['target_key_count'] != 87 or\n        _mask_record['classification']['target_keys_seen_count'] != 87", "_mask_record['classification']['target_key_count'] != 100 or\n        _mask_record['classification']['target_keys_seen_count'] != 100")
    s = change(s, "_md.serialized_size != _mask_record['input']['train_footer_bytes']", "_md.serialized_size != 50891")
    s = change(s, "_h.hexdigest() != _mask_record['projection']['ordered_sha256'] or\n        _h.hexdigest() != _mask_record['consumer_gate']['required_ordered_projection_sha256']", "_h.hexdigest() != _mask_record['projection']['ordered_sha256'] or\n        _h.hexdigest() != _manifest['train_ordered_projection_sha256']")
    s = change(s, "_private_test.num_rows != 250 or _private_test.column('collision_energy_ev').null_count != 250 or\n        len(_sample_ids) != 87", "_private_test.num_rows != 100 or _private_test.column('collision_energy_ev').null_count != 15 or\n        len(_sample_ids) != 100")
    s = change(s, "len(set(_sample_ids)) != 87", "len(set(_sample_ids)) != 100")
    s = change(s, "CE remains null (V17 default 25.0).", "85 CE lists retained; 15 null use V17 default 25.0.")
    s = change(s, "[O1] Private dev87 inputs and full-mask train identity verified;", "[FIXED100] Private inputs and full-mask train identity verified;")
    if "len(set(_sample_ids)) != 100" not in s or "len(set(_sample_ids)) != 87" in s:
        raise AssertionError("Fixed100 distinct-qid gate missing")
    derived["cells"][3]["source"] = s.splitlines(keepends=True)
    derived["cells"][10]["source"] = INFERENCE.splitlines(keepends=True)
    for i, (old, new) in enumerate(zip(base["cells"], derived["cells"])):
        if i not in (0, 3, 10) and old["source"] != new["source"]:
            raise ValueError("Unexpected science cell change: " + str(i))
    for i, cell in enumerate(derived["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{i}")
        cell["execution_count"] = None if cell["cell_type"] == "code" else cell.get("execution_count")
        cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
    if dry_run:
        return derived
    source = OUT / "source.ipynb"
    source.write_text(json.dumps(derived, ensure_ascii=False, indent=1) + "\n")
    diffs = []
    for i in (0, 3, 10):
        diffs.extend(difflib.unified_diff(base["cells"][i]["source"], derived["cells"][i]["source"],
                                          fromfile=f"O1/cell_{i}", tofile=f"fixed100/cell_{i}"))
    (OUT / "source.diff").write_text("".join(diffs))
    if json.loads(source.read_text())["cells"] != derived["cells"]:
        raise ValueError("Notebook JSON roundtrip changed")
    return source


def main():
    ns = {"np": np}
    function_source = INFERENCE[INFERENCE.index("def average_tied_desc_rank"):INFERENCE.index("rows_control, rows_rankagg")]
    exec(function_source, ns)
    probs = np.array([[0.5, 0.5, 0.2], [0.5, 0.5, 0.2]])
    ranks = np.mean([ns["average_tied_desc_rank"](p) for p in probs], axis=0)
    if not np.array_equal(ranks, [1.5, 1.5, 3.0]) or not np.array_equal(
            np.lexsort((np.array([9, 5, 8]), -probs.mean(axis=0), ranks)), [1, 0, 2]):
        raise AssertionError("Average-tie rank or original candidate index fixture failed")
    notebook(ADAPTER / "test.parquet", dry_run=True)
    if OUT.exists():
        raise FileExistsError("Cloud preparation run already exists")
    if sha(ADAPTER / "record.json") != ADAPTER_RECORD_SHA:
        raise ValueError("Adapter record changed")
    adapter_record = json.loads((ADAPTER / "record.json").read_text())
    mask_record = json.loads((MASK / "record.json").read_text())
    OUT.mkdir()
    archive, manifest = package(adapter_record, mask_record)
    source = notebook(archive)
    record = {"run_id": OUT.name, "status": "local_static_ready_not_uploaded",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "entry_sha256": sha(Path(__file__)), "base_notebook_sha256": BASE_SHA,
              "adapter_record_sha256": ADAPTER_RECORD_SHA,
              "mask_record_sha256": sha(MASK / "record.json"),
              "package_sha256": sha(archive), "package_bytes": archive.stat().st_size,
              "source_sha256": sha(source), "source_diff_sha256": sha(OUT / "source.diff"),
              "changed_cell_indices": [0, 3, 10],
              "unchanged_science_cells": [2, 4, 5, 6, 7, 8, 9],
              "expected_external_asset_versions": [
                  "prvsiyan/casmi26-ranker-features/6",
                  "prvsiyan/casmi26-fp-models-v2/1",
                  "aidensong123/casmi26-offline-rdkit-2026033/1",
                  "prvsiyan/chebi-lipidmaps-casmi26/1",
                  "prvsiyan/coconut-casmi26-candidates/2",
                  "thedevastator/open-source-natural-product-annotations/2",
              ],
              "package_members": list(manifest["files"]),
              "truth_mapping_not_packaged": True, "package_binding": "TO_BE_PINNED",
              "uploaded": False, "cloud_run": False, "competition_submission": False}
    (OUT / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "package_bytes": record["package_bytes"],
                      "changed_cells": record["changed_cell_indices"]}))


if __name__ == "__main__":
    main()
