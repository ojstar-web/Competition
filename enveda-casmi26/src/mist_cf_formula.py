"""Frozen, truth-free MIST-CF formula ranking over 87 existing EI-ambiguous queries."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/mistcf_20261025_01"
OFFICIAL = ROOT / "references/mist-cf-71c820b43825e0b0a58680a3690f29a7352290f4"
CHECKPOINT = ROOT / "references/mist_cf_nplib1_canopus_best.ckpt"
PATHS = {
    "loss": ROOT / "outputs/p0p1_20261025_01/loss_waterfall.json",
    "members": ROOT / "outputs/phase5_20261023_01/members.json",
    "queries": ROOT / "outputs/phase5_20261023_02/queries.json",
    "pools": ROOT / "outputs/phase5_20261023_02/pools.json",
    "primary_forms": ROOT / "outputs/phase5_20261023_02/formulas_primary.json",
    "remaining_forms": ROOT / "outputs/phase3_20261023_03/formula_audit.json",
    "mgf": ROOT / "outputs/phase5_20261023_02/hplus.mgf",
    "ei": ROOT / "outputs/phase5_20261023_06/selection.json",
}
FORM_TOKENS = re.compile(r"([A-Z][a-z]?)([0-9]*)")
MAX_PAIR_SECONDS = 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def formula_bound(formula):
    tokens = FORM_TOKENS.findall(formula)
    if "".join(a + b for a, b in tokens) != formula:
        raise ValueError(f"Unsupported formula syntax: {formula}")
    return math.prod((int(n) if n else 1) + 1 for _, n in tokens)


def freeze():
    loss = load(PATHS["loss"])
    queries = load(PATHS["queries"])
    pools = load(PATHS["pools"])
    primary = load(PATHS["primary_forms"])
    remaining = load(PATHS["remaining_forms"])["candidate_formula_by_key"]
    forms_by_key = {**primary, **remaining}
    ids = sorted(x["qid"] for x in loss["per_qid"]
                 if x["fallback_reason"] == "formula_ambiguous_explanation")
    assert len(ids) == 87 and len(set(ids)) == 87
    items = {}
    for qid in ids:
        q, pool = queries[qid], pools[qid]
        forms = q["formulas"]
        assert len(forms) == q["formula_count"] and len(forms) > 1
        assert len(pool) == q["candidate_count"]
        assert len({x["key"] for x in pool}) == len(pool)
        assert set(forms) == {forms_by_key[x["key"]]["formula"] for x in pool}
        specs = q["hplus_spectra"]
        assert [s["spec"] for s in specs] == q["hplus_positive_spec_ids"]
        assert specs and all(s["instrument"] == "timsTOF" for s in specs)
        masses = {}
        for form in forms:
            vals = [float(x["exact_mass"]) for x in pool
                    if forms_by_key[x["key"]]["formula"] == form]
            assert vals and max(vals) - min(vals) <= 1e-6
            masses[form] = vals[0]
        items[qid] = {
            "formulas": forms,
            "formula_masses_da": masses,
            "hplus_spectra": specs,
            "neutral_mass_da": q["neutral_mass_da"],
            "cartesian_bounds": {f: formula_bound(f) for f in forms},
            "pair_count": len(forms) * len(specs),
            "cost_proxy_w": len(specs) * sum(formula_bound(f) for f in forms),
        }
    extremes = [min(ids, key=lambda q: (items[q]["cost_proxy_w"], q)),
                max(ids, key=lambda q: (items[q]["cost_proxy_w"], q))]
    assert extremes == ["q007", "q085"]
    protocol = {
        "status": "frozen_before_mistcf_prediction_or_truth_evaluation",
        "source_commit": "71c820b43825e0b0a58680a3690f29a7352290f4",
        "checkpoint_md5": "bff0db38c4ae1e68bd93bf3e3ec46094",
        "checkpoint_sha256": sha(CHECKPOINT),
        "official_predict_sha256": sha(OFFICIAL / "src/mist_cf/mist_cf_score/predict.py"),
        "official_common_init_patched_sha256": sha(OFFICIAL / "src/mist_cf/common/__init__.py"),
        "parent_sha256": {k: sha(v) for k, v in PATHS.items()},
        "qid_order": ids,
        "smoke_extremes": extremes,
        "native_preprocessing": "official .ms parser/merge precision4, <=precursor+1, max-normalize; max_thresh_spec top100/intensity>0; all neutral formulas+[M+H]+; official instrument Unknown (LCMS) tolerance 15ppm; checkpoint max_subpeak20 and cls_mass_diff=False",
        "instrument_mapping": {"timsTOF": "Unknown (LCMS)"},
        "score_rule": "within each spectrum, log_softmax(raw model logits/T=1) over its complete candidate formula set; equal mean across all H+ spectra; descending score with formula-string tie",
        "failure_rule": "Any missing/invalid original H+ spectrum, unsupported candidate formula, incomplete candidate score or pair timeout fails whole qid; retains 87 denominator and model RR zero. Zero matched fragments is valid CLS-only.",
        "per_pair_technical_timeout_seconds": MAX_PAIR_SECONDS,
        "resource": {"threads": 2, "address_space_gib": 12, "single_job_seconds": 900,
                     "cumulative_cpu_wall_seconds": 3600, "new_download_mib": 100},
        "evaluation": "After full prediction freeze only, use official-key-linked true formula. Formula MRR is 1/full formula rank, no top25 truncation; Top1/Top2. Compare frozen mass-error and EI full formula orders, 87 denominator.",
        "sealed_scored": False,
    }
    dump(RUN / "inputs.json", items)
    protocol["inputs_sha256"] = sha(RUN / "inputs.json")
    protocol["adapter_source_sha256"] = sha(__file__)
    dump(RUN / "protocol.json", protocol)
    print(json.dumps({"qids": len(ids), "spectra": sum(len(items[q]["hplus_spectra"]) for q in ids),
                      "pairs": sum(items[q]["pair_count"] for q in ids),
                      "extremes": extremes, "protocol_sha256": sha(RUN / "protocol.json")}))
    

def frozen():
    protocol, items = load(RUN / "protocol.json"), load(RUN / "inputs.json")
    assert protocol["status"] == "frozen_before_mistcf_prediction_or_truth_evaluation"
    assert protocol["inputs_sha256"] == sha(RUN / "inputs.json")
    assert protocol["adapter_source_sha256"] == sha(__file__)
    assert protocol["checkpoint_sha256"] == sha(CHECKPOINT)
    for name, path in PATHS.items():
        assert protocol["parent_sha256"][name] == sha(path), name
    return protocol, items


def native():
    import sys
    sys.path.insert(0, str(OFFICIAL / "src"))
    import mist_cf.common as common
    return common


def mgf_spectra(common):
    parsed = common.parse_spectra_mgf(str(PATHS["mgf"]))
    out = {}
    for meta, pairs in parsed:
        name = meta["FEATURE_ID"]
        assert name not in out and len(pairs) == 1
        out[name] = (float(meta["PEPMASS"]), pairs[0][1])
    return out


def pair_alarm(_signum, _frame):
    raise TimeoutError("native formula-spectrum pair exceeded frozen 60s")


def prepare(qids):
    import numpy as np
    common = native()
    protocol, inputs = frozen()
    allowed = set(protocol["qid_order"])
    assert qids and len(set(qids)) == len(qids) and set(qids) <= allowed
    source = mgf_spectra(common)
    data_dir = RUN / "native"
    spec_dir = data_dir / "spec_files"
    subforms = RUN / "subforms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    subforms.mkdir(exist_ok=True)
    summaries = {}
    start = time.monotonic()
    for qid in qids:
        q = inputs[qid]
        summary_path = RUN / "prepared" / f"{qid}.json"
        summary_path.parent.mkdir(exist_ok=True)
        if summary_path.exists():
            summaries[qid] = load(summary_path)
            continue
        info = {"qid": qid, "status": "prepared", "specs": [],
                "formula_count": len(q["formulas"]), "pair_count": q["pair_count"],
                "cost_proxy_w": q["cost_proxy_w"], "root_only_pairs": 0}
        try:
            for sp in q["hplus_spectra"]:
                spec = sp["spec"]
                mz, raw = source[spec]
                if not math.isclose(mz, sp["precursor_mz"], abs_tol=1e-6):
                    raise ValueError(f"precursor mismatch {spec}")
                if len(raw) != sp["input_peaks"] or not np.isfinite(raw).all():
                    raise ValueError(f"raw peaks mismatch {spec}")
                # The author-native .ms writer retains original peak order and units.
                ms = common.spec_to_ms_str(
                    [("ms2", raw)],
                    {"parentmass": f"{mz:.12g}", "precursor_mz": f"{mz:.12g}"})
                (spec_dir / f"{spec}.ms").write_text(ms, encoding="utf-8")
                _, processed = common.process_spec_file_unbinned(spec, data_dir)
                if processed is None:
                    raise ValueError(f"empty processed spectrum {spec}")
                processed = common.max_thresh_spec(processed, max_peaks=100, inten_thresh=0.0)
                if len(processed) == 0:
                    raise ValueError(f"empty top100 spectrum {spec}")
                exports = []
                for form in q["formulas"]:
                    old = signal.signal(signal.SIGALRM, pair_alarm)
                    signal.alarm(MAX_PAIR_SECONDS)
                    try:
                        result = common.get_output_dict(spec, processed.copy(), form, "ppm",
                                                        common.get_instr_tol("Unknown (LCMS)"), "[M+H]+")
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old)
                    assert result["cand_form"] == form and result["cand_ion"] == "[M+H]+"
                    exports.append(result)
                    info["root_only_pairs"] += result["output_tbl"] is None
                data = {x["cand_form"]: {"cand_ion": x["cand_ion"],
                                         "cand_tbl": x["output_tbl"]} for x in exports}
                assert set(data) == set(q["formulas"])
                dump(subforms / f"{spec}.json", data)
                info["specs"].append({"spec": spec, "raw_peaks": len(raw),
                                      "processed_peaks": len(processed),
                                      "subform_sha256": sha(subforms / f"{spec}.json")})
            assert len(info["specs"]) == len(q["hplus_spectra"])
        except Exception as exc:
            info["status"] = "failed"
            info["error"] = type(exc).__name__ + ": " + str(exc)
            info["specs"] = []
        info["elapsed_seconds"] = time.monotonic() - start
        info["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(summary_path, info)
        summaries[qid] = info
        print(json.dumps({"qid": qid, "status": info["status"],
                          "elapsed": round(info["elapsed_seconds"], 2),
                          "root_only_pairs": info["root_only_pairs"]}), flush=True)
    return summaries


def model_and_dataset():
    import torch
    from mist_cf.mist_cf_score.mist_cf_model import MistNet
    from mist_cf.mist_cf_score.mist_cf_data import PredDataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = MistNet.load_from_checkpoint(str(CHECKPOINT), map_location="cpu", strict=True)
    assert model.max_subpeak == 20 and model.cls_mass_diff is False
    model.eval()
    return torch, model, PredDataset


def inference_rows(qids, inputs):
    import pandas as pd
    rows = []
    for qid in qids:
        for sp in inputs[qid]["hplus_spectra"]:
            for form in inputs[qid]["formulas"]:
                rows.append({"spec": sp["spec"], "cand_form": form,
                             "cand_ion": "[M+H]+", "parentmass": sp["precursor_mz"],
                             "instrument": "Unknown (LCMS)"})
    return pd.DataFrame(rows)


def infer(df, batch_size, model, dataset_type, torch):
    from torch.utils.data import DataLoader
    ds = dataset_type(df, subform_dir=RUN / "subforms", max_subpeak=model.max_subpeak,
                      num_workers=0, ablate_cls_error=not model.cls_mass_diff)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False,
                        collate_fn=ds.get_collate_fn())
    out = {}
    with torch.no_grad():
        for batch in loader:
            scores = model.forward(batch["num_peaks"], batch["types"],
                                   batch["form_vec"], batch["ion_vec"],
                                   batch["instrument_vec"], batch["intens"],
                                   batch["rel_mass_diffs"]).reshape(-1).cpu().numpy()
            names, forms = batch["names"], batch["str_forms"]
            assert len(scores) == len(names) == len(forms)
            for name, form, score in zip(names, forms, scores):
                key = (str(name), str(form))
                assert key not in out and math.isfinite(float(score))
                out[key] = float(score)
    assert len(out) == len(df)
    return out


def smoke():
    import numpy as np
    protocol, inputs = frozen()
    qids = protocol["smoke_extremes"]
    for q in qids:
        s = load(RUN / "prepared" / f"{q}.json")
        assert s["status"] == "prepared"
    native()
    torch, model, PredDataset = model_and_dataset()
    df = inference_rows(qids, inputs)
    specs = [s["spec"] for q in qids for s in inputs[q]["hplus_spectra"]]
    alone = {}
    for s in specs:
        alone.update(infer(df[df.spec == s], 1, model, PredDataset, torch))
    batch = infer(df, 2, model, PredDataset, torch)
    reverse = infer(df.iloc[::-1].copy(), 2, model, PredDataset, torch)
    assert set(alone) == set(batch) == set(reverse)
    diffs = [max(abs(alone[k]-batch[k]),abs(alone[k]-reverse[k])) for k in alone]
    for k in alone:
        assert np.isclose(alone[k], batch[k], atol=1e-6, rtol=1e-5)
        assert np.isclose(alone[k], reverse[k], atol=1e-6, rtol=1e-5)
    def ordered(scores, spec):
        return sorted([f for (s,f) in scores if s == spec],
                      key=lambda f: (-scores[(spec,f)],f))
    assert all(ordered(alone,s) == ordered(batch,s) == ordered(reverse,s) for s in specs)
    report = {"status": "native_smoke_pass", "qids": qids, "spectra": len(specs),
              "formula_spectrum_pairs": len(alone), "maximum_absolute_score_difference": max(diffs),
              "atol": 1e-6, "rtol": 1e-5, "checkpoint_sha256": sha(CHECKPOINT),
              "protocol_sha256": sha(RUN / "protocol.json"),
              "source_sha256": sha(__file__), "sealed_scored": False}
    dump(RUN / "smoke.json", report)
    print(json.dumps(report))


def predict():
    """Freeze complete formula rankings before reading any true formula."""
    import numpy as np
    protocol, inputs = frozen()
    statuses = {q: load(RUN / "prepared" / f"{q}.json")
                for q in protocol["qid_order"]}
    assert all(s["qid"] == q for q, s in statuses.items())
    runnable = [q for q in protocol["qid_order"]
                if statuses[q]["status"] == "prepared"]
    native()
    torch, model, PredDataset = model_and_dataset()
    df = inference_rows(runnable, inputs) if runnable else None
    scores = infer(df, 8, model, PredDataset, torch) if runnable else {}
    predictions = {}
    for qid in protocol["qid_order"]:
        inp, status = inputs[qid], statuses[qid]
        if status["status"] != "prepared":
            predictions[qid] = {"status": "technical_failure",
                                "reason": status.get("error", status["status"]),
                                "ranked_formulas": [], "scores": {}}
            continue
        per_formula = defaultdict(list)
        for sp in inp["hplus_spectra"]:
            spec = sp["spec"]
            vals = np.array([scores[(spec, form)] for form in inp["formulas"]],
                            dtype=np.float64)
            assert vals.shape == (len(inp["formulas"]),) and np.isfinite(vals).all()
            peak = float(vals.max())
            logz = peak + math.log(float(np.exp(vals - peak).sum()))
            for form, value in zip(inp["formulas"], vals - logz):
                per_formula[form].append(float(value))
        if (set(per_formula) != set(inp["formulas"]) or
                any(len(v) != len(inp["hplus_spectra"]) for v in per_formula.values())):
            raise ValueError(f"Incomplete native formula scores: {qid}")
        averaged = {form: sum(per_formula[form]) / len(inp["hplus_spectra"])
                    for form in inp["formulas"]}
        ranked = sorted(inp["formulas"], key=lambda f: (-averaged[f], f))
        predictions[qid] = {
            "status": "predicted", "ranked_formulas": ranked,
            "scores": averaged, "hplus_spectra": len(inp["hplus_spectra"]),
            "candidate_formulas": len(inp["formulas"]),
            "root_only_pairs": status["root_only_pairs"]}
    assert set(predictions) == set(protocol["qid_order"])
    result = {"status": "frozen_before_truth_evaluation",
              "protocol_sha256": sha(RUN / "protocol.json"),
              "inputs_sha256": sha(RUN / "inputs.json"),
              "checkpoint_sha256": sha(CHECKPOINT),
              "source_sha256": sha(__file__),
              "predictions": predictions, "sealed_scored": False}
    out = RUN / "predictions.json"
    if out.exists():
        raise FileExistsError(out)
    dump(out, result)
    print(json.dumps({"predictions_sha256": sha(out),
                      "n": len(predictions),
                      "predicted": sum(x["status"] == "predicted" for x in predictions.values()),
                      "technical_failures": sum(x["status"] != "predicted" for x in predictions.values())}))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["freeze", "prepare", "smoke", "predict"])
    p.add_argument("--qids", default="")
    args = p.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "prepare":
        prepare(args.qids.split(","))
    elif args.action == "smoke":
        smoke()
    else:
        predict()


if __name__ == "__main__":
    main()
