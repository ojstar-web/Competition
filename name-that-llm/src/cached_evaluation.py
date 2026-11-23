"""CPU-only full-row, prompt-isolated evaluation of existing frozen features."""
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import (SEED, blend_view_predictions, centered_rows, chunk_prompt_groups,
                      digest, infer_prompt_groups, modernbert_fold_plan, pipeline,
                      read_data, softmax_scores)

PLAN_SEED = 20261025
SCENARIOS = (1, 2, 3, 6, 18)


def context_batches(ids, prompts, size, seed=PLAN_SEED):
    """Use one disjoint chunk per true prompt in each batch; no row is dropped."""
    ids, prompts = np.asarray(ids, dtype=str), np.asarray(prompts, dtype=str)
    if len(ids) != len(prompts) or len(set(ids)) != len(ids) or size < 1:
        raise ValueError("Unique IDs, aligned prompts and positive size required")
    chunks = []
    for prompt in np.unique(prompts):
        rows = np.flatnonzero(prompts == prompt).tolist()
        rows.sort(key=lambda i: (hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest(), ids[i]))
        chunks.append([np.asarray(rows[i:i+size], dtype=int) for i in range(0, len(rows), size)])
    return [np.concatenate([parts[b] for parts in chunks if b < len(parts)])
            for b in range(max((len(parts) for parts in chunks), default=0))]


def _shrunk(covariance, alpha):
    d = covariance.shape[0]
    scale = max(float(np.trace(covariance) / d), 1e-12)
    return (1-alpha)*covariance + alpha*scale*np.eye(d)


def _positive_covariance(covariance):
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(np.abs(values)))*1e-7, 1e-10)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def fit_gaussian(z, y, prompts, classes):
    """Fit author means plus a zero-mean shared prompt effect on training rows only."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=int)
    _, inv = np.unique(prompts, return_inverse=True)
    counts = np.bincount(inv)
    n, d = z.shape
    if len(y) != n or len(inv) != n or set(y) != set(range(classes)):
        raise ValueError("Aligned training rows and every class required")
    effects = np.zeros((len(counts), d))
    means = np.zeros((classes, d))
    for _ in range(4):
        residual = z - effects[inv]
        means = np.stack([residual[y == c].mean(0) for c in range(classes)])
        residual = z - means[y]
        effects.fill(0)
        np.add.at(effects, inv, residual)
        effects /= counts[:, None]
        shift = effects.mean(0)
        effects -= shift
        means += shift
    error = z - means[y] - effects[inv]
    dof = max(n-len(counts)-classes+1, 1)
    noise_raw = error.T @ error / dof
    noise_alpha = float(ledoit_wolf_shrinkage(error, assume_centered=True))
    noise = _shrunk(noise_raw, noise_alpha)
    prompt_raw = effects.T @ effects / max(len(counts)-1, 1) - np.mean(1/counts)*noise_raw
    prompt_alpha = float(ledoit_wolf_shrinkage(effects, assume_centered=True))
    prompt_shrunk = _shrunk(prompt_raw, prompt_alpha)
    prompt_eigenvalues = np.linalg.eigvalsh(prompt_shrunk)
    prompt_covariance = _positive_covariance(prompt_shrunk)
    covariance = _positive_covariance(noise + prompt_covariance)
    return {"means": means, "covariance": covariance,
            "precision": np.linalg.inv(covariance), "noise_shrinkage": noise_alpha,
            "prompt_shrinkage": prompt_alpha,
            "prompt_preclip_min_eigenvalue": float(prompt_eigenvalues.min()),
            "prompt_nonpositive_eigenvalues": int(np.sum(prompt_eigenvalues <= 0)),
            "dimensions": d}


def gaussian_probabilities(z, params):
    """Marginalise the unobserved prompt; equal author priors, no count constraints."""
    delta = np.asarray(z, dtype=np.float64)[:, None, :] - params["means"][None]
    scores = -.5*np.einsum("ncd,de,nce->nc", delta, params["precision"], delta, optimize=True)
    result = softmax(scores, axis=1)
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1)
    return result


def paired_prompt_delta(actual, pred, baseline, prompts, labels, draws=2000):
    _, inverse = np.unique(prompts, return_inverse=True)
    class_ids = np.searchsorted(labels, actual)
    counts = np.zeros((inverse.max()+1, len(labels)), dtype=int)
    delta = np.zeros_like(counts)
    np.add.at(counts, (inverse, class_ids), 1)
    np.add.at(delta, (inverse, class_ids),
              (pred == actual).astype(int) - (baseline == actual).astype(int))
    rng = np.random.default_rng(PLAN_SEED)
    values = []
    for _ in range(draws):
        take = rng.integers(0, len(counts), len(counts))
        denominator = counts[take].sum(0)
        if np.all(denominator > 0):
            values.append(float((delta[take].sum(0)/denominator).mean()))
    if len(values) < .95*draws:
        raise ValueError("Insufficient bootstrap draws containing every label")
    return {"delta95": np.quantile(values, [.025, .975]).tolist(), "valid_draws": len(values)}


def v8_fusion_triplet(old_plain, old_center, frozen_plain, frozen_center,
                      adapted_plain, adapted_center, style, groups):
    """Compare legacy v8 with matched frozen/adapted Qwen8B view replacements."""
    old_plain=np.asarray(old_plain)
    old_center=np.asarray(old_center)
    style=np.asarray(style)
    groups=np.asarray(groups)
    if (old_plain.ndim!=3 or old_plain.shape[1]!=5 or old_center.shape!=old_plain.shape
            or style.shape!=(len(old_plain),old_plain.shape[2]) or len(groups)!=len(old_plain)):
        raise ValueError("Expected aligned five-view v8 probabilities, style and groups")
    new_views=[]
    for value in (frozen_plain,frozen_center,adapted_plain,adapted_center):
        value=np.asarray(value)
        if value.shape!=(len(old_plain),2,old_plain.shape[2]) or not np.isfinite(value).all():
            raise ValueError("Expected two aligned Qwen8B probability views")
        new_views.append(value)
    def fuse(plain,center):
        return .8*blend_view_predictions(plain,center,groups,"adaptive","group_middle")+.2*style
    result={"legacy_v8":fuse(old_plain,old_center)}
    for name,first,last in (("matched_frozen",new_views[0],new_views[1]),
                            ("adapted",new_views[2],new_views[3])):
        plain=old_plain.copy();plain[:,3:]=first
        center=old_center.copy();center[:,3:]=last
        result[name]=fuse(plain,center)
    if not all(np.isfinite(value).all() and np.allclose(value.sum(1),1) for value in result.values()):
        raise ValueError("Invalid v8 fusion probabilities")
    return result


def checked_context_plan(entries, ids, prompts, texts, folds, sizes=SCENARIOS):
    """Recreate the saved context batches and unlabelled groups exactly."""
    ids=np.asarray(ids,dtype=str)
    prompts=np.asarray(prompts,dtype=str)
    texts=np.asarray(texts,dtype=object)
    folds=np.asarray(folds,dtype=int)
    if (len(ids)!=len(set(ids)) or len(prompts)!=len(ids) or len(texts)!=len(ids)
            or len(folds)!=len(ids) or set(folds)!={0,1,2}):
        raise ValueError("Invalid official row identity or fold mapping")
    saved={}
    for entry in entries:
        key=(entry["size"],entry["fold"],entry["batch"])
        if key in saved:
            raise ValueError("Duplicate context batch")
        saved[key]=entry
    used=set()
    result={}
    for size in sizes:
        plan=[]
        assigned=np.zeros(len(ids),dtype=int)
        for fold in range(3):
            valid=np.flatnonzero(folds==fold)
            for batch,take in enumerate(context_batches(ids[valid],prompts[valid],size)):
                rows=valid[take]
                key=(size,fold,batch)
                if key not in saved or saved[key]["ids"]!=ids[rows].tolist():
                    raise ValueError("Saved context row IDs differ from fixed plan")
                groups=infer_prompt_groups(texts[rows])
                if saved[key]["inferred_groups"]!=groups.tolist():
                    raise ValueError("Saved unlabelled context groups differ")
                used.add(key)
                assigned[rows]+=1
                plan.append((rows,groups))
        if not np.all(assigned==1):
            raise ValueError("Every official row must occur once per context size")
        result[size]=plan
    if used!=set(saved):
        raise ValueError("Unexpected or missing saved context batches")
    return result


def _report_predictions(actual, labels, folds, prompts, probabilities):
    base = labels[probabilities["v8"].argmax(1)]
    base_score = balanced_accuracy_score(actual, base)
    plain_pred = labels[probabilities["plain"].argmax(1)]
    plain_score = balanced_accuracy_score(actual, plain_pred)
    result = {}
    for name, prob in probabilities.items():
        assert prob.shape == (len(actual), len(labels))
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1)
        pred = labels[prob.argmax(1)]
        result[name] = {
            "BA": float(balanced_accuracy_score(actual, pred)),
            "delta_vs_v8": float(balanced_accuracy_score(actual, pred)-base_score),
            "delta_vs_plain": float(balanced_accuracy_score(actual, pred)-plain_score),
            "NLL": float(log_loss(actual, prob, labels=labels)),
            "fold_BA": [float(balanced_accuracy_score(actual[folds == f], pred[folds == f])) for f in range(3)],
            "class_recall": dict(zip(labels.tolist(), recall_score(actual, pred, labels=labels, average=None).tolist())),
            "rescued": int(np.sum((pred == actual) & (base != actual))),
            "harmed": int(np.sum((pred != actual) & (base == actual))),
            "paired_prompt": paired_prompt_delta(actual, pred, base, prompts, labels),
            "paired_prompt_vs_plain": paired_prompt_delta(actual, pred, plain_pred, prompts, labels),
        }
    return result


def readout_compatibility(rebuilt, saved, fold, arm):
    """Bound cross-sklearn refit drift while recording any prediction changes."""
    rebuilt=np.asarray(rebuilt,dtype=np.float64)
    saved=np.asarray(saved,dtype=np.float64)
    if rebuilt.shape!=saved.shape or rebuilt.ndim!=2 or not np.isfinite(rebuilt).all() or not np.isfinite(saved).all():
        raise ValueError("Invalid Qwen readout comparison")
    difference=np.abs(rebuilt-saved)
    result={"fold":fold,"arm":arm,"rows":len(rebuilt),
            "max_abs_probability_diff":float(difference.max()),
            "mean_abs_probability_diff":float(difference.mean()),
            "argmax_flips":int(np.sum(rebuilt.argmax(1)!=saved.argmax(1))),
            "max_limit":.01,"mean_limit":.001,"flip_fraction_limit":.01}
    if (result["max_abs_probability_diff"]>=result["max_limit"]
            or result["mean_abs_probability_diff"]>=result["mean_limit"]
            or result["argmax_flips"]>np.ceil(result["flip_fraction_limit"]*len(rebuilt))):
        raise ValueError(f"Qwen fold readout reconstruction exceeds cross-runtime bounds: {result}")
    return result


def evaluate_qwen_fusion(root, cache_path, reference_dir, fold_runs, out):
    """CPU-only, three-arm v8 comparison after three complete Qwen outer runs."""
    import baseline as core
    from qwen_outer_aggregate import merge_qwen_outer, prompt_bootstrap

    root,cache_path,reference_dir,out=map(Path,(root,cache_path,reference_dir,out))
    if len(fold_runs)!=3 or out.exists():
        raise ValueError("Three fold directories and a new output directory required")
    train,_=core.read_data(root)
    ids=train.id.to_numpy(dtype=str)
    actual=train.model.to_numpy(dtype=str)
    prompts=train.prompt_id.to_numpy(dtype=str)
    texts=train.text.to_numpy(dtype=object)
    labels=np.array(sorted(train.model.unique()),dtype=str)
    folds=np.array([core.MODERNBERT_MIXED_FOLDS[p] for p in prompts])
    train_hash=core.digest(root/"train.csv")
    if (train_hash!="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
            or len(ids)!=2699 or len(set(ids))!=2699 or len(labels)!=18):
        raise ValueError("Official train identity or shape changed")
    manifest=json.loads(cache_path.with_name("frozen_feature_manifest.json").read_text())
    if manifest["train_sha256"]!=train_hash:
        raise ValueError("Legacy feature cache belongs to another train set")
    with np.load(cache_path,allow_pickle=False) as cache:
        old_parts=[cache[f"part_{view}"] for view in range(5)]
        if not np.array_equal(cache["ids"],ids):
            raise ValueError("Legacy feature row order differs")
    if any(x.shape[0]!=len(ids) or not np.isfinite(x).all() for x in old_parts):
        raise ValueError("Legacy feature cache has invalid values")
    with np.load(reference_dir/"single_text_oof.npz",allow_pickle=False) as ref:
        for key,expected in (("ids",ids),("actual",actual),("labels",labels),
                             ("folds",folds),("prompts",prompts)):
            if not np.array_equal(ref[key],expected):
                raise ValueError(f"Legacy single-text {key} differs")
        old_plain=ref["plain_views"]
        style=ref["style"]
    if (old_plain.shape!=(len(ids),5,len(labels)) or style.shape!=(len(ids),len(labels))
            or not np.isfinite(old_plain).all() or not np.isfinite(style).all()):
        raise ValueError("Invalid legacy probabilities")
    plan_file=reference_dir/"context_plan.json"
    plans=checked_context_plan(json.loads(plan_file.read_text()),ids,prompts,texts,folds)

    out.mkdir(parents=True,exist_ok=False)
    standalone=merge_qwen_outer(root,fold_runs,out/"standalone",core)
    runs={}
    for directory in map(Path,fold_runs):
        info=json.loads((directory/"fold_manifest.json").read_text())
        runs[info["outer_fold"]]=directory
    if set(runs)!={0,1,2}:
        raise ValueError("Three distinct Qwen folds required")
    old_heads={};new_heads={};valid_features={};readout_gates=[]
    new_plain={name:np.empty((len(ids),2,len(labels))) for name in ("matched_frozen","adapted")}
    for fold in range(3):
        fit=np.flatnonzero(folds!=fold)
        valid=np.flatnonzero(folds==fold)
        old_heads[fold]=[]
        for x in old_parts:
            center=make_pipeline(StandardScaler(),LogisticRegression(C=.0025,max_iter=2000,random_state=SEED))
            xc=[centered_rows(x[fit],chunk_prompt_groups(prompts[fit],k,SEED+k)) for k in (2,4,18)]
            center.fit(np.vstack(xc),np.tile(actual[fit],3),
                       logisticregression__sample_weight=np.ones(len(fit)*3))
            if not np.array_equal(center.classes_,labels):
                raise ValueError("Legacy center class order differs")
            old_heads[fold].append(center)
        new_heads[fold]={};valid_features[fold]={}
        for arm,filename in (("matched_frozen","frozen_features.npz"),
                             ("adapted","epoch1_features.npz")):
            with np.load(runs[fold]/filename,allow_pickle=False) as data:
                fold_ids=np.asarray(data["ids"],dtype=str)
                saved_prob=np.asarray(data["readout"])
                layers=[np.asarray(data[key],dtype=np.float32) for key in ("layer7","layer22")]
            position=pd.Index(ids).get_indexer(fold_ids)
            if (len(fold_ids)!=len(ids) or len(set(fold_ids))!=len(ids)
                    or np.any(position<0) or saved_prob.shape!=(len(ids),len(labels))
                    or any(x.shape!=(len(ids),4096) or not np.isfinite(x).all() for x in layers)):
                raise ValueError("Qwen fold feature identities or shape differ")
            new_heads[fold][arm]=[];valid_features[fold][arm]=[]
            for view,x_fold in enumerate(layers):
                x=np.empty_like(x_fold);x[position]=x_fold
                ordinary=make_pipeline(StandardScaler(),LogisticRegression(C=.0025,max_iter=2000,random_state=SEED))
                ordinary.fit(x[fit],actual[fit])
                center=make_pipeline(StandardScaler(),LogisticRegression(C=.0025,max_iter=2000,random_state=SEED))
                xc=[centered_rows(x[fit],chunk_prompt_groups(prompts[fit],k,SEED+k)) for k in (2,4,18)]
                center.fit(np.vstack(xc),np.tile(actual[fit],3),
                           logisticregression__sample_weight=np.ones(len(fit)*3))
                if not (np.array_equal(ordinary.classes_,labels) and np.array_equal(center.classes_,labels)):
                    raise ValueError("Qwen head class order differs")
                new_plain[arm][valid,view]=ordinary.predict_proba(x[valid])
                valid_features[fold][arm].append(x[valid])
                new_heads[fold][arm].append(center)
            rebuilt=new_plain[arm][valid].mean(axis=1)
            saved_at=np.empty(len(ids),dtype=int)
            saved_at[position]=np.arange(len(ids))
            original=saved_prob[saved_at[valid]]
            readout_gates.append(readout_compatibility(rebuilt,original,fold,arm))

    def metrics(probabilities):
        base=labels[probabilities["legacy_v8"].argmax(1)]
        frozen=labels[probabilities["matched_frozen"].argmax(1)]
        result={}
        for name,prob in probabilities.items():
            pred=labels[prob.argmax(1)]
            result[name]={"BA":float(balanced_accuracy_score(actual,pred)),
                          "NLL":float(log_loss(actual,prob,labels=labels)),
                          "fold_BA":[float(balanced_accuracy_score(actual[folds==f],pred[folds==f])) for f in range(3)],
                          "class_recall":dict(zip(labels.tolist(),recall_score(actual,pred,labels=labels,average=None).tolist())),
                          "rescued_vs_legacy":int(np.sum((pred==actual)&(base!=actual))),
                          "harmed_vs_legacy":int(np.sum((pred!=actual)&(base==actual))),
                          "paired_prompt_vs_legacy":paired_prompt_delta(actual,pred,base,prompts,labels)}
        result["adapted_vs_matched_frozen"]={
            "delta_BA":result["adapted"]["BA"]-result["matched_frozen"]["BA"],
            "delta_NLL":result["adapted"]["NLL"]-result["matched_frozen"]["NLL"],
            "rescued":int(np.sum((labels[probabilities["adapted"].argmax(1)]==actual)&(frozen!=actual))),
            "harmed":int(np.sum((labels[probabilities["adapted"].argmax(1)]!=actual)&(frozen==actual))),
            "prompt_bootstrap":prompt_bootstrap(actual,prompts,probabilities["matched_frozen"],
                                                 probabilities["adapted"],labels)}
        return result

    singleton=np.arange(len(ids))
    single=v8_fusion_triplet(old_plain,old_plain,
                             new_plain["matched_frozen"],new_plain["matched_frozen"],
                             new_plain["adapted"],new_plain["adapted"],style,singleton)
    plain_reference=.8*old_plain.mean(axis=1)+.2*style
    single_diff=float(np.max(np.abs(single["legacy_v8"]-plain_reference)))
    if single_diff>=1e-12:
        raise ValueError("Single-text legacy v8 reconstruction failed")
    summaries={"single_text":metrics(single)}
    reference_gates={"single_text":{"max_abs_probability_diff":single_diff,"rows":len(ids)}}
    np.savez_compressed(out/"single_text_fusion_oof.npz",ids=ids,actual=actual,
                        labels=labels,folds=folds,prompts=prompts,**single)
    for size,plan in plans.items():
        probabilities={name:np.empty((len(ids),len(labels))) for name in
                       ("legacy_v8","matched_frozen","adapted")}
        assigned=np.zeros(len(ids),dtype=int)
        for rows,groups in plan:
            fold=int(folds[rows[0]])
            if not np.all(folds[rows]==fold):
                raise ValueError("Context batch crossed outer folds")
            old_center=np.stack([head.predict_proba(centered_rows(old_parts[view][rows],groups))
                                 for view,head in enumerate(old_heads[fold])],axis=1)
            new_center={}
            local=np.searchsorted(np.flatnonzero(folds==fold),rows)
            for arm in ("matched_frozen","adapted"):
                new_center[arm]=np.stack([
                    head.predict_proba(centered_rows(valid_features[fold][arm][view][local],groups))
                    for view,head in enumerate(new_heads[fold][arm])],axis=1)
            triplet=v8_fusion_triplet(old_plain[rows],old_center,
                                      new_plain["matched_frozen"][rows],new_center["matched_frozen"],
                                      new_plain["adapted"][rows],new_center["adapted"],style[rows],groups)
            for name,prob in triplet.items():probabilities[name][rows]=prob
            assigned[rows]+=1
        if not np.all(assigned==1):raise ValueError("Incomplete context coverage")
        with np.load(reference_dir/f"context_{size}_oof.npz",allow_pickle=False) as reference:
            if not np.array_equal(reference["ids"],ids):
                raise ValueError("Legacy context row order differs")
            diff=float(np.max(np.abs(probabilities["legacy_v8"]-reference["v8"])))
        if diff>=1e-9:
            raise ValueError(f"Legacy v8 probability reconstruction failed at context {size}: {diff}")
        reference_gates[str(size)]={"max_abs_probability_diff":diff,"rows":len(ids)}
        summaries[str(size)]=metrics(probabilities)
        np.savez_compressed(out/f"context_{size}_fusion_oof.npz",ids=ids,actual=actual,
                            labels=labels,folds=folds,prompts=prompts,**probabilities)
    report={"status":"COMPLETE_DEVELOPMENT_FUSION_OOF","candidate_promoted":False,
            "model_scope":"v8 five-view/context fusion; only Qwen8B last two views replaced",
            "train_sha256":train_hash,"rows":len(ids),"prompts":len(set(prompts)),
            "scenarios":["single_text",*plans],"reference_gates":reference_gates,
            "readout_reconstruction":readout_gates,"local_sklearn":sklearn.__version__,
            "fold_sklearn":[json.loads((runs[i]/"fold_manifest.json").read_text()).get("libraries",{}).get("scikit-learn") for i in range(3)],
            "metrics":summaries,
            "standalone_summary":standalone["status"],
            "sources":{"cached_evaluation_sha256":core.digest(Path(__file__)),
                       "baseline_sha256":core.digest(Path(core.__file__)),
                       "cache_sha256":core.digest(cache_path),"context_plan_sha256":core.digest(plan_file)},
            "fold_runs":[str(runs[i]) for i in range(3)],
            "limitation":"Reused development prompts; no blind or public leaderboard result."}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def evaluate_cached(root, cache_path, reference_dir, out):
    """Fixed protocol: baseline reconstruction, full-row contexts, two new CPU heads."""
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train, _ = read_data(root)
    ids = train.id.to_numpy(dtype=str)
    actual = train.model.to_numpy(dtype=str)
    texts = train.text.to_numpy(dtype=object)
    prompts = train.prompt_id.to_numpy(dtype=str)
    labels = np.array(sorted(train.model.unique()))
    y = np.searchsorted(labels, actual)
    manifest = json.loads(cache_path.with_name("frozen_feature_manifest.json").read_text())
    assert manifest["train_sha256"] == digest(root/"train.csv")
    with np.load(cache_path, allow_pickle=False) as cache:
        assert np.array_equal(cache["ids"], ids)
        parts = [cache[k] for k in sorted(cache.files) if k.startswith("part_")]
    assert len(parts) == 5 and all(len(x) == len(train) and np.isfinite(x).all() for x in parts)
    meta = json.loads((reference_dir/"partial_validation.json").read_text())
    _, folds = modernbert_fold_plan(train, meta, meta)
    second_dir = reference_dir.parents[1]/"night_mixed_confirm_20261022_24"/"run"
    references = []
    for source in (reference_dir, second_dir):
        metadata = json.loads((source/"partial_validation.json").read_text())
        modernbert_fold_plan(train, meta, metadata)
        with np.load(source/"partial_0_oof.npz", allow_pickle=False) as data:
            reference = {key: data[key] for key in data.files}
        assert np.array_equal(reference["labels"], labels)
        assert len(np.unique(reference["ids"])) == len(reference["ids"])
        rows = pd.Index(ids).get_indexer(reference["ids"])
        assert (rows >= 0).all() and np.array_equal(reference["actual"], actual[rows])
        references.append((source, metadata, reference, rows))
    config = {"mode": "evaluate-cached", "seed": PLAN_SEED, "dimensions": 128,
              "scenarios": list(SCENARIOS), "class_prior": "uniform",
              "gaussian": "singleton author+prompt random effect; no injective assignment",
              "PCA": "train-only within-prompt residual; randomized SVD",
              "reference_dir": str(reference_dir), "cache": str(cache_path),
              "train_sha256": manifest["train_sha256"], "feature_manifest": manifest,
              "fold_mapping": meta["validation_folds"], "fold_seed": meta["fold_seed"],
              "source_sha256": digest(Path(__file__)), "baseline_sha256": digest(Path(__file__).with_name("baseline.py")),
              "python": sys.version, "sklearn": sklearn.__version__, "numpy": np.__version__,
              "hostname": socket.gethostname(), "platform": platform.platform(),
              "cpu_threads": 2, "wall_timeout_seconds": 1800,
              "note": "Reused development prompts; paired intervals conditional on frozen predictions; not blind/selection-adjusted."}
    (out/"effective_config.json").write_text(json.dumps(config, indent=2))
    plain = np.empty((len(train), 5, len(labels)))
    style = np.empty((len(train), len(labels)))
    heads = {}
    gate_diffs = []
    rebuilt = {metadata["sampling_seed"]:
               {"plain": np.empty_like(reference["view_plain"]),
                "center": np.empty_like(reference["view_centered"])}
               for _, metadata, reference, _ in references}
    for fold in range(3):
        fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(prompts[fit]) & set(prompts[valid])
        style_head = pipeline("style").fit(texts[fit], actual[fit])
        assert np.array_equal(style_head.classes_, labels)
        style[valid] = softmax_scores(style_head.decision_function(texts[valid]))
        del style_head
        heads[fold] = []
        for view, x in enumerate(parts):
            ordinary = make_pipeline(StandardScaler(), LogisticRegression(C=.0025, max_iter=2000, random_state=SEED))
            ordinary.fit(x[fit], actual[fit])
            center = make_pipeline(StandardScaler(), LogisticRegression(C=.0025, max_iter=2000, random_state=SEED))
            xc = [centered_rows(x[fit], chunk_prompt_groups(prompts[fit], k, SEED+k)) for k in (2, 4, 18)]
            center.fit(np.vstack(xc), np.tile(actual[fit], 3),
                       logisticregression__sample_weight=np.ones(len(fit)*3))
            del xc
            assert np.array_equal(ordinary.classes_, labels) and np.array_equal(center.classes_, labels)
            plain[valid, view] = ordinary.predict_proba(x[valid])
            heads[fold].append(center)
            for source, metadata, reference, rows in references:
                take = np.flatnonzero(folds[rows] == fold)
                selected = rows[take]
                groups = infer_prompt_groups(texts[selected])
                centered = center.predict_proba(centered_rows(x[selected], groups))
                # Match cached_analysis: invalid-size groups must use the ordinary head.
                _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
                eligible = (counts[inverse] >= 2) & (counts[inverse] <= 18)
                centered[~eligible] = plain[selected, view][~eligible]
                diff = max(float(np.max(np.abs(plain[selected, view]-reference["view_plain"][take, view]))),
                           float(np.max(np.abs(centered-reference["view_centered"][take, view]))))
                rebuilt[metadata["sampling_seed"]]["plain"][take, view] = plain[selected, view]
                rebuilt[metadata["sampling_seed"]]["center"][take, view] = centered
                gate_diffs.append({"seed": metadata["sampling_seed"], "fold": fold, "view": view, "max_abs_diff": diff})
                assert diff < 1e-9, ("Legacy head reconstruction failed", source, fold, view, diff)
            print(f"baseline fold={fold} view={view} elapsed={time.monotonic()-started:.1f}", flush=True)
    legacy = []
    for source, metadata, reference, rows in references:
        delta = float(np.max(np.abs(style[rows]-reference["style"])))
        assert delta < 1e-9, ("Style reconstruction", delta)
        groups = np.empty(len(rows), dtype=int)
        for fold in range(3):
            take = np.flatnonzero(folds[rows] == fold)
            groups[take] = infer_prompt_groups(texts[rows[take]]) + fold*100000
        probability = .8*blend_view_predictions(reference["view_plain"], reference["view_centered"], groups, "adaptive", "group_middle")+.2*reference["style"]
        reconstruction = .8*blend_view_predictions(
            rebuilt[metadata["sampling_seed"]]["plain"], rebuilt[metadata["sampling_seed"]]["center"],
            groups, "adaptive", "group_middle")+.2*style[rows]
        assert np.array_equal(reconstruction.argmax(1), probability.argmax(1))
        max_probability_diff = float(np.max(np.abs(reconstruction-probability)))
        assert max_probability_diff < 1e-9
        ba = float(balanced_accuracy_score(actual[rows], labels[reconstruction.argmax(1)]))
        np.savez_compressed(out/f"legacy_reconstruction_{metadata['sampling_seed']}.npz",
                            ids=ids[rows], labels=labels, actual=actual[rows],
                            reconstructed=reconstruction, reference=probability)
        expected = {20261024: .470500777568342, 20261025: .5252359805569528}[metadata["sampling_seed"]]
        assert abs(ba-expected) < 1e-12
        legacy.append({"seed": metadata["sampling_seed"], "BA": ba, "style_max_abs_diff": delta,
                       "max_probability_diff": max_probability_diff, "argmax_identical": True})
    plain_ba = float(balanced_accuracy_score(actual, labels[(.8*plain.mean(1)+.2*style).argmax(1)]))
    assert abs(plain_ba-.43597564006959977) < 1e-12
    (out/"baseline_reconstruction.json").write_text(json.dumps(
        {"status": "PASS", "heads": gate_diffs, "legacy": legacy, "full_plain_BA": plain_ba}, indent=2))
    print("BASELINE_RECONSTRUCTION_PASS", json.dumps(legacy), flush=True)

    candidates = {name: np.empty_like(plain) for name in ("lda", "gaussian")}
    fitted = []
    for fold in range(3):
        fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        for view, x in enumerate(parts):
            scaler = StandardScaler().fit(x[fit])
            xs = scaler.transform(x[fit])
            # Only training prompt IDs define this residual PCA.
            residual = xs.copy()
            for prompt in np.unique(prompts[fit]):
                take = np.flatnonzero(prompts[fit] == prompt)
                residual[take] -= residual[take].mean(0)
            pca = PCA(n_components=128, svd_solver="randomized", random_state=PLAN_SEED).fit(residual)
            zfit, zvalid = pca.transform(xs), pca.transform(scaler.transform(x[valid]))
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(len(labels), 1/len(labels)))
            lda.fit(zfit, y[fit])
            assert np.array_equal(lda.classes_, np.arange(len(labels)))
            params = fit_gaussian(zfit, y[fit], prompts[fit], len(labels))
            candidates["lda"][valid, view] = lda.predict_proba(zvalid)
            candidates["gaussian"][valid, view] = gaussian_probabilities(zvalid, params)
            fitted.append({"fold": fold, "view": view, "noise_shrinkage": params["noise_shrinkage"],
                           "prompt_shrinkage": params["prompt_shrinkage"],
                           "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(params["covariance"]).min()),
                           "prompt_preclip_min_eigenvalue": params["prompt_preclip_min_eigenvalue"],
                           "prompt_nonpositive_eigenvalues": params["prompt_nonpositive_eigenvalues"],
                           "dimensions": params["dimensions"]})
            print(f"candidate fold={fold} view={view} elapsed={time.monotonic()-started:.1f}", flush=True)
    probabilities = {"plain": .8*plain.mean(1)+.2*style}
    probabilities.update({name: .8*prob.mean(1)+.2*style for name, prob in candidates.items()})
    np.savez_compressed(out/"single_text_oof.npz", ids=ids, actual=actual, labels=labels, folds=folds,
                        prompts=prompts, plain_views=plain, style=style,
                        lda_views=candidates["lda"], gaussian_views=candidates["gaussian"])

    summaries, plans = {}, []
    for size in SCENARIOS:
        v8 = np.empty_like(style)
        assigned = np.zeros(len(train), dtype=int)
        group_ids = np.full(len(train), -1)
        for fold in range(3):
            valid = np.flatnonzero(folds == fold)
            for batch, take in enumerate(context_batches(ids[valid], prompts[valid], size)):
                rows = valid[take]
                groups = infer_prompt_groups(texts[rows])
                center_views = np.stack([head.predict_proba(centered_rows(parts[v][rows], groups))
                                         for v, head in enumerate(heads[fold])], axis=1)
                v8[rows] = .8*blend_view_predictions(plain[rows], center_views, groups, "adaptive", "group_middle")+.2*style[rows]
                assigned[rows] += 1
                group_ids[rows] = fold*1000000 + batch*10000 + groups
                plans.append({"size": size, "fold": fold, "batch": batch, "ids": ids[rows].tolist(),
                              "inferred_groups": groups.tolist()})
        assert np.all(assigned == 1) and np.all(group_ids >= 0)
        current = dict(probabilities, v8=v8)
        summaries[str(size)] = _report_predictions(actual, labels, folds, prompts, current)
        summaries[str(size)]["context"] = {"rows": len(train), "unique_prompts": len(set(prompts)),
                                          "each_row_exposures": [int(assigned.min()), int(assigned.max())]}
        np.savez_compressed(out/f"context_{size}_oof.npz", ids=ids, actual=actual, labels=labels,
                            folds=folds, prompts=prompts, inferred_groups=group_ids, **current)
        print(f"context={size} BA="+json.dumps({k:v["BA"] for k,v in summaries[str(size)].items() if k != "context"}), flush=True)
    (out/"context_plan.json").write_text(json.dumps(plans))
    (out/"gaussian_fit_diagnostics.json").write_text(json.dumps(fitted, indent=2))
    result = {"status": "COMPLETE", "seconds": time.monotonic()-started, "scenarios": summaries,
              "note": config["note"], "candidate_promotion": False, "production_profile_changed": False}
    (out/"summary.json").write_text(json.dumps(result, indent=2))
    print("EVALUATION_COMPLETE", json.dumps({"seconds": result["seconds"]}), flush=True)
