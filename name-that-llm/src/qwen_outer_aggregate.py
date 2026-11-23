"""CPU-only merge of three fold-specific Qwen outer predictions; never refit on OOF."""
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss


def extract_outer(manifest, frozen, adapted, official, prompt_folds, labels):
    """Reject incomplete or leaking folds, then return held-out predictions only."""
    fold=manifest.get("outer_fold")
    if (manifest.get("status")!="COMPLETE" or manifest.get("protocol")!="outer"
            or manifest.get("epochs_completed")!=1 or fold not in (0,1,2)
            or not manifest.get("checkpoint_reload",{}).get("verified_after_reload")):
        raise ValueError("Complete single-epoch outer run with verified reload required")
    ids=np.asarray(frozen["ids"],dtype=str)
    actual=np.asarray(frozen["actual"],dtype=str)
    labels=np.asarray(labels,dtype=str)
    fit_count=int(frozen["fit_count"])
    if len(ids)!=len(set(ids)) or not 0<fit_count<len(ids):
        raise ValueError("Unique fit and outer row IDs required")
    update=(manifest.get("epochs") or [{}])[0].get("training",{})
    if (manifest.get("max_epochs")!=1 or manifest.get("probe_reset_exact") is not True
            or update.get("rows")!=fit_count or update.get("gradient_seen") is not True
            or not isinstance(update.get("actual_steps"),int) or update["actual_steps"]<=0):
        raise ValueError("Outer fold lacks a complete train-only update after probe reset")
    for key in ("ids","actual","labels","fit_count"):
        if not np.array_equal(frozen[key],adapted[key]):
            raise ValueError(f"Frozen/adapted {key} mismatch")
    if not np.array_equal(frozen["labels"],labels):
        raise ValueError("Class order mismatch")
    if ids[:fit_count].tolist()!=manifest["split"]["fit_ids"] or ids[fit_count:].tolist()!=manifest["split"]["outer_ids"]:
        raise ValueError("Feature row order differs from the manifest split")
    prompts=[]
    for position,(identifier,truth) in enumerate(zip(ids,actual)):
        if identifier not in official or official[identifier][1]!=truth:
            raise ValueError("Official ID or label mismatch")
        prompt=official[identifier][0]
        if prompt not in prompt_folds or (prompt_folds[prompt]==fold)!=(position>=fit_count):
            raise ValueError("Prompt fold boundary mismatch")
        prompts.append(prompt)
    if set(prompts[:fit_count]) & set(prompts[fit_count:]):
        raise ValueError("Prompt appears in both fit and outer rows")
    expected_fit={identifier for identifier,(prompt,_) in official.items() if prompt_folds[prompt]!=fold}
    expected_outer=set(official)-expected_fit
    if set(ids[:fit_count])!=expected_fit or set(ids[fit_count:])!=expected_outer:
        raise ValueError("Fold fit or outer IDs do not match the complete official partition")
    probability=[]
    for payload in (frozen,adapted):
        value=np.asarray(payload["readout"],dtype=np.float64)
        if (value.shape!=(len(ids),len(labels)) or not np.isfinite(value).all()
                or np.any(value<0) or np.any(value>1)
                or not np.allclose(value.sum(1),1,atol=1e-6)):
            raise ValueError("Invalid fold probabilities")
        probability.append(value[fit_count:])
    return {"ids":ids[fit_count:],"actual":actual[fit_count:],
            "prompts":np.array(prompts[fit_count:]),
            "frozen":probability[0],"adapted":probability[1],
            "folds":np.full(len(ids)-fit_count,fold,dtype=int)}


def prompt_bootstrap(actual, prompts, frozen, adapted, labels, draws=10000, seed=20261025):
    """Paired cluster bootstrap of macro recall and NLL across official prompts."""
    labels=np.asarray(labels,dtype=str)
    prompts=np.asarray(prompts,dtype=str)
    index={label:i for i,label in enumerate(labels)}
    y=np.array([index[x] for x in actual],dtype=int)
    pids=np.unique(prompts)
    counts=np.zeros((len(pids),len(labels)),dtype=int)
    right0=np.zeros_like(counts);right1=np.zeros_like(counts)
    loss0=np.zeros(len(pids));loss1=np.zeros(len(pids))
    guess0=frozen.argmax(1);guess1=adapted.argmax(1)
    for pi,prompt in enumerate(pids):
        rows=np.flatnonzero(prompts==prompt)
        for c in range(len(labels)):
            at=rows[y[rows]==c]
            counts[pi,c]=len(at)
            right0[pi,c]=np.sum(guess0[at]==c)
            right1[pi,c]=np.sum(guess1[at]==c)
        loss0[pi]=-np.log(np.clip(frozen[rows,y[rows]],1e-15,1)).sum()
        loss1[pi]=-np.log(np.clip(adapted[rows,y[rows]],1e-15,1)).sum()
    rng=np.random.default_rng(seed)
    delta_ba=[];delta_nll=[]
    for _ in range(draws):
        sample=rng.integers(0,len(pids),size=len(pids))
        denominator=counts[sample].sum(0)
        if np.any(denominator==0):
            continue
        delta_ba.append(float(np.mean((right1[sample].sum(0)-right0[sample].sum(0))/denominator)))
        delta_nll.append(float((loss1[sample].sum()-loss0[sample].sum())/denominator.sum()))
    if len(delta_ba)<.99*draws:
        raise ValueError("Bootstrap lacks class coverage")
    return {"draws_requested":draws,"draws_valid":len(delta_ba),"seed":seed,
            "delta_BA_95pct":np.quantile(delta_ba,[.025,.975]).tolist(),
            "delta_NLL_95pct":np.quantile(delta_nll,[.025,.975]).tolist()}


def merge_qwen_outer(data_root, fold_runs, output, core):
    """Load only fold-held-out probabilities; assert every official row appears once."""
    started=time.monotonic()
    data_root=Path(data_root);output=Path(output)
    if len(fold_runs)!=3 or output.exists():
        raise ValueError("Exactly three fold runs and a new output directory required")
    train,_=core.read_data(data_root)
    train_hash=core.digest(data_root/"train.csv")
    if train_hash!="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707":
        raise ValueError("Official train data identity changed")
    labels=np.array(sorted(train.model.unique()),dtype=str)
    if len(labels)!=18 or len(train)!=2699 or train.prompt_id.nunique()!=150:
        raise ValueError("Unexpected official train shape")
    official=dict(zip(train.id.astype(str),zip(train.prompt_id.astype(str),train.model.astype(str))))
    if len(official)!=len(train):raise ValueError("Duplicate official IDs")
    prompt_folds=core.MODERNBERT_MIXED_FOLDS
    collected=[];identities=[];fold_summaries=[]
    for directory in map(Path,fold_runs):
        manifest_path=directory/"fold_manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("outer_evaluated") or manifest.get("candidate_promoted"):
            raise ValueError("Invalid outer evaluation or premature candidate promotion")
        with np.load(directory/"frozen_features.npz",allow_pickle=False) as frozen:
            with np.load(directory/"epoch1_features.npz",allow_pickle=False) as adapted:
                part=extract_outer(manifest,frozen,adapted,official,prompt_folds,labels)
        for name,probability in (("frozen",part["frozen"]),("adapted",part["adapted"])):
            score=float(balanced_accuracy_score(part["actual"],labels[probability.argmax(1)]))
            nll=float(log_loss(part["actual"],probability,labels=labels))
            recorded=manifest["frozen_baseline"]["outer"] if name=="frozen" else manifest["epochs"][0]["outer"]
            if abs(score-recorded["readout_BA"])>1e-12 or abs(nll-recorded["readout_NLL"])>1e-12:
                raise ValueError("Fold metric differs from saved probabilities")
        identities.append((manifest["source_sha256"],manifest["formal_baseline_sha256"],
                           manifest["model_config_sha256"],json.dumps(manifest["training"],sort_keys=True)))
        checkpoint_files={}
        for name in ("adapter_model.safetensors","aux_head.safetensors","adapter_config.json"):
            path=directory/"epoch1_adapter"/name
            if not path.is_file() or path.stat().st_size==0:
                raise ValueError(f"Missing or empty checkpoint file: {path}")
            digest=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):
                    digest.update(block)
            checkpoint_files[name]={"bytes":path.stat().st_size,"sha256":digest.hexdigest()}
        fold_summaries.append({"fold":manifest["outer_fold"],"outer_rows":len(part["ids"]),
                               "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                               "checkpoint_files":checkpoint_files,
                               "checkpoint_reload":manifest["checkpoint_reload"]})
        collected.append(part)
    if len(set(identities))!=1 or sorted(x["fold"] for x in fold_summaries)!=[0,1,2]:
        raise ValueError("Fold source, recipe or identity differs")
    adapter_hashes=[part["checkpoint_files"]["adapter_model.safetensors"]["sha256"]
                    for part in fold_summaries]
    if len(set(adapter_hashes))!=3:
        raise ValueError("Outer folds share an adapter checkpoint")
    concatenated={key:np.concatenate([part[key] for part in collected]) for key in collected[0]}
    ids=concatenated["ids"]
    if len(ids)!=len(train) or len(set(ids))!=len(train) or set(ids)!=set(official):
        raise ValueError("Held-out OOF rows do not cover official train exactly once")
    position={identifier:i for i,identifier in enumerate(ids)}
    order=np.array([position[identifier] for identifier in train.id.astype(str)])
    merged={key:value[order] for key,value in concatenated.items()}
    frozen=merged["frozen"];adapted=merged["adapted"];truth=merged["actual"]
    metrics={}
    for name,probability in (("frozen",frozen),("adapted",adapted)):
        metrics[name]={"BA":float(balanced_accuracy_score(truth,labels[probability.argmax(1)])),
                       "NLL":float(log_loss(truth,probability,labels=labels))}
    metrics["delta_BA"]=metrics["adapted"]["BA"]-metrics["frozen"]["BA"]
    metrics["delta_NLL"]=metrics["adapted"]["NLL"]-metrics["frozen"]["NLL"]
    metrics["prompt_bootstrap"]=prompt_bootstrap(truth,merged["prompts"],frozen,adapted,labels)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/"outer_oof_probabilities.npz",ids=merged["ids"],actual=truth,
                        prompts=merged["prompts"],folds=merged["folds"],labels=labels,
                        frozen=frozen,adapted=adapted)
    pd.DataFrame({"id":merged["ids"],"prompt_id":merged["prompts"],"outer_fold":merged["folds"],
        "actual":truth,"frozen_pred":labels[frozen.argmax(1)],
        "adapted_pred":labels[adapted.argmax(1)]}).to_csv(output/"outer_oof_predictions.csv",index=False)
    report={"status":"COMPLETE_DEVELOPMENT_OOF","source":"official train only",
        "model_scope":"Qwen8B two-view standalone readout; v8 five-view/context fusion not evaluated",
        "candidate_promoted":False,
        "train_sha256":train_hash,"rows":len(train),"prompts":train.prompt_id.nunique(),
        "folds":fold_summaries,"shared_source_and_recipe":identities[0],"metrics":metrics,
        "server":socket.gethostname(),"environment":platform.python_version(),
        "seconds":time.monotonic()-started,
        "limitation":"Reused development prompts; this is not blind evidence or leaderboard score."}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
