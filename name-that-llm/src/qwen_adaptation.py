"""Matched Qwen adapter training; called only through baseline's explicit GPU modes."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np


def pooled_views(hidden_states, input_ids, attention_mask, special_ids, layers=(7, 22)):
    mask = attention_mask.float()
    for token in special_ids:
        mask = mask * input_ids.ne(token)
    mask = mask.unsqueeze(-1)
    return tuple((hidden_states[layer].float()*mask).sum(1)/mask.sum(1).clamp(min=1) for layer in layers)


def snapshot_trainable(model, head):
    return {"model": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
            "head": {name: p.detach().cpu().clone() for name, p in head.state_dict().items()}}


def restore_trainable(model, head, snapshot):
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot["model"].items():
            params[name].copy_(value.to(params[name].device))
    head.load_state_dict(snapshot["head"])


def can_start_epoch(elapsed, training_bound, evaluation_bound, budget, save_margin=90):
    return elapsed + training_bound + evaluation_bound + save_margin < budget


def register_plain_lora(config):
    """Select PEFT's native Linear for verified nonquantized targets."""
    import torch
    from peft.tuners.lora.layer import Linear
    config._register_custom_module({torch.nn.Linear: Linear})
    return config


def run_bounded_process(command, cwd, environment, logfile, seconds):
    """POSIX supervisor: native CUDA work runs only in the killable child group."""
    import subprocess
    if os.name != "posix" or seconds <= 0:
        raise ValueError("Positive POSIX process budget required")
    started = time.monotonic()
    with Path(logfile).open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            exit_code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            exit_code = process.wait(timeout=5)
    return {"status": "TIMEOUT" if timed_out else "COMPLETE" if exit_code == 0 else "FAIL",
            "exit_code": exit_code, "pid": process.pid, "seconds": time.monotonic()-started,
            "hard_wait_seconds": seconds, "process_group_kill_requested": timed_out}


def truncate_qwen_config(config):
    """Apply the exact 22-layer backbone shape used by training and reload."""
    if config.num_hidden_layers < 22:
        raise ValueError("Qwen backbone has fewer than 22 layers")
    config.num_hidden_layers = 22
    if getattr(config, "layer_types", None) is not None:
        if len(config.layer_types) < 22:
            raise ValueError("Qwen layer_types has fewer than 22 entries")
        config.layer_types = config.layer_types[:22]
    config.use_cache = False
    return config


def make_aux_head(device):
    """Keep the saved auxiliary head architecture identical on train and reload."""
    import torch
    return torch.nn.Sequential(torch.nn.LayerNorm(8192,elementwise_affine=False),
                               torch.nn.Linear(8192,18)).to(device)


def load_qwen_checkpoint(source, checkpoint, device="cuda:0"):
    """Rebuild the 22-layer base and load native PEFT Linear and auxiliary weights."""
    import torch
    from peft import LoraConfig, PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    config=truncate_qwen_config(AutoConfig.from_pretrained(
        source,local_files_only=True,trust_remote_code=False))
    base=AutoModel.from_pretrained(source,config=config,dtype=torch.float16,
        device_map={"":device},attn_implementation="sdpa",
        local_files_only=True,trust_remote_code=False)
    base.requires_grad_(False)
    assert len([module for name,module in base.named_modules()
                if name.endswith(("q_proj","v_proj")) and type(module) is torch.nn.Linear])==44
    lora=register_plain_lora(LoraConfig.from_pretrained(checkpoint))
    model=PeftModel.from_pretrained(base,checkpoint,config=lora,is_trainable=False)
    assert len([name for name,module in model.named_modules()
                if hasattr(module,"lora_A") and name.endswith(("q_proj","v_proj"))])==44
    head=make_aux_head(device)
    head.load_state_dict(load_file(str(Path(checkpoint)/"aux_head.safetensors"),device="cpu"))
    model.eval();head.eval()
    return model,head


def compare_checkpoint_outputs(expected_views,loaded_views,expected_logits,loaded_logits):
    """Reject a reload whose pooled features or auxiliary outputs have drifted."""
    import torch
    if len(expected_views)!=len(loaded_views) or not expected_views:
        raise ValueError("Reload view count differs")
    views=[]
    for expected,loaded in zip(expected_views,loaded_views):
        expected=expected.detach().float().cpu();loaded=loaded.detach().float().cpu()
        if expected.shape!=loaded.shape or not torch.allclose(expected,loaded,atol=2e-3,rtol=2e-3):
            raise ValueError("Reloaded pooled view differs")
        views.append(float((expected-loaded).abs().max()))
    expected_logits=expected_logits.detach().float().cpu()
    loaded_logits=loaded_logits.detach().float().cpu()
    if expected_logits.shape!=loaded_logits.shape or not torch.allclose(
            expected_logits,loaded_logits,atol=2e-3,rtol=2e-3):
        raise ValueError("Reloaded auxiliary logits differ")
    return {"max_view_abs_diff":max(views),
            "max_aux_abs_diff":float((expected_logits-loaded_logits).abs().max()),
            "atol":2e-3,"rtol":2e-3}


def outer_partition(train, prompt_folds, outer_fold):
    """Return fit/outer row indices with each prompt wholly in one fold."""
    if outer_fold not in (0, 1, 2):
        raise ValueError("outer_fold must be 0, 1, or 2")
    prompts = train.prompt_id.to_numpy(dtype=str)
    if set(prompts) != set(prompt_folds) or len(prompt_folds) != 150:
        raise ValueError("Expected an exact mapping for 150 official prompts")
    if set(prompt_folds.values()) != {0, 1, 2}:
        raise ValueError("Prompt folds must be 0, 1, and 2")
    folds = np.array([prompt_folds[p] for p in prompts], dtype=int)
    fit, outer = np.flatnonzero(folds != outer_fold), np.flatnonzero(folds == outer_fold)
    fit_prompts, outer_prompts = set(prompts[fit]), set(prompts[outer])
    if len(fit_prompts) != 100 or len(outer_prompts) != 50 or fit_prompts & outer_prompts:
        raise ValueError("Expected disjoint 100/50 prompt partitions")
    classes = set(train.model)
    if set(train.model.iloc[fit]) != classes or set(train.model.iloc[outer]) != classes:
        raise ValueError("Each partition must contain every official label")
    return fit, outer


def train_qwen_pilot(root, core, protocol="inner", outer_fold=0):
    """Run the fixed Qwen recipe for an inner, outer, or full-data fit."""
    if protocol not in ("inner", "outer", "full") or outer_fold not in (0, 1, 2):
        raise ValueError("Expected inner/outer/full protocol and outer fold 0, 1, or 2")
    if protocol in ("inner", "full") and outer_fold != 0:
        raise ValueError("Inner and full protocols do not accept another outer fold")
    job_started = float(os.environ.get("NTL_JOB_STARTED_MONOTONIC", time.monotonic()))
    budget = float(os.environ.get("NTL_CHILD_BUDGET_SECONDS", "1100"))  # Parent enforces the hard budget.
    max_epochs = int(os.environ.get("NTL_MAX_EPOCHS", "1" if protocol in ("outer", "full") else "3"))
    if max_epochs not in (1, 2, 3):
        raise ValueError("NTL_MAX_EPOCHS must be 1, 2, or 3")
    if protocol in ("outer", "full") and max_epochs != 1:
        raise ValueError("Outer and full protocols require exactly one fixed epoch")
    assert Path("/kaggle/input").is_dir() and str(root).startswith("/kaggle/input/")
    out = Path("/kaggle/working/qwen_lora_pilot" if protocol == "inner"
               else "/kaggle/working/qwen_full_adapter" if protocol == "full"
               else f"/kaggle/working/qwen_outer_fold{outer_fold}")
    out.mkdir(exist_ok=False)
    manifest_name = "pilot_manifest.json" if protocol == "inner" else "full_manifest.json" if protocol == "full" else "fold_manifest.json"
    record = {"status": "STARTED", "protocol": protocol, "max_epochs": max_epochs,
              "outer_fold": outer_fold, "outer_evaluated": False,
              "program_budget_seconds": budget, "seed": 20261025, "epochs": [],
              "architecture": "Qwen3-8B-Base/1 truncated to22; hidden7/22 mean excluding special/padding",
              "training": {"rank":8, "alpha":16, "targets":["q_proj","v_proj"], "dropout":0.,
                           "adapter_lr":5e-5, "aux_head_lr":5e-4, "weight_decay":.01,
                           "micro_batch":1, "effective_batch":16, "max_length":512,
                           "precision":"FP16 frozen backbone, FP32 trainable parameters, FP16 autocast",
                           "aux_head":"fixed per-row LayerNorm then linear18; raw pooled features used by matched LR",
                           "schedule":f"constant LR; max{max_epochs} complete epochs; start only if full epoch+evaluation fits budget"},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "formal_baseline_sha256":os.environ.get("NTL_FORMAL_BASELINE_SHA256"),
              "bundle_sha256":os.environ.get("NTL_BUNDLE_SHA256"),
              "note":("Reused inner prompts; diagnostic, not blind evidence or an automatic candidate promotion."
                      if protocol == "inner" else
                      "Predeclared outer validation; no score-dependent training, epoch or head selection." if protocol == "outer" else
                       "Final full-data fit; training-set metrics are diagnostic only, with no held-out validation.")}
    def save_record():
        record["seconds"] = time.monotonic()-job_started
        (out/manifest_name).write_text(json.dumps(record,indent=2))
    def deadline(*_):
        raise TimeoutError("Qwen pilot reached total program wall-clock deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.01, budget-(time.monotonic()-job_started)))
    try:
        import importlib.metadata
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import save_file
        from sklearn.metrics import balanced_accuracy_score, log_loss
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
        record["libraries"] = {name: importlib.metadata.version(name) for name in
                               ("torch","transformers","peft","numpy","scikit-learn","safetensors","accelerate")}
        assert record["libraries"]["transformers"] == "5.16.1" and record["libraries"]["peft"] == "0.20.0"
        record["device"] = {"visible_count":torch.cuda.device_count(),"used_device":0,
                            "name":torch.cuda.get_device_name(0),
                            "total_bytes":torch.cuda.get_device_properties(0).total_memory}
        seed=20261025
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        train, _ = core.read_data(root)
        assert core.digest(root/"train.csv") == "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
        labels = np.array(sorted(train.model.unique()))
        assert len(labels) == 18
        folds = np.array([core.MODERNBERT_MIXED_FOLDS[p] for p in train.prompt_id])
        if protocol == "inner":
            partition=core.modernbert_inner_plan(train, folds)[0]
            fit, validation, reserved_outer = (partition[k] for k in ("fit_rows","inner_rows","outer_rows"))
            assert train.prompt_id.iloc[fit].nunique()==80 and train.prompt_id.iloc[validation].nunique()==20
            validation_name="inner"
            inner_ids=train.id.iloc[validation].tolist()
            outer_ids=train.id.iloc[reserved_outer].tolist()
        elif protocol == "outer":
            fit, validation=outer_partition(train,core.MODERNBERT_MIXED_FOLDS,outer_fold)
            validation_name="outer"
            inner_ids=[]
            outer_ids=train.id.iloc[validation].tolist()
        else:
            fit=np.arange(len(train),dtype=int)
            validation=np.array([],dtype=int)
            validation_name=None
            inner_ids=[]
            outer_ids=[]
        assert len(set(train.model.iloc[fit]))==18
        if protocol != "full":
            assert len(set(train.model.iloc[validation]))==18
        all_rows=np.concatenate([fit,validation])
        record["split"]={"fit_prompts":int(train.prompt_id.iloc[fit].nunique()),
                         "inner_prompts":len(set(train.prompt_id.iloc[validation])) if protocol == "inner" else 0,
                         "outer_prompts":0 if protocol == "full" else 50,"validation_role":validation_name,
                         "fit_ids":train.id.iloc[fit].tolist(),"inner_ids":inner_ids,
                         "outer_ids":outer_ids,"folds":core.MODERNBERT_MIXED_FOLDS}
        model_paths=[]
        for config_file in Path("/kaggle/input").rglob("config.json"):
            if "8b-base" in str(config_file).lower():
                config_data=json.loads(config_file.read_text())
                if config_data.get("model_type")=="qwen3" and config_data.get("hidden_size")==4096:
                    model_paths.append(config_file.parent)
        assert len(model_paths)==1
        source=model_paths[0]
        assert core.digest(source/"config.json")=="3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a"
        record["model_source"]=str(source);record["model_config_sha256"]=core.digest(source/"config.json")
        record["checkpoint_reload"]={
            "base_config_must_be_truncated_before_loading":22,
            "layer_types_must_also_be_truncated":True,
            "pooling_layers":[7,22],"verified_after_reload":False,
            "recipe":"Load original AutoConfig; truncate num_hidden_layers/layer_types to22; load AutoModel; register_plain_lora(LoraConfig.from_pretrained(adapter)); then PeftModel.from_pretrained(base,adapter,config=config). Auxiliary LayerNorm+linear18 uses aux_head.safetensors. Require reload output check before full OOF or inference."}
        config=truncate_qwen_config(AutoConfig.from_pretrained(
            source,local_files_only=True,trust_remote_code=False))
        (out/"effective_model_config.json").write_text(config.to_json_string())
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
        tokenizer.padding_side="right"
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        texts=train.text.to_numpy(dtype=object)
        actual=train.model.to_numpy(dtype=str)
        y=np.searchsorted(labels,actual)
        encoded=tokenizer([str(texts[r]) for r in all_rows],add_special_tokens=True,truncation=False,padding=False)
        tokens={int(r):{k:encoded[k][i] for k in encoded} for i,r in enumerate(all_rows)}
        lengths={int(r):len(tokens[int(r)]["input_ids"]) for r in all_rows}
        assert max(lengths.values())<=512
        record["max_length_observed"]=max(lengths.values());record["truncated_rows"]=0
        def inputs(rows):
            return {k:v.to("cuda:0") for k,v in tokenizer.pad(
                [tokens[int(r)] for r in rows],padding=True,return_tensors="pt").items()}
        model=AutoModel.from_pretrained(source,config=config,dtype=torch.float16,
                    device_map={"":"cuda:0"},attn_implementation="sdpa",local_files_only=True,trust_remote_code=False)
        model.requires_grad_(False)
        model.eval()
        stress_row=max(fit,key=lambda r:lengths[int(r)])
        def features(rows):
            batch=inputs(rows)
            result=model(**batch,output_hidden_states=True,use_cache=False)
            return pooled_views(result.hidden_states,batch["input_ids"],batch["attention_mask"],tokenizer.all_special_ids)
        with torch.inference_mode():
            before=tuple(x.detach().cpu() for x in features([stress_row]))
        targets=[module for name,module in model.named_modules() if name.endswith(("q_proj","v_proj"))]
        assert len(targets)==44 and all(type(target) is torch.nn.Linear for target in targets)
        assert not getattr(model,"is_quantized",False)
        lora_config=register_plain_lora(LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],
                                                lora_dropout=0.,bias="none",init_lora_weights=True))
        model=get_peft_model(model,lora_config)
        record["lora_dispatch"]="PEFT native Linear via documented custom mapping; nonquantized nn.Linear only"
        adapters=[name for name,module in model.named_modules() if hasattr(module,"lora_A") and name.endswith(("q_proj","v_proj"))]
        assert len(adapters)==44
        trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in name and p.dtype==torch.float32 for name,p in trainable)
        record["trainable_adapter_parameters"]=sum(p.numel() for _,p in trainable)
        record["adapter_modules"]=adapters
        model.eval()
        with torch.inference_mode():
            after=tuple(x.detach().cpu() for x in features([stress_row]))
        zero_diff=max(float((a-b).abs().max()) for a,b in zip(before,after))
        assert all(torch.allclose(a,b,atol=2e-3,rtol=2e-3) for a,b in zip(before,after))
        record["zero_adapter_max_abs_diff"]=zero_diff
        record["zero_adapter_check_scope"]="Same current-runtime backbone before/after zero LoRA; not an old-cache byte comparison."
        del before,after
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        head=make_aux_head("cuda:0")
        initial=snapshot_trainable(model,head)
        def optimizer():
            return torch.optim.AdamW([{"params":[p for _,p in trainable],"lr":5e-5},
                                      {"params":head.parameters(),"lr":5e-4}],weight_decay=.01)
        def update(opt,scaler):
            scaler.unscale_(opt)
            params=[p for _,p in trainable]+list(head.parameters())
            nonfinite=any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params)
            if not nonfinite:torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            old_scale=scaler.get_scale()
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            return scaler.get_scale()>=old_scale,nonfinite
        def train_rows(rows,opt,scaler):
            model.train();head.train();opt.zero_grad(set_to_none=True)
            seconds=[];success=skipped=0;loss_sum=0.;seen=0;grad_seen=False
            for i,row in enumerate(rows):
                torch.cuda.synchronize();tick=time.monotonic()
                window_start=(i//16)*16
                window_size=min(16,len(rows)-window_start)
                with torch.autocast("cuda",dtype=torch.float16):
                    views=features([row])
                    logits=head(torch.cat(views,dim=1))
                    loss=torch.nn.functional.cross_entropy(logits.float(),torch.as_tensor([y[row]],device="cuda:0"))
                scaler.scale(loss/window_size).backward()
                if not grad_seen:
                    grad_seen = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0
                                    for name,p in trainable if "lora_B" in name)
                loss_sum+=float(loss.detach());seen+=1
                del views,logits,loss
                if (i+1)%16==0 or i+1==len(rows):
                    ok,nonfinite=update(opt,scaler);success+=int(ok);skipped+=int(not ok)
                torch.cuda.synchronize();seconds.append(time.monotonic()-tick)
            return {"actual_steps":success,"amp_skips":skipped,"planned_steps":int(np.ceil(len(rows)/16)),
                    "online_mean_CE":loss_sum/seen,"rows":seen,"gradient_seen":grad_seen,
                    "micro_seconds":seconds}
        ordered=sorted(fit,key=lambda r:(lengths[int(r)],int(r)))
        probe_rows=np.array(ordered)[np.linspace(0,len(ordered)-1,32,dtype=int)]
        probe_optimizer=optimizer()
        probe_scaler=torch.amp.GradScaler("cuda",init_scale=128)
        probe=train_rows(probe_rows,probe_optimizer,probe_scaler)
        assert probe["actual_steps"]==2 and probe["amp_skips"]==0 and probe["gradient_seen"]
        record["probe"]={k:v for k,v in probe.items() if k!="micro_seconds"}
        record["probe"]["peak_allocated_bytes"]=torch.cuda.max_memory_allocated(0)
        record["probe"]["mean_micro_seconds"]=float(np.mean(probe["micro_seconds"]))
        restore_trainable(model,head,initial)
        del probe_optimizer,probe_scaler
        torch.cuda.empty_cache()
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        assert all(torch.equal(p.detach().cpu(),initial["model"][name]) for name,p in trainable)
        record["probe_reset_exact"]=True
        training_bound=float(np.mean(probe["micro_seconds"])*len(fit)*1.25)
        def evaluate(tag):
            tick=time.monotonic()
            model.eval();head.eval()
            values=[np.empty((len(all_rows),4096),dtype=np.float32) for _ in range(2)]
            order=np.argsort([lengths[int(r)] for r in all_rows])
            with torch.inference_mode():
                for start in range(0,len(order),4):
                    positions=order[start:start+4]
                    views=features(all_rows[positions])
                    for j,view in enumerate(views):values[j][positions]=view.cpu().numpy()
            assert all(np.isfinite(x).all() for x in values)
            probabilities=[]
            for value in values:
                p,classes=core.dense_head(value[:len(fit)],actual[fit],value)
                assert np.array_equal(classes,labels)
                probabilities.append(p)
            probability=np.mean(probabilities,axis=0)
            aux_logits=[]
            with torch.inference_mode():
                for start in range(0,len(all_rows),64):
                    joined=np.concatenate([v[start:start+64] for v in values],axis=1)
                    aux_logits.append(head(torch.from_numpy(joined).to("cuda:0")).float().cpu())
            logits=torch.cat(aux_logits)
            aux_probability=logits.softmax(1).numpy()
            metrics={}
            for name,take in ((("fit",slice(0,len(fit))),) if protocol == "full" else (("fit",slice(0,len(fit))),(validation_name,slice(len(fit),len(all_rows))))):
                truth=actual[all_rows[take]]
                metrics[name]={"readout_BA":float(balanced_accuracy_score(truth,labels[probability[take].argmax(1)])),
                               "readout_NLL":float(log_loss(truth,probability[take],labels=labels)),
                               "aux_BA":float(balanced_accuracy_score(truth,labels[aux_probability[take].argmax(1)])),
                               "aux_CE":float(log_loss(truth,aux_probability[take],labels=labels))}
            np.savez_compressed(out/f"{tag}_features.npz",ids=train.id.iloc[all_rows].to_numpy(dtype=str),
                                labels=labels,actual=actual[all_rows],fit_count=len(fit),
                                layer7=values[0],layer22=values[1],readout=probability,aux=aux_probability)
            metrics["seconds"]=time.monotonic()-tick
            return metrics
        baseline=evaluate("frozen")
        record["frozen_baseline"]=baseline
        record["outer_evaluated"]=protocol == "outer"
        evaluation_bound=baseline["seconds"]*1.25
        save_margin=90 if protocol == "inner" else 360  # Checkpoint write plus full reload gate.
        record["budget_projection"]={"training_bound_per_epoch":training_bound,"evaluation_bound":evaluation_bound,
                                     "elapsed_after_frozen":time.monotonic()-job_started,"save_margin":save_margin}
        save_record()
        print("QWEN_PROBE_AND_FROZEN_PASS",json.dumps(record["budget_projection"]),flush=True)
        opt=optimizer();scaler=torch.amp.GradScaler("cuda",init_scale=128)
        best=None
        for epoch in range(1,max_epochs+1):
            elapsed=time.monotonic()-job_started
            if not can_start_epoch(elapsed,training_bound,evaluation_bound,budget,save_margin):
                record["stop_reason"]="Complete next epoch and evaluation do not fit fixed wall budget"
                break
            rows=np.random.default_rng(seed+epoch).permutation(fit)
            stats=train_rows(rows,opt,scaler)
            assert stats["rows"]==len(fit) and len(set(rows))==len(fit)
            assert stats["actual_steps"]>0 and stats["gradient_seen"]
            stats.pop("micro_seconds")
            metrics=evaluate(f"epoch{epoch}")
            metrics.update(epoch=epoch,training=stats)
            record["epochs"].append(metrics)
            checkpoint=out/f"epoch{epoch}_adapter"
            model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
            save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(checkpoint/"aux_head.safetensors"))
            if protocol in ("outer", "full"):
                best=metrics  # Exactly one epoch; never select on held-out labels.
            elif best is None or metrics["inner"]["readout_BA"]>best["inner"]["readout_BA"]:
                best=metrics
            save_record()
            print("QWEN_PILOT_EPOCH" if protocol == "inner" else "QWEN_OUTER_EPOCH",
                  json.dumps(metrics),flush=True)
        if protocol in ("outer", "full") and best:
            # Two fixed rows; outer includes one held-out row, full uses two fit rows.
            import gc
            reference_rows=np.array([fit[0],fit[-1] if protocol == "full" else validation[0]],dtype=int)
            model.eval();head.eval()
            with torch.inference_mode():
                old_views=features(reference_rows)
                expected_views=tuple(x.detach().cpu() for x in old_views)
                expected_logits=head(torch.cat(old_views,dim=1)).float().cpu()
            allocated_before_release=torch.cuda.memory_allocated(0)
            del old_views,model,head,opt,scaler,targets,trainable,initial
            gc.collect();torch.cuda.empty_cache()
            allocated_after_release=torch.cuda.memory_allocated(0)
            record["checkpoint_reload"].update(
                allocated_before_release_bytes=allocated_before_release,
                allocated_after_release_bytes=allocated_after_release)
            if allocated_after_release>=allocated_before_release or allocated_after_release>=2*1024**3:
                raise RuntimeError("Original Qwen backbone remains allocated before checkpoint reload")
            model,head=load_qwen_checkpoint(source,checkpoint)
            with torch.inference_mode():
                loaded_views=features(reference_rows)
                loaded_logits=head(torch.cat(loaded_views,dim=1)).float().cpu()
            record["checkpoint_reload"].update(compare_checkpoint_outputs(
                expected_views,loaded_views,expected_logits,loaded_logits))
            record["checkpoint_reload"].update(
                verified_after_reload=True,reference_ids=train.id.iloc[reference_rows].tolist())
            save_record()
            print("QWEN_OUTER_RELOAD_PASS",json.dumps(record["checkpoint_reload"]),flush=True)
            del loaded_views,loaded_logits,expected_views,expected_logits
        record["epochs_completed"]=len(record["epochs"])
        record["peak_allocated_bytes"]=torch.cuda.max_memory_allocated(0)
        record["best_epoch"]=best["epoch"] if best else None
        if protocol == "inner":
            record["best_inner_delta"]=(best["inner"]["readout_BA"]-baseline["inner"]["readout_BA"]) if best else None
            record["eligible_for_full_oof_design"]=bool(best and record["best_inner_delta"]>=.01
                 and best["fit"]["aux_BA"]>baseline["fit"]["aux_BA"]
                 and best["training"]["amp_skips"]<=.05*best["training"]["planned_steps"])
        else:
            if protocol == "outer":
                record["outer_delta_BA"]=(best["outer"]["readout_BA"]-baseline["outer"]["readout_BA"]) if best else None
            record["eligible_for_full_oof_design"]=False
        record["status"]="COMPLETE" if best else "BUDGET_STOP"
        record["candidate_promoted"]=False
    except Exception as exc:
        record.update(status="TIMEOUT" if isinstance(exc,TimeoutError) else "FAIL",error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        save_record()
        print("QWEN_PILOT_RESULT",json.dumps({k:record.get(k) for k in
              ("status","seconds","epochs_completed","best_epoch","best_inner_delta","outer_delta_BA","error")}),flush=True)
