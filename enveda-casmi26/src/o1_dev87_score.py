"""Local safe87 paired MRR@25 scorer; never included in the inference package."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import rdkit
import baseline

ROOT=Path(__file__).resolve().parents[1]
MEMBERS=ROOT/"outputs/phase5_20261023_01/members.json"
RECORD=ROOT/"outputs/p5_mass2_20261027_01/full_input/record.json"
MEMBERS_SHA="577fb395cdb8122b783b3f143f7af640d0871d89a01103fb865580fa909de943"
RECORD_SHA="c36b5c00ff4a2e54d68a26588ecc13e139eba24f3c34fdc8e5a76f88edc71946"
BASELINE_SHA="6ca77ff438fba8804bcf966cecd48189ad3ee908418bfd6e367a4480d3564ea9"
OFFICIAL_METRIC_SHA="bf8535f01bf73268a52819b38e7f6b205e37ec127c268e48081b780a10b609dd"
OFFICIAL_METRIC_FILE=ROOT/"outputs/o1_dev87_scoring_20261030_01/official_metric_copy/casmi-mean-reciprocal-rank.ipynb"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def safe_truth():
    if sha(OFFICIAL_METRIC_FILE)!=OFFICIAL_METRIC_SHA:
        raise ValueError("Official metric copy SHA changed")
    if sha(MEMBERS)!=MEMBERS_SHA or sha(RECORD)!=RECORD_SHA:
        raise ValueError("Pinned safe87 member or cohort record SHA changed")
    if sha(ROOT/"src/baseline.py")!=BASELINE_SHA or rdkit.__version__!="2026.03.3":
        raise ValueError("Pinned canonicalization source or RDKit version changed")
    record=json.loads(RECORD.read_text())
    qids=record["qids"]
    if record["status"]!="p5_model_inputs_frozen" or len(qids)!=87 or len(set(qids))!=87 or record["model_qids"]!=qids:
        raise ValueError("safe87 qid identity or order changed")
    members=json.loads(MEMBERS.read_text())
    if members.get("sealed_scored") is not False or len(members["members"])!=140:
        raise ValueError("Exposed development member scope changed")
    selected={m["qid"]:m["canonical_target_key"] for m in members["members"] if m["qid"] in set(qids)}
    if len(selected)!=87 or any(len(k)!=14 for k in selected.values()):
        raise ValueError("safe87 target key intersection incomplete")
    return qids,selected

def parse_predictions(path,qids):
    allowed=set(qids); seen={}
    with open(path,newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or not {"molecule_id","smiles"}.issubset(reader.fieldnames):
            raise ValueError("Missing molecule_id/smiles columns")
        for row in reader:
            mid=row.get("molecule_id")
            raw=row.get("smiles")
            if not isinstance(mid,str) or not mid:
                raise ValueError("Empty molecule_id")
            if mid not in allowed:
                raise ValueError("Unknown molecule_id: "+mid)
            if mid in seen:
                raise ValueError("Duplicate molecule_id: "+mid)
            if not isinstance(raw,str):
                raise ValueError("Null smiles for "+mid)
            # Official metric: strip and discard empty split segments before assigning ranks.
            guesses=[x for x in (part.strip() for part in raw.split(";")) if x]
            if not guesses or len(guesses)>25:
                raise ValueError("Expected 1..25 nonempty guesses for "+mid)
            seen[mid]=guesses
    return seen

def arm_rows(qids,truth,preds):
    cache={}; rows=[]
    for qid in qids:
        guesses=preds.get(qid,[])
        keys=[]
        for smile in guesses:
            if smile not in cache: cache[smile]=baseline.canonical_key(smile)
            keys.append(cache[smile])
        rank=next((i for i,key in enumerate(keys,1) if key==truth[qid]),None)
        rows.append({"qid":qid,"rank":rank,"rr":0.0 if rank is None else 1/rank,
                     "guess_count":len(guesses),"invalid_nonempty_count":sum(k is None for k in keys),
                     "duplicate_canonical_count":len([k for k in keys if k])-len(set(k for k in keys if k)),
                     "top25_truth_hit":rank is not None})
    return rows

def read_rankings(path,qids):
    if path is None: return {"status":"not_provided","pre_rank_truth_recall":"not_evaluated_no_structure_key_map"},{}
    rows={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            item=json.loads(line)
            q=item["molecule_id"]
            if q not in qids or q in rows: raise ValueError("Unknown/duplicate rankings qid")
            indices=item["candidate_pool_indices"]; n=len(indices)
            if len(set(indices))!=n or any(not isinstance(i,int) or i<0 for i in indices):
                raise ValueError("Candidate pool indices invalid")
            for arm in ("sqrt","linear"):
                order=item[arm+"_order_positions"]; probs=item[arm+"_probabilities"]
                if len(order)!=n or set(order)!=set(range(n)) or len(probs)!=n:
                    raise ValueError("Rank order/probability length or permutation invalid: "+q)
                if any(not math.isfinite(float(v)) for v in probs):
                    raise ValueError("Nonfinite rank probability: "+q)
            rows[q]=item
    if set(rows)!=set(qids): raise ValueError("Incomplete full rankings qid set")
    packed=json.dumps([[q,rows[q]["candidate_pool_indices"]] for q in qids],separators=(",",":")).encode()
    identity={"status":"verified_shared_index_list","qid_count":len(qids),
              "candidate_pool_ordered_indices_sha256":hashlib.sha256(packed).hexdigest(),
              "candidate_count_total":sum(len(rows[q]["candidate_pool_indices"]) for q in qids),
              "pre_rank_truth_recall":"not_evaluated_no_structure_key_map"}
    return identity,{q:len(rows[q]["candidate_pool_indices"]) for q in qids}

def compare(qids,truth,sqrt_preds,linear_preds,rankings=None):
    s=arm_rows(qids,truth,sqrt_preds); l=arm_rows(qids,truth,linear_preds)
    identity,counts=read_rankings(rankings,qids)
    individual=[]; deltas=[]
    for left,right in zip(s,l):
        assert left["qid"]==right["qid"]
        delta=right["rr"]-left["rr"]; deltas.append(delta)
        individual.append({"qid":left["qid"],"sqrt_rank":left["rank"],"linear_rank":right["rank"],
          "sqrt_rr":left["rr"],"linear_rr":right["rr"],"delta_rr":delta,
          "sqrt_top25_hit":left["top25_truth_hit"],"linear_top25_hit":right["top25_truth_hit"],
          "sqrt_guess_count":left["guess_count"],"linear_guess_count":right["guess_count"],
          "sqrt_invalid_nonempty":left["invalid_nonempty_count"],
          "linear_invalid_nonempty":right["invalid_nonempty_count"],
          "candidate_pool_count":counts.get(left["qid"])})
    n=len(qids)
    if n==0: raise ValueError("Empty scoring cohort")
    metric=lambda rows:{"mrr25":sum(r["rr"] for r in rows)/n,
      "top1_count":sum(r["rank"]==1 for r in rows),
      "top25_count":sum(r["rank"] is not None for r in rows),
      "top1":sum(r["rank"]==1 for r in rows)/n,
      "top25":sum(r["rank"] is not None for r in rows)/n,
      "top25_truth_coverage":sum(r["top25_truth_hit"] for r in rows)/n}
    ms,ml=metric(s),metric(l)
    max_delta=max(deltas); drop_idx=deltas.index(max_delta)
    without=(sum(deltas)-max_delta)/(n-1) if n>1 else float("nan")
    complete=len(sqrt_preds)==n and len(linear_preds)==n
    gate={"complete_inputs":complete,"delta_mrr_positive":ml["mrr25"]>ms["mrr25"],
          "top25_not_lower":ml["top25_count"]>=ms["top25_count"],
          "drop_largest_delta_still_positive":without>0}
    summary={"cohort_size":n,"sqrt":ms,"linear":ml,"delta_mrr25":ml["mrr25"]-ms["mrr25"],
      "wins":sum(d>0 for d in deltas),"ties":sum(d==0 for d in deltas),
      "losses":sum(d<0 for d in deltas),
      "largest_single_delta_rr":{"qid":qids[drop_idx],"value":max_delta},
      "delta_mrr_without_largest_single_gain":without,
      "missing_sqrt":[q for q in qids if q not in sqrt_preds],
      "missing_linear":[q for q in qids if q not in linear_preds],
      "candidate_pool_identity":identity,
      "gate":gate,"gate_pass":all(gate.values())}
    return summary,individual

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sqrt-csv",type=Path,required=True)
    ap.add_argument("--linear-csv",type=Path,required=True)
    ap.add_argument("--rankings-jsonl",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    qids,truth=safe_truth()
    sqrt_preds=parse_predictions(args.sqrt_csv,qids)
    linear_preds=parse_predictions(args.linear_csv,qids)
    summary,individual=compare(qids,truth,sqrt_preds,linear_preds,args.rankings_jsonl)
    summary["input_sha256"]={"sqrt_csv":sha(args.sqrt_csv),"linear_csv":sha(args.linear_csv),
      "rankings_jsonl":sha(args.rankings_jsonl) if args.rankings_jsonl else None,
      "safe87_record":RECORD_SHA,"dev_members":MEMBERS_SHA,"baseline_source":BASELINE_SHA,
      "official_metric_copy":OFFICIAL_METRIC_SHA}
    summary["environment"]={"rdkit":rdkit.__version__,"python_entry":"project .venv/bin/python"}
    summary["scope"]="local exposed safe87 truth only; no sealed70 or inference input mutation"
    args.out.mkdir(parents=True)
    (args.out/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.out/"per_qid.json").write_text(json.dumps(individual,indent=2)+"\n")
    record={"status":"scored_local_safe87","source_sha256":sha(Path(__file__)),
      "metrics_sha256":sha(args.out/"metrics.json"),"per_qid_sha256":sha(args.out/"per_qid.json"),
      "complete":summary["gate"]["complete_inputs"],"gate_pass":summary["gate_pass"],
      "input_sha256":summary["input_sha256"]}
    (args.out/"run_record.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"status":record["status"],"complete":record["complete"],
                      "gate_pass":record["gate_pass"],"delta_mrr25":summary["delta_mrr25"]}))
if __name__=="__main__": main()

