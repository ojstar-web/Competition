"""Local fixed100 paired rank-aggregation scorer; truth never goes to Kaggle."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import rdkit
import baseline
from o1_dev87_score import arm_rows, parse_predictions, sha

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "outputs/rank_aggregation_fixed100_adapter_20261030_01"
TRUTH_SHA = "338dc5fce6184371d8eb123f2d2bec74085d69a2bce686cd864ae2d8a82eaff2"
SAMPLE_SHA = "86d145377a10798b25382b4cc7f4628a65cd035658d45fa557e6b303caca5cb0"
BASELINE_SHA = "6ca77ff438fba8804bcf966cecd48189ad3ee908418bfd6e367a4480d3564ea9"

def cohort():
    t, s = ADAPTER / "truth_local.json", ADAPTER / "sample_submission.csv"
    if sha(t) != TRUTH_SHA or sha(s) != SAMPLE_SHA:
        raise ValueError("Frozen cohort SHA changed")
    if sha(ROOT / "src/baseline.py") != BASELINE_SHA or rdkit.__version__ != "2026.03.3":
        raise ValueError("Canonicalization changed")
    with s.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["molecule_id"]:
            raise ValueError("Sample schema changed")
        qids = [row["molecule_id"] for row in reader]
    truth_rows = json.loads(t.read_text())
    truth = {row["molecule_id"]: row["canonical_key"] for row in truth_rows}
    if len(qids) != 100 or len(set(qids)) != 100 or len(truth_rows) != 100 or set(qids) != set(truth):
        raise ValueError("Fixed100 cohort changed")
    return qids, truth

def tied_rank(p):
    order = sorted(range(len(p)), key=lambda i: -p[i])
    result = [0.] * len(p)
    a = 0
    while a < len(p):
        b = a + 1
        while b < len(p) and p[order[b]] == p[order[a]]:
            b += 1
        for i in order[a:b]:
            result[i] = (a + 1 + b) / 2.
        a = b
    return result

def check_rankings(path, qids, truth, control, rankagg):
    seen, cache, recall = set(), {}, {}
    total = empty = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["molecule_id"]
            if q not in truth or q in seen:
                raise ValueError("Unknown/duplicate ranking qid")
            seen.add(q)
            indices, smiles = row["candidate_pool_indices"], row["candidate_smiles"]
            n = len(indices)
            if len(set(indices)) != n or len(smiles) != n:
                raise ValueError("Candidate pool changed: " + q)
            total += n
            if n == 0:
                empty += 1
                if control[q] != ["CCO"] * 25 or rankagg[q] != ["CCO"] * 25:
                    raise ValueError("Empty-pool fallback changed: " + q)
                recall[q] = False
                continue
            probs = np.asarray(row["per_model_probabilities"], dtype=float)
            if probs.shape != (8, n) or not np.isfinite(probs).all():
                raise ValueError("Eight GBM vectors missing/nonfinite: " + q)
            mean = np.mean(probs, axis=0)
            ranks = np.mean([tied_rank(p) for p in probs], axis=0)
            if not np.allclose(mean, row["mean_probabilities"], rtol=0, atol=1e-12):
                raise ValueError("Mean probabilities changed: " + q)
            if not np.allclose(ranks, row["mean_ranks"], rtol=0, atol=1e-12):
                raise ValueError("Mean tie ranks changed: " + q)
            co = np.argsort(-mean).tolist()
            ra = np.lexsort((indices, -mean, ranks)).tolist()
            if row["control_order_positions"] != co or row["rankagg_order_positions"] != ra:
                raise ValueError("Order differs from one shared pool: " + q)
            for name, order, pred in (("control", co, control), ("rankagg", ra, rankagg)):
                expected = [smiles[i] for i in order[:25]]
                expected += ["CCO"] * (25 - len(expected))
                if pred[q] != expected:
                    raise ValueError(name + " CSV differs from full rankings: " + q)
            for smi in smiles:
                if smi not in cache:
                    cache[smi] = baseline.canonical_key(smi)
            recall[q] = any(cache[smi] == truth[q] for smi in smiles)
    if seen != set(qids):
        raise ValueError("Full rankings incomplete")
    return {"candidate_count_total": total, "empty_pool_count": empty,
            "candidate_pool_truth_recall_count": sum(recall.values())}, recall

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control-csv", type=Path, required=True)
    p.add_argument("--rankagg-csv", type=Path, required=True)
    p.add_argument("--rankings-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    qids, truth = cohort()
    control = parse_predictions(a.control_csv, qids)
    rankagg = parse_predictions(a.rankagg_csv, qids)
    if set(control) != set(qids) or set(rankagg) != set(qids):
        raise ValueError("CSV qid set incomplete")
    pool, recall = check_rankings(a.rankings_jsonl, qids, truth, control, rankagg)
    c, r = arm_rows(qids, truth, control), arm_rows(qids, truth, rankagg)
    deltas = [y["rr"] - x["rr"] for x, y in zip(c, r)]
    def metric(rows):
        return {"mrr25": sum(x["rr"] for x in rows)/100,
                "top1_count": sum(x["rank"] == 1 for x in rows),
                "top25_count": sum(x["rank"] is not None for x in rows)}
    cm, rm = metric(c), metric(r)
    without = (sum(deltas) - max(deltas))/99
    gate = {"mrr25_strictly_higher": rm["mrr25"] > cm["mrr25"],
            "top1_not_lower": rm["top1_count"] >= cm["top1_count"],
            "top25_not_lower": rm["top25_count"] >= cm["top25_count"],
            "drop_largest_gain_delta_positive": without > 0}
    result = {"scope": "local fixed100 proxy; no sealed70 or competition score",
              "denominator": 100, "control": cm, "rankagg": rm,
              "delta_mrr25": rm["mrr25"] - cm["mrr25"],
              "wins": sum(x > 0 for x in deltas), "ties": sum(x == 0 for x in deltas),
              "losses": sum(x < 0 for x in deltas), "delta_mrr_without_largest_gain": without,
              "pool": pool, "gate": gate, "gate_pass": all(gate.values()),
              "input_sha256": {"control": sha(a.control_csv), "rankagg": sha(a.rankagg_csv),
                               "rankings": sha(a.rankings_jsonl), "truth_local": TRUTH_SHA,
                               "sample": SAMPLE_SHA, "baseline": BASELINE_SHA},
              "environment": {"rdkit": rdkit.__version__}}
    per_qid = [{"qid": q, "control_rank": x["rank"], "rankagg_rank": y["rank"],
                "control_rr": x["rr"], "rankagg_rr": y["rr"], "delta_rr": d,
                "candidate_pool_truth_recall": recall[q]}
               for q, x, y, d in zip(qids, c, r, deltas)]
    a.out.mkdir(parents=True)
    (a.out/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    (a.out/"per_qid.json").write_text(json.dumps(per_qid, indent=2)+"\n")
    print(json.dumps({"gate_pass": result["gate_pass"], "delta_mrr25": result["delta_mrr25"]}))

if __name__ == "__main__":
    main()
