"""Validate the pilot-fixed score-gap gate on the campaign's final molecules."""
from __future__ import annotations

import json
import shutil
import traceback
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from run_candidate_queue import (EVALUATOR, GENERATOR, OUT, ROOT, SOURCE,
                                 rank, sha, score_rows, verify_sources, write_json)
from strategy_campaign import assess, clean_rejected


RUN = OUT / "final_score_gap_gate"
ADAPTIVE = ROOT / "outputs/candidate_fusion_20261119_01/adaptive.py"
ADAPTIVE_REPORT = ADAPTIVE.parent / "adaptive_report.json"
CONSUMER = OUT / "GATE_ACTIVE_CONSUMER.json"


def finish_consumer(status: str) -> None:
    record = json.loads(CONSUMER.read_text())
    record["status"] = status
    write_json(CONSUMER, record)


def gate_rank(rows: list[dict], threshold: float) -> list[str]:
    totals = defaultdict(float)
    for row in rows:
        pool, pc = row["a_top25"], row["pubchem_shortlist"]
        gap = pc[0][1] - pool[0][1] if pc and pool else (float("inf") if pc else float("-inf"))
        policy = "alternate1_1" if gap > threshold else "pool20_pc5"
        for pos, key in enumerate(rank(policy, row), 1):
            totals[key] += 1 / pos
    return [key for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:25]]


def main() -> None:
    campaign = json.loads((OUT / "protocol.json").read_text())
    verify_sources(campaign)
    selected = json.loads(ADAPTIVE_REPORT.read_text())["selected"]
    if selected["low_gap_policy"] != "pool20_pc5":
        raise ValueError("Frozen low-gap policy changed")
    threshold = selected["threshold"]
    source_receipt = OUT / "final_alternate1_1/generation.json"
    receipt = json.loads(source_receipt.read_text())
    candidates_file = RUN / "candidates.jsonl"
    if sha(candidates_file) != receipt["candidate_sha256"]:
        raise ValueError("Gate candidate SHA differs from generation receipt")
    shutil.copy2(source_receipt, RUN / "generation.json")
    part = campaign["partitions"]["final"]
    labels = {int(row["query_id"]): row for row in
              pq.read_table(SOURCE / "scoring_labels_corrected.parquet").to_pylist()}
    candidates = [json.loads(line) for line in candidates_file.open()]
    rows = score_rows(part, candidates, labels, "pool20_pc5", "pool20_pc5")
    by_group = defaultdict(list)
    for candidate in candidates:
        by_group[int(labels[candidate["query_id"]]["identity_group"])].append(candidate)
    for row in rows:
        ranked = gate_rank(by_group[row["identity_group"]], threshold)
        truth = row["truth_key"]
        row["candidate_rr"] = 1 / (ranked.index(truth) + 1) if truth in ranked else 0.0
        row["candidate_recall"] = int(truth in ranked)
    design = set(sum(campaign["partitions"]["design"]["groups"].values(), []))
    prior = set(campaign["prior_groups"]) | set(sum(campaign["partitions"]["confirm"]["groups"].values(), []))
    protocol = {"partition": "final", "policy": "score_gap_gate",
                "baseline_policy": "pool20_pc5", "score_gap_threshold": threshold,
                "low_gap_policy": selected["low_gap_policy"],
                "confirm_groups": sorted(sum(part["groups"].values(), [])),
                "train_groups": [], "design_groups": sorted(design),
                "prior_confirm_groups": sorted(prior),
                "candidate_universe_changed": True,
                "campaign_protocol_sha256": sha(OUT / "protocol.json"),
                "pilot_selection_sha256": sha(ADAPTIVE_REPORT),
                "pilot_implementation_sha256": sha(ADAPTIVE),
                "generator_sha256": sha(GENERATOR), "evaluator_sha256": sha(EVALUATOR),
                "model_sha256": json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())["model_sha256"],
                "labels_sha256": sha(SOURCE / "scoring_labels_corrected.parquet"),
                "generation_sha256": sha(RUN / "generation.json")}
    write_json(RUN / "protocol.json", protocol)
    shutil.copy2(Path(__file__), RUN / "source_snapshot.py")
    (RUN / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assess(protocol, rows)
    result.update({"partition": "final", "policy": "score_gap_gate",
                   "protocol_sha256": sha(RUN / "protocol.json"),
                   "rows_sha256": sha(RUN / "rows.jsonl"),
                   "candidates_sha256": sha(candidates_file),
                   "queries": len(candidates), "molecules": len(rows),
                   "baseline_mrr": sum(r["baseline_rr"] for r in rows) / len(rows),
                   "candidate_mrr": sum(r["candidate_rr"] for r in rows) / len(rows)})
    write_json(RUN / "report.json", result)
    if not result["accepted"]:
        clean_rejected(RUN)
    finish_consumer("terminal")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "failure.json", {"traceback": traceback.format_exc()})
        finish_consumer("failed")
        raise
