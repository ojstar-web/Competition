"""Run two frozen, structure-disjoint local candidate policies in sequence.

The first generation may already be running at campaign root. This runner waits
for its receipt; subsequent generation and decisions are unattended.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from strategy_campaign import assess, clean_rejected


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/fold1_ood_retrieval_20261110_01"
OUT = ROOT / "outputs/local_strategy_campaign_20261119_01"
GENERATOR = SOURCE / "generate_candidates.py"
EVALUATOR = ROOT / "outputs/candidate_fusion_20261119_01/evaluate.py"
sys.path.insert(0, str(EVALUATOR.parent))
from evaluate import rank  # noqa: E402

QUEUE = (("confirm", "pool20_pc5"), ("final", "alternate1_1"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def verify_sources(campaign: dict) -> None:
    for name, expected in campaign["sources_sha256"].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    active = json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())
    if active["model_sha256"] != sha(ROOT / active["model_path"]):
        raise ValueError("Active model SHA changed")
    if active["generator_sha256"] != sha(GENERATOR):
        raise ValueError("Candidate generator changed")


def molecule_rank(query_rows: list[dict], policy: str) -> list[str]:
    """Combine spectra before scoring one molecule; singleton ranking is unchanged."""
    totals = defaultdict(float)
    for row in query_rows:
        for pos, key in enumerate(rank(policy, row), 1):
            totals[key] += 1 / pos
    return [key for key, _ in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:25]]


def score_rows(part: dict, candidates: list[dict], labels: dict, policy: str,
               baseline_policy: str = "pool25") -> list[dict]:
    if sorted(r["query_id"] for r in candidates) != part["query_ids"]:
        raise ValueError("Candidate IDs differ from frozen partition")
    by_group = defaultdict(list)
    expected = {int(g): arm for arm, group in part["groups"].items() for g in group}
    for row in candidates:
        label = labels[row["query_id"]]
        group = int(label["identity_group"])
        if group not in expected or label["arm"] != expected[group]:
            raise ValueError(f"Candidate identity/arm mismatch: {row['query_id']}")
        by_group[group].append(row)
    if set(by_group) != set(expected):
        raise ValueError("Candidate groups differ from frozen partition")
    scored = []
    for group, query_rows in sorted(by_group.items()):
        truths = {labels[row["query_id"]]["truth_key"] for row in query_rows}
        if len(truths) != 1:
            raise ValueError(f"Multiple truths for molecule {group}")
        truth = truths.pop()
        baseline = molecule_rank(query_rows, baseline_policy)
        candidate = molecule_rank(query_rows, policy)
        scored.append({"molecule_id": group, "identity_group": group,
                       "query_ids": [r["query_id"] for r in query_rows],
                       "arm": expected[group], "truth_key": truth,
                       "baseline_rr": 1 / (baseline.index(truth) + 1) if truth in baseline else 0.0,
                       "candidate_rr": 1 / (candidate.index(truth) + 1) if truth in candidate else 0.0,
                       "baseline_recall": int(truth in baseline),
                       "candidate_recall": int(truth in candidate)})
    return scored


def generate(name: str, run: Path) -> None:
    candidate = run / "candidates.jsonl"
    receipt = run / "generation.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if sha(candidate) != saved["candidate_sha256"]:
            raise ValueError(f"Completed {name} candidate SHA mismatch")
        return
    active = json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())
    external_pid = active.get(f"external_{name}_pid")
    if external_pid is not None:
        expected = str(GENERATOR).encode()
        while not receipt.exists():
            command = Path(f"/proc/{external_pid}/cmdline")
            if not command.exists() or expected not in command.read_bytes():
                raise RuntimeError(f"External {name} generator ended without a receipt")
            time.sleep(20)
        saved = json.loads(receipt.read_text())
        if sha(candidate) != saved["candidate_sha256"]:
            raise ValueError(f"External {name} candidate SHA mismatch")
        return
    with (run / "run.log").open("a") as log:
        subprocess.run([sys.executable, str(GENERATOR), "--scope", "campaign",
                        "--ids-file", str(OUT / f"{name}_ids.json"),
                        "--output-file", str(candidate), "--receipt-file", str(receipt)],
                       cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    saved = json.loads(receipt.read_text())
    if sha(candidate) != saved["candidate_sha256"]:
        raise ValueError(f"Generated {name} candidate SHA mismatch")


def run_one(name: str, policy: str, campaign: dict, labels: dict) -> dict:
    run = OUT / f"{name}_{policy}"
    run.mkdir(exist_ok=True)
    report_file = run / "report.json"
    if report_file.exists():
        saved = json.loads(report_file.read_text())
        if (sha(run / "protocol.json") != saved["protocol_sha256"]
                or sha(run / "rows.jsonl") != saved["rows_sha256"]):
            raise ValueError(f"Completed {name} report identity mismatch")
        if not saved["accepted"] and not (run / "cleanup.json").exists():
            clean_rejected(run)
        return saved
    if name == "confirm" and not (OUT / "confirm_generation.json").exists():
        pid = json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())["external_confirm_pid"]
        while not (OUT / "confirm_generation.json").exists():
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("External confirmation generator ended without a receipt") from error
            time.sleep(20)
    if name == "confirm" and (OUT / "confirm_generation.json").exists():
        receipt = json.loads((OUT / "confirm_generation.json").read_text())
        if sha(OUT / "confirm_candidates.jsonl") != receipt["candidate_sha256"]:
            raise ValueError("External confirm generation SHA mismatch")
        (OUT / "confirm_candidates.jsonl").replace(run / "candidates.jsonl")
        (OUT / "confirm_generation.json").replace(run / "generation.json")
    generate(name, run)
    part = campaign["partitions"][name]
    prior = set(campaign["prior_groups"])
    design = set(sum(campaign["partitions"]["design"]["groups"].values(), []))
    if name == "final":
        prior.update(sum(campaign["partitions"]["confirm"]["groups"].values(), []))
    baseline_policy = "pool25" if name == "confirm" else "pool20_pc5"
    protocol = {"partition": name, "policy": policy,
                "baseline_policy": baseline_policy,
                "confirm_groups": sorted(sum(part["groups"].values(), [])),
                "train_groups": [], "design_groups": sorted(design),
                "prior_confirm_groups": sorted(prior),
                "candidate_universe_changed": True,
                "campaign_protocol_sha256": sha(OUT / "protocol.json"),
                "generator_sha256": sha(GENERATOR), "evaluator_sha256": sha(EVALUATOR),
                "model_sha256": json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())["model_sha256"],
                "labels_sha256": sha(SOURCE / "scoring_labels_corrected.parquet"),
                "generation_sha256": sha(run / "generation.json")}
    write_json(run / "protocol.json", protocol)
    shutil.copy2(Path(__file__), run / "source_snapshot.py")
    candidates = [json.loads(line) for line in (run / "candidates.jsonl").open()]
    rows = score_rows(part, candidates, labels, policy, baseline_policy)
    (run / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assess(protocol, rows)
    result.update({"partition": name, "policy": policy,
                   "protocol_sha256": sha(run / "protocol.json"),
                   "rows_sha256": sha(run / "rows.jsonl"),
                   "candidates_sha256": sha(run / "candidates.jsonl"),
                   "queries": len(candidates), "molecules": len(rows),
                   "baseline_mrr": sum(r["baseline_rr"] for r in rows) / len(rows),
                   "candidate_mrr": sum(r["candidate_rr"] for r in rows) / len(rows)})
    write_json(report_file, result)
    if not result["accepted"]:
        clean_rejected(run)
    return result


def main() -> None:
    results = []
    write_json(OUT / "queue_status.json", {"status": "running", "completed": results,
                                            "remaining": [n for n, _ in QUEUE]})
    try:
        campaign = json.loads((OUT / "protocol.json").read_text())
        verify_sources(campaign)
        labels = {int(row["query_id"]): row for row in
                  pq.read_table(SOURCE / "scoring_labels_corrected.parquet").to_pylist()}
        for name, policy in QUEUE:
            result = run_one(name, policy, campaign, labels)
            results.append({"partition": name, "policy": policy, "accepted": result["accepted"],
                            "mrr_delta": result["mrr_delta"]})
            write_json(OUT / "queue_status.json", {"status": "running", "completed": results,
                                                    "remaining": [n for n, _ in QUEUE[len(results):]]})
            print(json.dumps(results[-1]), flush=True)
        write_json(OUT / "queue_status.json", {"status": "complete", "completed": results,
                                                "remaining": []})
        consumer = json.loads((OUT / "ACTIVE_CONSUMER.json").read_text())
        consumer["status"] = "terminal"
        write_json(OUT / "ACTIVE_CONSUMER.json", consumer)
    except Exception:
        failure = {"completed": results, "traceback": traceback.format_exc()}
        write_json(OUT / "queue_failure.json", failure)
        write_json(OUT / "queue_status.json", {"status": "failed", "completed": results,
                                                "remaining": [n for n, _ in QUEUE[len(results):]]})
        raise


if __name__ == "__main__":
    main()
