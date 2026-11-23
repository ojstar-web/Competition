"""Decision and owned-output cleanup for local CASMI strategy experiments.

An experiment writes rows.jsonl with one row per scored molecule. The runner does
not infer independence: a frozen protocol must identify the exact confirmation
groups and all groups already used for training, tuning, or earlier confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261119
KEEP = {"protocol.json", "rows.jsonl", "report.json", "generation.json", "run.log",
        "failure.json", "cleanup.json", "source_snapshot.py"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(rows: list[dict], seed: int) -> list[float]:
    groups = sorted({r["identity_group"] for r in rows})
    sums = np.asarray([sum(r["candidate_rr"] - r["baseline_rr"]
                           for r in rows if r["identity_group"] == group) for group in groups])
    counts = np.asarray([sum(r["identity_group"] == group for r in rows) for group in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(10000, len(groups)))
    return [float(x) for x in np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [0.025, 0.975])]


def assess(protocol: dict, rows: list[dict]) -> dict:
    confirm = set(protocol["confirm_groups"])
    prior = set(protocol["train_groups"]) | set(protocol["design_groups"]) | set(protocol["prior_confirm_groups"])
    if not confirm or confirm & prior:
        raise ValueError("Confirmation groups overlap training, design, or earlier confirmation")
    if not rows or {r["identity_group"] for r in rows} != confirm:
        raise ValueError("Confirmation rows do not match frozen identity groups")
    if len({r["molecule_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate molecule result")
    if any(r["arm"] not in ("control", "ood") for r in rows):
        raise ValueError("Unknown subgroup")
    for row in rows:
        for key in ("baseline_rr", "candidate_rr"):
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key}")
    delta = np.asarray([r["candidate_rr"] - r["baseline_rr"] for r in rows])
    control = [r for r in rows if r["arm"] == "control"]
    if not control:
        raise ValueError("A pool-in control subgroup is required")
    overall_ci = interval(rows, SEED)
    control_ci = interval(control, SEED + 1)
    result = {"molecules": len(rows), "mrr_delta": float(delta.mean()),
              "mrr_delta_ci95": overall_ci, "control_delta_ci95": control_ci,
              "overall_pass": overall_ci[0] > 0,
              "control_pass": control_ci[0] > -0.003}
    if protocol["candidate_universe_changed"]:
        ood = [r for r in rows if r["arm"] == "ood"]
        if not ood:
            raise ValueError("Candidate-universe change requires OOD examples")
        old_recall = np.mean([r["baseline_recall"] for r in ood])
        new_recall = np.mean([r["candidate_recall"] for r in ood])
        result["ood_recall"] = {"baseline": float(old_recall), "candidate": float(new_recall)}
        result["recall_pass"] = bool(new_recall > old_recall)
    result["accepted"] = bool(result["overall_pass"] and result["control_pass"]
                              and result.get("recall_pass", True))
    return result


def clean_rejected(run: Path) -> dict:
    owned = ROOT / "outputs" / "local_strategy_campaign_20261119_01"
    if run.parent.resolve() != owned.resolve() or run.is_symlink():
        raise ValueError("Cleanup is restricted to campaign-owned experiment directories")
    report = json.loads((run / "report.json").read_text())
    if report["accepted"]:
        raise ValueError("Accepted experiments must retain their outputs")
    removed = []
    for path in run.iterdir():
        if path.name in KEEP:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected cleanup target: {path}")
        removed.append({"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    for item in removed:
        (run / item["name"]).unlink()
    receipt = {"run": run.name, "removed": removed, "retained": sorted(p.name for p in run.iterdir())}
    (run / "cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol_path, rows_path = run / "protocol.json", run / "rows.jsonl"
    protocol = json.loads(protocol_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    result = assess(protocol, rows)
    result["input_sha256"] = {"protocol": sha(protocol_path), "rows": sha(rows_path)}
    (run / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["accepted"]:
        result["cleanup"] = clean_rejected(run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
