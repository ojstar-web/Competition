"""Replay the confirm-selected GLACIER-only weight on frozen final controls."""
from __future__ import annotations

import json
from pathlib import Path

from bond_fragment_probe import ROOT, sha
from orthogonal_weight_probe import GRID, OUT as PRIOR_OUT, load_rows, summarize


OUT = ROOT / "outputs/glacier_weight_20261119_01"


def main() -> None:
    confirm_path = PRIOR_OUT / "confirm.json"
    confirm = json.loads(confirm_path.read_text())
    if confirm["source_sha256"] != sha(Path(load_rows.__code__.co_filename)):
        raise ValueError("Frozen confirm-grid source changed")
    choices = [row for row in confirm["grid"] if row["weights"][1] == 0.0]
    if len(choices) != len([pair for pair in GRID if pair[1] == 0.0]):
        raise ValueError("Incomplete confirm GLACIER-only grid")
    winner = max(choices, key=lambda row: (row["mrr"], -row["weights"][0]))
    selected = tuple(winner["weights"])
    if selected != (0.75, 0.0):
        raise ValueError("Confirm GLACIER-only selection changed")
    rows, inputs = load_rows("final")
    result = {
        "partition": "final", "selected_weights": list(selected),
        "selection_rule": "max confirm MRR among ICEBERG=0, tie: lower GLACIER weight",
        "confirm_mrr": winner["mrr"], "comparison": summarize(rows, selected),
        "inputs_sha256": {**inputs, "confirm_grid": sha(confirm_path)},
        "source_sha256": sha(Path(__file__)),
        "scope": "Frozen pool-control top25; final partition was viewed in earlier GLACIER experiments."
    }
    OUT.mkdir(exist_ok=False)
    (OUT / "final.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
