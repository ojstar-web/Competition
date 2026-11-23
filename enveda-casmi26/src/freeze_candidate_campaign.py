"""Freeze unused fold1 identity groups before local candidate-policy scoring."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/fold1_ood_retrieval_20261110_01"
OUT = ROOT / "outputs/local_strategy_campaign_20261119_01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection_path = SOURCE / "SELECTION.json"
    labels_path = SOURCE / "scoring_labels_corrected.parquet"
    pilot_path = SOURCE / "PILOT_QUERY_IDS.json"
    old_confirm_path = SOURCE / "CONFIRM_QUERY_IDS.json"
    selection = json.loads(selection_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    old_confirm = json.loads(old_confirm_path.read_text())
    labels = pq.read_table(labels_path).to_pylist()
    by_group = {}
    for row in labels:
        by_group.setdefault(int(row["identity_group"]), []).append(int(row["query_id"]))
    used = set(selection["pilot_groups_32"]) | set(old_confirm["ood_groups"]) | set(old_confirm["control_groups"])
    assert not set(selection["pilot_groups_32"]) & (set(old_confirm["ood_groups"]) | set(old_confirm["control_groups"]))
    assert set(pilot["query_ids"]) <= {q for g in selection["pilot_groups_32"] for q in by_group[g]}
    pools = {arm: [int(g) for g in selection[f"{arm}_groups_1024"] if g not in used]
             for arm in ("ood", "control")}
    counts = {"design": 32, "confirm": 128, "final": 128}
    if any(len(pool) < sum(counts.values()) for pool in pools.values()):
        raise ValueError("Not enough unused groups")
    frozen = {"scope": "fold1 structure-disjoint simulated OOD; artificial balanced arms",
              "sources_sha256": {p.name: sha(p) for p in (selection_path, labels_path, pilot_path, old_confirm_path)},
              "prior_groups": sorted(used), "partitions": {}}
    offset = 0
    for name, size in counts.items():
        groups = {arm: pools[arm][offset:offset + size] for arm in pools}
        offset += size
        ids = sorted(q for group in groups.values() for g in group for q in by_group[g])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate query ID")
        frozen["partitions"][name] = {"groups": groups, "query_ids": ids}
    all_groups = [g for part in frozen["partitions"].values() for arm in part["groups"].values() for g in arm]
    assert len(all_groups) == len(set(all_groups)) and not set(all_groups) & used
    OUT.mkdir(exist_ok=False)
    (OUT / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for name, part in frozen["partitions"].items():
        (OUT / f"{name}_ids.json").write_text(json.dumps({"query_ids": part["query_ids"]}, indent=2) + "\n")
    print(json.dumps({name: {"groups_per_arm": len(part["groups"]["ood"]),
                              "queries": len(part["query_ids"])}
                      for name, part in frozen["partitions"].items()}))


if __name__ == "__main__":
    main()
