"""Probe reference collision-energy matching on frozen S1 retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

import baseline as b
import phase1_eval as p


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs/phase1_20261023_01/split_manifest.json'
CONTROL = ROOT / 'outputs/peak_filter_ab_20261119_fresh_control_01/predictions.json'
OUTPUT = ROOT / 'outputs/reference_energy_20261119_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(partition='development', output=OUTPUT):
    start = time.monotonic()
    if sha(MANIFEST) != (MANIFEST.parent / 'split_manifest.sha256').read_text().split()[0]:
        raise ValueError('Frozen split manifest SHA mismatch')
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit 2026.03.3 required')
    manifest = json.loads(MANIFEST.read_text())
    primary = manifest['primary_fresh_dev_keys']
    keys = (primary if partition == 'development' else
            [key for key in manifest['partitions']['fresh_dev'] if key not in set(primary)])
    if not keys or set(keys) & set(manifest['partitions']['sealed']):
        raise ValueError('Requested partition is incomplete or overlaps sealed structures')
    if partition == 'independent' and set(keys) & set(primary):
        raise ValueError('Independent confirmation overlaps prior development structures')
    groups, truth, spectrum_count = p.load_fresh_queries(ROOT / 'data', keys)
    key_to_qid = {key: f'q{i:03}' for i, key in enumerate(keys)}
    energies = {qid: [] for qid in groups}
    for batch in pq.ParquetFile(ROOT / 'data/train.parquet').iter_batches(batch_size=4096):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        selected = [i for i, source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not selected:
            continue
        for row in batch.take(selected).to_pylist():
            key = b.canonical_key(row['normalized_smiles'])
            if key not in key_to_qid or row['adduct'] not in b.ADDUCTS:
                continue
            qid = key_to_qid[key]
            energy = row['collision_energy_ev']
            energies[qid].append(float(np.median(energy)) if energy else None)
    if any(len(energies[qid]) != len(rows) for qid, rows in groups.items()):
        raise ValueError('Collision-energy rows do not align with frozen queries')
    query_masses = [b.neutral_mass(row['precursor_mz'], row['adduct'])
                    for rows in groups.values() for row in rows]
    train = b.read_metadata(ROOT / 'data')
    metadata, matrix, raw, signatures, reference_audit = p.build_reference(
        ROOT / 'data', train, query_masses)
    reference_energies = [float(np.median(row['collision_energy_ev']))
                          if row['collision_energy_ev'] else None for row in raw]
    if len(reference_energies) != len(metadata) or len(reference_energies) != matrix.shape[0]:
        raise AssertionError('Reference energy provenance does not align with feature rows')
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([i for i, sig in enumerate(signatures) if sig not in query_signatures])
    allowed_masses = np.asarray([metadata[i][2] for i in allowed])
    stored = json.loads(CONTROL.read_text())['A_binned'] if partition == 'development' else None
    policies = ('mean', 'match_10ev', 'match_20ev', 'match_40ev')
    predictions = {policy: {} for policy in policies}
    ranks = {}
    energy_counts = {}
    for qid, rows in sorted(groups.items()):
        estimates = [b.neutral_mass(row['precursor_mz'], row['adduct']) for row in rows]
        mass = float(np.median(estimates))
        tol = max(b.CONFIG['mass_floor_da'], mass * b.CONFIG['mass_ppm'] * 1e-6)
        eligible = allowed[np.abs(allowed_masses - mass) <= tol]
        ref_keys = [metadata[i][0] for i in eligible]
        smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
        vectors = b.sparse_vectors([b.peak_vector(row['ms2_mzs'],
                    row['ms2_normalized_intensities'], row['precursor_mz']) for row in rows])
        scores = (vectors @ matrix[eligible].T).toarray()
        for j, row in enumerate(rows):
            scores[j, [metadata[i][3] != row['ionization_mode'] for i in eligible]] = 0.
        pool_keys = sorted(set(ref_keys))
        index = {key: j for j, key in enumerate(pool_keys)}
        evidence = {policy: np.zeros((len(rows), len(pool_keys)), dtype=float)
                    for policy in policies}
        labels = energies[qid]
        energy_counts[qid] = len(set(labels))
        for j, key in enumerate(ref_keys):
            col = index[key]
            np.maximum(evidence['mean'][:, col], scores[:, j], out=evidence['mean'][:, col])
            ref_energy = reference_energies[eligible[j]]
            for tau in (10, 20, 40):
                weights = np.asarray([np.exp(-abs(label-ref_energy)/tau)
                                      if label is not None and ref_energy is not None else 1.
                                      for label in labels])
                policy = f'match_{tau}ev'
                np.maximum(evidence[policy][:, col], scores[:, j] * weights,
                           out=evidence[policy][:, col])
        values = {policy: score.mean(axis=0) for policy, score in evidence.items()}
        ranks[qid] = {}
        for policy, score in values.items():
            order = np.lexsort((np.asarray(pool_keys), -score))[:25]
            ranked = [smiles[pool_keys[i]] for i in order]
            predictions[policy][qid] = ranked
            ranks[qid][policy] = b.reciprocal_rank(truth[qid], ranked)
        if stored is not None and predictions['mean'][qid] != stored[qid]:
            raise AssertionError(f'Baseline replay differs for {qid}')
    metrics = {policy: b.evaluate(truth, predictions[policy]) for policy in policies}
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(groups), size=(10000, len(groups)))
    paired = {}
    for policy in policies[1:]:
        diff = np.asarray([ranks[qid][policy] - ranks[qid]['mean'] for qid in sorted(groups)])
        paired[policy] = {'mrr_delta': float(diff.mean()),
                          'bootstrap95': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                          'improved': int((diff > 0).sum()), 'worsened': int((diff < 0).sum())}
    report = {'experiment': 'S1 fresh-development fixed mass pool, reference-energy matching',
              'partition': {'name': partition, 'molecules': len(keys),
                            'keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest()},
              'input_sha256': {'manifest': sha(MANIFEST),
                               'control_predictions': sha(CONTROL) if stored is not None else None,
                               'train_parquet': sha(ROOT / 'data/train.parquet'),
                               'source': sha(Path(__file__)), 'baseline_source': sha(Path(b.__file__)),
                               'phase1_eval_source': sha(Path(p.__file__))},
              'query_structures': len(groups), 'query_spectra': spectrum_count,
              'reference_spectra': len(allowed), 'reference_audit': reference_audit,
              'energy_group_counts': energy_counts,
              'reference_energy_available': sum(value is not None for value in reference_energies),
              'baseline_replay': f'{len(keys)}/{len(keys)} exact top25 lists' if stored is not None else None,
              'metrics': metrics, 'paired_vs_mean': paired,
              'elapsed_seconds': time.monotonic() - start,
              'limitation': 'Exploratory S1 only. Energy labels are row-median values; missing energy has unit weight. Independent structure-disjoint confirmation is required.'}
    output.mkdir(exist_ok=False)
    b.write_json(output / 'report.json', report)
    b.write_json(output / 'ranks.json', ranks)
    b.write_json(output / 'predictions.json', predictions)
    print(json.dumps({'metrics': metrics, 'paired_vs_mean': paired,
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=['development', 'independent'], default='development')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.partition, args.output)
