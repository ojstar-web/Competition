"""Compare query-spectrum aggregation on fresh-development S1 splits."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import baseline as b
import phase1_eval as p


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs/phase1_20261023_01/split_manifest.json'
CONTROL = ROOT / 'outputs/peak_filter_ab_20261119_fresh_control_01/predictions.json'
OUTPUT = ROOT / 'outputs/spectrum_aggregation_20261119_01'
POLICIES = ('mean', 'median', 'top_half', 'max', 'balanced_mode', 'balanced_adduct')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(evidence, rows, policy):
    if policy == 'mean':
        return evidence.mean(axis=0)
    if policy == 'median':
        return np.median(evidence, axis=0)
    if policy == 'top_half':
        count = (len(rows) + 1) // 2
        return np.sort(evidence, axis=0)[-count:].mean(axis=0)
    if policy == 'max':
        return evidence.max(axis=0)
    field = 'ionization_mode' if policy == 'balanced_mode' else 'adduct'
    groups = sorted({row[field] for row in rows})
    return np.mean([evidence[[row[field] == group for row in rows]].mean(axis=0)
                    for group in groups], axis=0)


def main(partition='development', output=OUTPUT):
    start = time.monotonic()
    if sha(MANIFEST) != (MANIFEST.parent / 'split_manifest.sha256').read_text().split()[0]:
        raise ValueError('Frozen split manifest SHA mismatch')
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit 2026.03.3 required')
    manifest = json.loads(MANIFEST.read_text())
    primary = manifest['primary_fresh_dev_keys'][:64]
    keys = (primary if partition == 'development' else
            [key for key in manifest['partitions']['fresh_dev'] if key not in set(primary)])
    if set(keys) & (set(manifest['partitions']['legacy_dev']) | set(manifest['partitions']['sealed'])):
        raise ValueError('Fresh split overlaps exposed or sealed structures')
    groups, truth, spectrum_count = p.load_fresh_queries(ROOT / 'data', keys)
    query_masses = [b.neutral_mass(row['precursor_mz'], row['adduct'])
                    for rows in groups.values() for row in rows]
    train = b.read_metadata(ROOT / 'data')
    metadata, matrix, _, signatures, reference_audit = p.build_reference(
        ROOT / 'data', train, query_masses)
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([i for i, signature in enumerate(signatures)
                          if signature not in query_signatures], dtype=int)
    stored = json.loads(CONTROL.read_text())['A_binned'] if partition == 'development' else None
    predictions = {policy: {} for policy in POLICIES}
    group_scores = {}
    for qid, rows in sorted(groups.items()):
        estimates = np.asarray([b.neutral_mass(row['precursor_mz'], row['adduct']) for row in rows])
        mass = float(np.median(estimates))
        tol = max(b.CONFIG['mass_floor_da'], mass * b.CONFIG['mass_ppm'] * 1e-6)
        eligible = allowed[np.abs(np.asarray([metadata[i][2] for i in allowed]) - mass) <= tol]
        ref_keys = [metadata[i][0] for i in eligible]
        smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
        vectors = b.sparse_vectors([b.peak_vector(row['ms2_mzs'],
                    row['ms2_normalized_intensities'], row['precursor_mz']) for row in rows])
        scores = (vectors @ matrix[eligible].T).toarray()
        for j, row in enumerate(rows):
            scores[j, [metadata[i][3] != row['ionization_mode'] for i in eligible]] = 0.
        keys_in_pool = sorted(set(ref_keys))
        index = {key: j for j, key in enumerate(keys_in_pool)}
        evidence = np.zeros((len(rows), len(keys_in_pool)), dtype=float)
        for j, key in enumerate(ref_keys):
            np.maximum(evidence[:, index[key]], scores[:, j], out=evidence[:, index[key]])
        group_scores[qid] = {}
        for policy in POLICIES:
            values = aggregate(evidence, rows, policy)
            order = np.lexsort((np.asarray(keys_in_pool), -values))[:25]
            ranked = [smiles[keys_in_pool[i]] for i in order]
            predictions[policy][qid] = ranked
            group_scores[qid][policy] = b.reciprocal_rank(truth[qid], ranked)
        if stored is not None and predictions['mean'][qid] != stored[qid]:
            raise AssertionError(f'Baseline replay differs for {qid}')
    metrics = {policy: b.evaluate(truth, predictions[policy]) for policy in POLICIES}
    base = np.asarray([group_scores[qid]['mean'] for qid in sorted(groups)])
    paired = {}
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(base), size=(10000, len(base)))
    for policy in POLICIES[1:]:
        diff = np.asarray([group_scores[qid][policy] for qid in sorted(groups)]) - base
        paired[policy] = {'mrr_delta': float(diff.mean()),
                          'bootstrap95': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                          'improved': int((diff > 0).sum()), 'worsened': int((diff < 0).sum())}
    report = {'experiment': 'S1 fresh-development fixed mass pool, binned retrieval',
              'partition': {'name': partition, 'molecules': len(keys),
                            'keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest()},
              'input_sha256': {'manifest': sha(MANIFEST),
                               'control_predictions': sha(CONTROL) if stored is not None else None,
                               'train_parquet': sha(ROOT / 'data/train.parquet'),
                               'source': sha(Path(__file__)), 'baseline_source': sha(Path(b.__file__)),
                               'phase1_eval_source': sha(Path(p.__file__))},
              'query_structures': len(groups), 'query_spectra': spectrum_count,
              'reference_spectra': len(allowed), 'reference_audit': reference_audit,
              'baseline_replay': f'{len(keys)}/{len(keys)} exact top25 lists' if stored is not None else None,
              'metrics': metrics,
              'paired_vs_mean': paired, 'elapsed_seconds': time.monotonic() - start,
              'limitation': 'S1 queries have reference structures; this does not measure unseen-structure performance.'}
    output.mkdir(exist_ok=False)
    b.write_json(output / 'report.json', report)
    b.write_json(output / 'group_scores.json', group_scores)
    b.write_json(output / 'predictions.json', predictions)
    print(json.dumps({'metrics': metrics, 'paired_vs_mean': paired,
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=['development', 'independent'], default='development')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.partition, args.output)
