"""Test reference-spectrum peak frequency weighting on the frozen S1 library."""
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
POLICIES = {'baseline': 0., 'freq_quarter': .25, 'freq_half': .5}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reweight(matrix, weights):
    weighted = matrix.multiply(weights).tocsr()
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    if np.any(norms == 0):
        raise ValueError('Peak weighting yielded an empty vector')
    return weighted.multiply((1. / norms)[:, None]).tocsr()


def main():
    start = time.monotonic()
    if sha(MANIFEST) != (MANIFEST.parent / 'split_manifest.sha256').read_text().split()[0]:
        raise ValueError('Frozen split manifest SHA mismatch')
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit 2026.03.3 required')
    manifest = json.loads(MANIFEST.read_text())
    keys = manifest['primary_fresh_dev_keys'][:64]
    if set(keys) & (set(manifest['partitions']['legacy_dev']) | set(manifest['partitions']['sealed'])):
        raise ValueError('Fresh split overlaps exposed or sealed structures')
    groups, truth, spectrum_count = p.load_fresh_queries(ROOT / 'data', keys)
    query_masses = [b.neutral_mass(row['precursor_mz'], row['adduct'])
                    for rows in groups.values() for row in rows]
    metadata, matrix, _, signatures, reference_audit = p.build_reference(
        ROOT / 'data', b.read_metadata(ROOT / 'data'), query_masses)
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([i for i, signature in enumerate(signatures)
                          if signature not in query_signatures], dtype=int)
    frequency = np.asarray((matrix[allowed] != 0).sum(axis=0)).ravel()
    structures = len({metadata[i][0] for i in allowed})
    reference_masses = np.asarray([metadata[i][2] for i in allowed])
    stored = json.loads(CONTROL.read_text())['A_binned']
    predictions = {policy: {} for policy in POLICIES}
    group_scores = {}
    for qid, rows in sorted(groups.items()):
        estimates = np.asarray([b.neutral_mass(row['precursor_mz'], row['adduct']) for row in rows])
        mass = float(np.median(estimates))
        tol = max(b.CONFIG['mass_floor_da'], mass * b.CONFIG['mass_ppm'] * 1e-6)
        eligible = allowed[np.abs(reference_masses - mass) <= tol]
        ref_keys = [metadata[i][0] for i in eligible]
        smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
        query = b.sparse_vectors([b.peak_vector(row['ms2_mzs'], row['ms2_normalized_intensities'],
                                                row['precursor_mz']) for row in rows])
        group_scores[qid] = {}
        for policy, exponent in POLICIES.items():
            if exponent == 0:
                scores = (query @ matrix[eligible].T).toarray()
            else:
                weights = ((frequency + 1.) / (len(allowed) + 1.)) ** exponent
                scores = (reweight(query, weights) @ reweight(matrix[eligible], weights).T).toarray()
            for j, row in enumerate(rows):
                scores[j, [metadata[i][3] != row['ionization_mode'] for i in eligible]] = 0.
            ranked = b.rank_candidates(scores, ref_keys, set())[:25]
            predictions[policy][qid] = [smiles[key] for key, _ in ranked]
            group_scores[qid][policy] = b.reciprocal_rank(truth[qid], predictions[policy][qid])
        if predictions['baseline'][qid] != stored[qid]:
            raise AssertionError(f'Baseline top25 replay differs for {qid}')
    metrics = {policy: b.evaluate(truth, predictions[policy]) for policy in POLICIES}
    paired = {}
    for policy in ('freq_quarter', 'freq_half'):
        diff = np.asarray([group_scores[qid][policy] - group_scores[qid]['baseline']
                           for qid in sorted(groups)])
        rng = np.random.default_rng(20261119)
        draws = rng.integers(0, len(diff), size=(10000, len(diff)))
        paired[policy] = {'mrr_delta': float(diff.mean()),
                          'bootstrap95': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                          'improved': int((diff > 0).sum()), 'worsened': int((diff < 0).sum()),
                          'changed_top25_lists': sum(predictions[policy][qid] != predictions['baseline'][qid]
                                                     for qid in groups)}
    report = {'experiment': 'S1 fixed-reference spectrum-frequency fragment/loss weighted cosine',
              'molecules': len(keys), 'query_spectra': spectrum_count,
              'keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
              'input_sha256': {'manifest': sha(MANIFEST), 'train_parquet': sha(ROOT / 'data/train.parquet'),
                               'control_predictions': sha(CONTROL), 'source': sha(Path(__file__)),
                               'baseline_source': sha(Path(b.__file__)),
                               'phase1_eval_source': sha(Path(p.__file__))},
              'reference_spectra': len(allowed), 'reference_structures': structures,
              'feature_frequency': {'nonzero_features': int((frequency > 0).sum()),
                                    'max_spectrum_frequency': int(frequency.max()),
                                    'smoothing': '(frequency + 1) / (reference spectra + 1)'},
              'reference_audit': reference_audit, 'policies': POLICIES,
              'baseline_replay': f'{len(keys)}/{len(keys)} exact top25 lists',
              'metrics': metrics, 'paired_vs_baseline': paired,
              'elapsed_seconds': time.monotonic() - start,
              'limitation': 'S1 reference structures are available; this does not test unseen structures.'}
    output = ROOT / 'outputs/fragment_frequency_20261119_dev_01'
    output.mkdir(exist_ok=False)
    b.write_json(output / 'report.json', report)
    b.write_json(output / 'group_scores.json', group_scores)
    print(json.dumps({'metrics': metrics, 'paired': paired,
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
