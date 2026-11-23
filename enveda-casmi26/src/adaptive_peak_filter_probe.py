"""Fixed-library test of relaxing query peak filters only for sparse spectra."""
import argparse
from collections import defaultdict
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_vector(row, floor, cap):
    mzs = np.asarray(row['ms2_mzs'], dtype=float)
    intensities = np.asarray(row['ms2_normalized_intensities'], dtype=float)
    precursor = row['precursor_mz']
    if mzs.shape != intensities.shape or mzs.ndim != 1:
        raise ValueError('Peak arrays must be aligned one-dimensional arrays')
    valid = (np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0) & (mzs <= 2000)
             & (mzs <= precursor + 2) & (intensities > 0)
             & (np.abs(mzs - precursor) > 1.5))
    mzs, intensities = mzs[valid], intensities[valid]
    if not len(mzs):
        return {}, 0
    keep = intensities >= floor * intensities.max()
    mzs, intensities = mzs[keep], intensities[keep]
    order = np.argsort(-intensities, kind='stable')[:cap]
    mzs, weights = mzs[order], np.sqrt(intensities[order])
    result = {}
    fragment_bins = 0
    for positions, offset, channel_weight in [(mzs, 0, .7), (precursor - mzs, b.N_BINS, .3)]:
        channel = defaultdict(float)
        for pos, weight in zip(positions, weights):
            if 0 < pos <= 2000:
                channel[int(np.floor(pos / b.BIN_WIDTH + .5))] += float(weight)
        if offset == 0:
            fragment_bins = len(channel)
        norm = np.sqrt(sum(w*w for w in channel.values()))
        if norm:
            result.update({i + offset: np.sqrt(channel_weight)*w/norm
                           for i, w in channel.items()})
    norm = np.sqrt(sum(w*w for w in result.values()))
    return ({i: w/norm for i, w in result.items()} if norm else {}), fragment_bins


def main(partition, output):
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
    metadata, matrix, _, signatures, reference_audit = p.build_reference(
        ROOT / 'data', b.read_metadata(ROOT / 'data'), query_masses)
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([i for i, signature in enumerate(signatures)
                          if signature not in query_signatures], dtype=int)
    reference_masses = np.asarray([metadata[i][2] for i in allowed])
    stored = json.loads(CONTROL.read_text())['A_binned'] if partition == 'development' else None
    predictions = {'mean': {}, 'adaptive': {}}
    group_scores = {}
    sparse_spectra = 0
    changed_spectra = 0
    for qid, rows in sorted(groups.items()):
        estimates = np.asarray([b.neutral_mass(row['precursor_mz'], row['adduct']) for row in rows])
        mass = float(np.median(estimates))
        tol = max(b.CONFIG['mass_floor_da'], mass * b.CONFIG['mass_ppm'] * 1e-6)
        eligible = allowed[np.abs(reference_masses - mass) <= tol]
        ref_keys = [metadata[i][0] for i in eligible]
        smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
        baseline_vectors, adaptive_vectors = [], []
        for row in rows:
            baseline, count = query_vector(row, .005, 64)
            if baseline != b.peak_vector(row['ms2_mzs'], row['ms2_normalized_intensities'],
                                          row['precursor_mz']):
                raise AssertionError(f'Default vector replay differs for {qid}')
            adaptive = query_vector(row, .001, 512)[0] if count < 5 else baseline
            sparse_spectra += count < 5
            changed_spectra += adaptive != baseline
            baseline_vectors.append(baseline)
            adaptive_vectors.append(adaptive)
        group_scores[qid] = {}
        for policy, vectors in [('mean', baseline_vectors), ('adaptive', adaptive_vectors)]:
            scores = (b.sparse_vectors(vectors) @ matrix[eligible].T).toarray()
            for j, row in enumerate(rows):
                scores[j, [metadata[i][3] != row['ionization_mode'] for i in eligible]] = 0.
            ranked = b.rank_candidates(scores, ref_keys, set())[:25]
            predictions[policy][qid] = [smiles[key] for key, _ in ranked]
            group_scores[qid][policy] = b.reciprocal_rank(truth[qid], predictions[policy][qid])
        if stored is not None and predictions['mean'][qid] != stored[qid]:
            raise AssertionError(f'Baseline top25 replay differs for {qid}')
    metrics = {policy: b.evaluate(truth, predictions[policy]) for policy in predictions}
    diff = np.asarray([group_scores[qid]['adaptive'] - group_scores[qid]['mean']
                       for qid in sorted(groups)])
    rng = np.random.default_rng(20261119)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    paired = {'mrr_delta': float(diff.mean()),
              'bootstrap95': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
              'improved': int((diff > 0).sum()), 'worsened': int((diff < 0).sum()),
              'changed_top25_lists': sum(predictions['mean'][qid] != predictions['adaptive'][qid]
                                         for qid in groups)}
    report = {'experiment': 'S1 fixed-reference adaptive query filtering for fewer than 5 fragment bins',
              'partition': partition, 'molecules': len(keys),
              'keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
              'input_sha256': {'manifest': sha(MANIFEST), 'train_parquet': sha(ROOT / 'data/train.parquet'),
                               'control_predictions': sha(CONTROL) if stored is not None else None,
                               'source': sha(Path(__file__)), 'baseline_source': sha(Path(b.__file__)),
                               'phase1_eval_source': sha(Path(p.__file__))},
              'query_spectra': spectrum_count, 'sparse_query_spectra': sparse_spectra,
              'changed_query_spectra': changed_spectra,
              'reference_spectra': len(allowed), 'reference_audit': reference_audit,
              'baseline_replay': f'{len(keys)}/{len(keys)} exact top25 lists' if stored is not None else None,
              'metrics': metrics, 'paired_vs_mean': paired, 'elapsed_seconds': time.monotonic()-start,
              'limitation': 'S1 reference structures are available; this does not test unseen structures.'}
    output.mkdir(exist_ok=False)
    b.write_json(output / 'report.json', report)
    b.write_json(output / 'group_scores.json', group_scores)
    print(json.dumps({'metrics': metrics, 'paired': paired, 'sparse': sparse_spectra,
                      'changed': changed_spectra, 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=['development', 'independent'], default='development')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.partition, args.output)
