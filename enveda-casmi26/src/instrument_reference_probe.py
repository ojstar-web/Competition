"""Probe instrument-matched S1 reference scoring on fixed candidates."""
from collections import Counter, defaultdict
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
BOOSTS = {'baseline': 1.0, 'same_instrument_1_25': 1.25, 'same_instrument_2': 2.0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_instruments(data, train, query_masses, signatures):
    masses = np.full(len(train['adduct']), np.nan)
    for i, (mz, adduct) in enumerate(zip(train['precursor_mz'], train['adduct'])):
        if adduct in b.ADDUCTS and mz is not None and np.isfinite(mz) and mz > 0:
            masses[i] = b.neutral_mass(mz, adduct)
    targets = np.sort(np.asarray(query_masses))
    pos = np.searchsorted(targets, masses)
    lo = targets[np.clip(pos - 1, 0, len(targets) - 1)]
    hi = targets[np.clip(pos, 0, len(targets) - 1)]
    tolerance = np.maximum(.006, masses * b.CONFIG['prefilter_ppm'] * 1e-6)
    keep = (np.minimum(abs(masses - lo), abs(masses - hi)) <= tolerance) & np.isfinite(masses)
    keep &= np.asarray(train['ingest_lib']) != b.CONFIG['validation_source']
    wanted = set(signatures)
    found = defaultdict(set)
    start = 0
    for batch in pq.ParquetFile(data / 'train.parquet').iter_batches(batch_size=4096):
        local = np.flatnonzero(keep[start:start + batch.num_rows])
        start += batch.num_rows
        if not len(local):
            continue
        for row in batch.take(local).to_pylist():
            signature = p.spectrum_signature(row)
            if signature in wanted:
                found[signature].add(str(row['instrument_type']))
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'{len(missing)} reference signatures have no instrument metadata')
    ambiguous = sum(len(found[signature]) > 1 for signature in wanted)
    labels = [next(iter(found[signature])) if len(found[signature]) == 1 else None
              for signature in signatures]
    return labels, ambiguous


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
    train = b.read_metadata(ROOT / 'data')
    metadata, matrix, _, signatures, reference_audit = p.build_reference(
        ROOT / 'data', train, query_masses)
    labels, ambiguous = reference_instruments(ROOT / 'data', train, query_masses, signatures)
    query_signatures = {p.spectrum_signature(row) for rows in groups.values() for row in rows}
    allowed = np.asarray([i for i, signature in enumerate(signatures)
                          if signature not in query_signatures], dtype=int)
    reference_masses = np.asarray([metadata[i][2] for i in allowed])
    stored = json.loads(CONTROL.read_text())['A_binned']
    predictions = {policy: {} for policy in BOOSTS}
    group_scores = {}
    matching_refs = sum(labels[i] == 'timsTOF' for i in allowed)
    for qid, rows in sorted(groups.items()):
        estimates = np.asarray([b.neutral_mass(row['precursor_mz'], row['adduct']) for row in rows])
        mass = float(np.median(estimates))
        tol = max(b.CONFIG['mass_floor_da'], mass * b.CONFIG['mass_ppm'] * 1e-6)
        eligible = allowed[np.abs(reference_masses - mass) <= tol]
        ref_keys = [metadata[i][0] for i in eligible]
        smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
        same = np.asarray([labels[i] == 'timsTOF' for i in eligible])
        vectors = [b.peak_vector(row['ms2_mzs'], row['ms2_normalized_intensities'],
                                 row['precursor_mz']) for row in rows]
        scores = (b.sparse_vectors(vectors) @ matrix[eligible].T).toarray()
        for j, row in enumerate(rows):
            scores[j, [metadata[i][3] != row['ionization_mode'] for i in eligible]] = 0.
        group_scores[qid] = {}
        for policy, boost in BOOSTS.items():
            weighted = scores if boost == 1.0 else scores * np.where(same, boost, 1.)[None, :]
            ranked = b.rank_candidates(weighted, ref_keys, set())[:25]
            predictions[policy][qid] = [smiles[key] for key, _ in ranked]
            group_scores[qid][policy] = b.reciprocal_rank(truth[qid], predictions[policy][qid])
        if predictions['baseline'][qid] != stored[qid]:
            raise AssertionError(f'Baseline top25 replay differs for {qid}')
    metrics = {policy: b.evaluate(truth, predictions[policy]) for policy in BOOSTS}
    paired = {}
    for policy in BOOSTS:
        if policy == 'baseline':
            continue
        diff = np.asarray([group_scores[qid][policy] - group_scores[qid]['baseline']
                           for qid in sorted(groups)])
        rng = np.random.default_rng(20261119)
        draws = rng.integers(0, len(diff), size=(10000, len(diff)))
        paired[policy] = {'mrr_delta': float(diff.mean()),
                          'bootstrap95': np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                          'improved': int((diff > 0).sum()), 'worsened': int((diff < 0).sum()),
                          'changed_top25_lists': sum(predictions[policy][qid] != predictions['baseline'][qid]
                                                     for qid in groups)}
    report = {'experiment': 'S1 fixed-reference same-instrument spectrum boost',
              'molecules': len(keys), 'query_spectra': spectrum_count,
              'keys_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
              'input_sha256': {'manifest': sha(MANIFEST), 'train_parquet': sha(ROOT / 'data/train.parquet'),
                               'control_predictions': sha(CONTROL), 'source': sha(Path(__file__)),
                               'baseline_source': sha(Path(b.__file__)),
                               'phase1_eval_source': sha(Path(p.__file__))},
              'reference_spectra': len(allowed), 'matching_reference_spectra': matching_refs,
              'ambiguous_reference_signatures': ambiguous, 'reference_instruments': dict(Counter(labels)),
              'reference_audit': reference_audit, 'boosts': BOOSTS,
              'baseline_replay': f'{len(keys)}/{len(keys)} exact top25 lists',
              'metrics': metrics, 'paired_vs_baseline': paired,
              'elapsed_seconds': time.monotonic() - start,
              'limitation': 'S1 reference structures are available; this does not test unseen structures.'}
    output = ROOT / 'outputs/instrument_reference_20261119_dev_01'
    output.mkdir(exist_ok=False)
    b.write_json(output / 'report.json', report)
    b.write_json(output / 'group_scores.json', group_scores)
    print(json.dumps({'metrics': metrics, 'paired': paired, 'matching_refs': matching_refs,
                      'reference_spectra': len(allowed), 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
