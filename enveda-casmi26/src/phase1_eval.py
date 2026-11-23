"""Phase-one A/B spectral retrieval on a frozen fresh-development split."""
from array import array
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
import baseline as b
from phase1 import greedy_similarity


def spectrum_signature(row):
    h = hashlib.sha256()
    h.update(str(row['adduct']).encode())
    h.update(str(row['ionization_mode']).encode())
    h.update(np.asarray([row['precursor_mz']], dtype=np.float64).tobytes())
    h.update(np.asarray(row['ms2_mzs'], dtype=np.float64).tobytes())
    h.update(np.asarray(row['ms2_normalized_intensities'], dtype=np.float64).tobytes())
    return h.digest()


def load_fresh_queries(data, keys, include_truth=True):
    selected = set(keys)
    groups = defaultdict(list)
    truth = {}
    spectra = 0
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i, source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            key = b.canonical_key(row['normalized_smiles'])
            if key not in selected or row['adduct'] not in b.ADDUCTS:
                continue
            qid = f'q{keys.index(key):03}'
            if include_truth:
                truth[qid] = row['normalized_smiles']
            # The scoring path receives only fields available in test.parquet.
            query = {k: row[k] for k in ['ms2_mzs', 'ms2_normalized_intensities',
                     'adduct', 'ionization_mode', 'precursor_mz']}
            b.neutral_mass(query['precursor_mz'], query['adduct'])
            b.peak_vector(query['ms2_mzs'], query['ms2_normalized_intensities'], query['precursor_mz'])
            groups[qid].append(query)
            spectra += 1
    if len(groups) != len(keys):
        raise ValueError(f'Only {len(groups)}/{len(keys)} fresh-development molecules have usable spectra')
    return dict(groups), truth, spectra


def exact_channels(row):
    mzs = np.asarray(row['ms2_mzs'], dtype=float)
    ints = np.asarray(row['ms2_normalized_intensities'], dtype=float)
    precursor = row['precursor_mz']
    valid = (np.isfinite(mzs) & np.isfinite(ints) & (mzs > 0) & (mzs <= 2000)
             & (mzs <= precursor+2) & (ints > 0) & (abs(mzs-precursor) > 1.5))
    mzs, ints = mzs[valid], ints[valid]
    if not len(mzs):
        return [], []
    keep = ints >= b.CONFIG['intensity_floor']*ints.max()
    mzs, ints = mzs[keep], ints[keep]
    order = np.argsort(-ints, kind='stable')[:b.CONFIG['top_peaks']]
    mzs, weights = mzs[order], np.sqrt(ints[order])
    frag = [(float(mz), float(w)) for mz, w in zip(mzs, weights) if 0 < mz <= 2000]
    loss = [(float(precursor-mz), float(w)) for mz, w in zip(mzs, weights)
            if 0 < precursor-mz <= 2000]
    return frag, loss


def exact_score(query_channels, reference_channels):
    qfrag, qloss = query_channels
    rfrag, rloss = reference_channels
    qw = (.7 if qfrag and qloss else 1.) if qfrag else 0.
    rw = (.7 if rfrag and rloss else 1.) if rfrag else 0.
    qlw = 1.-qw if qloss else 0.
    rlw = 1.-rw if rloss else 0.
    frag = greedy_similarity(qfrag, rfrag) if qfrag and rfrag else 0.
    loss = greedy_similarity(qloss, rloss) if qloss and rloss else 0.
    return float(np.sqrt(qw*rw)*frag + np.sqrt(qlw*rlw)*loss)


def build_reference(data, train, query_masses):
    n = len(train['adduct'])
    masses = np.full(n, np.nan)
    unknown = Counter()
    for i, (mz, adduct) in enumerate(zip(train['precursor_mz'], train['adduct'])):
        if adduct in b.ADDUCTS and mz is not None and np.isfinite(mz) and mz > 0:
            masses[i] = b.neutral_mass(mz, adduct)
        else:
            unknown[str(adduct)] += 1
    targets = np.sort(np.asarray(query_masses))
    pos = np.searchsorted(targets, masses)
    lo = targets[np.clip(pos-1, 0, len(targets)-1)]
    hi = targets[np.clip(pos, 0, len(targets)-1)]
    distance = np.minimum(abs(masses-lo), abs(masses-hi))
    tolerance = np.maximum(.006, masses*b.CONFIG['prefilter_ppm']*1e-6)
    keep = (distance <= tolerance) & np.isfinite(masses)
    keep &= np.asarray(train['ingest_lib']) != b.CONFIG['validation_source']
    metadata, raw, signatures = [], [], []
    skipped, counts = Counter(), Counter()
    feature_indices, feature_values, indptr = array('i'), array('f'), array('q', [0])
    start = 0
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096):
        local = np.flatnonzero(keep[start:start+batch.num_rows])
        start += batch.num_rows
        for row in batch.take(local).to_pylist():
            info = b.structure_info(row['normalized_smiles'])
            if not info:
                skipped['invalid_charged_or_multicomponent_structure'] += 1
                continue
            if b.ADDUCTS[row['adduct']][:2] != (1, 1):
                skipped['non_monomer_or_multicharged'] += 1
                continue
            vector = b.peak_vector(row['ms2_mzs'], row['ms2_normalized_intensities'], row['precursor_mz'])
            if not vector:
                skipped['no_usable_peaks'] += 1
                continue
            identity_adduct = (info[0], row['adduct'])
            if counts[identity_adduct] >= b.CONFIG['max_spectra_per_structure_adduct']:
                skipped['spectrum_cap'] += 1
                continue
            counts[identity_adduct] += 1
            if len(metadata) >= b.CONFIG['max_reference_spectra']:
                raise RuntimeError('Accepted reference spectrum budget exceeded')
            metadata.append((info[0], row['normalized_smiles'], info[1], row['ionization_mode']))
            raw.append({k: row[k] for k in ['ms2_mzs', 'ms2_normalized_intensities',
                       'precursor_mz', 'collision_energy_ev']})
            signatures.append(spectrum_signature(row))
            for feature, value in sorted(vector.items()):
                feature_indices.append(feature)
                feature_values.append(value)
            indptr.append(len(feature_indices))
    if not metadata:
        raise RuntimeError('No reference spectra')
    matrix = csr_matrix((np.frombuffer(feature_values, dtype=np.float32),
                         np.frombuffer(feature_indices, dtype=np.int32),
                         np.frombuffer(indptr, dtype=np.int64)),
                        shape=(len(metadata), b.N_FEATURES))
    return metadata, matrix, raw, signatures, {
        'prefilter_rows': int(keep.sum()), 'reference_spectra': len(metadata),
        'reference_structures': len({x[0] for x in metadata}),
        'unsupported_adducts': dict(unknown), 'skipped': dict(skipped)}


def rank_a_keys_for_pool(rows, metadata, matrix, allowed, pool):
    """Score the existing binned A method within one frozen external mass pool."""
    estimates = np.asarray([b.neutral_mass(r['precursor_mz'], r['adduct']) for r in rows])
    mass = float(np.median(estimates))
    tol = max(b.CONFIG['mass_floor_da'], mass*b.CONFIG['mass_ppm']*1e-6)
    pool_masses = {c['key']: c['exact_mass'] for c in pool}
    eligible = np.asarray([i for i in allowed
                           if abs(metadata[i][2]-mass) <= tol
                           and metadata[i][0] in pool_masses
                           and abs(pool_masses[metadata[i][0]]-mass) <= tol], dtype=int)
    if not len(eligible):
        return [], {'mass': mass, 'reference_spectra_in_pool': 0,
                    'reference_structures_in_pool': 0, 'reference_keys_in_pool': [], 'mode_qualified_reference_keys': [], 'positive_structures': 0}
    keys = [metadata[i][0] for i in eligible]
    q = b.sparse_vectors([b.peak_vector(r['ms2_mzs'], r['ms2_normalized_intensities'],
                                       r['precursor_mz']) for r in rows])
    scores = (q @ matrix[eligible].T).toarray()
    for j, row in enumerate(rows):
        mismatch = np.asarray([metadata[i][3] != row['ionization_mode'] for i in eligible])
        scores[j, mismatch] = 0.
    ranked = [(key, score) for key, score in b.rank_candidates(scores, keys, set()) if score > 0]
    return ranked, {'mass': mass, 'reference_spectra_in_pool': len(eligible),
                    'reference_structures_in_pool': len(set(keys)),
                    'reference_keys_in_pool': sorted(set(keys)),
                    'mode_qualified_reference_keys': sorted({metadata[i][0] for i in eligible
                       if any(metadata[i][3] == row['ionization_mode'] for row in rows)}),
                    'positive_structures': len(ranked)}

def rank_one(rows, metadata, matrix, raw, allowed, shortlist_size=200):
    masses = np.asarray([metadata[i][2] for i in allowed])
    estimates = np.asarray([b.neutral_mass(r['precursor_mz'], r['adduct']) for r in rows])
    mass = float(np.median(estimates))
    tol = max(b.CONFIG['mass_floor_da'], mass*b.CONFIG['mass_ppm']*1e-6)
    eligible = allowed[abs(masses-mass) <= tol]
    keys = [metadata[i][0] for i in eligible]
    smiles = {metadata[i][0]: metadata[i][1] for i in eligible}
    if not len(eligible):
        return [], [], {'mass':mass, 'candidate_structures':0, 'shortlist':0}
    q = b.sparse_vectors([b.peak_vector(r['ms2_mzs'], r['ms2_normalized_intensities'],
                                             r['precursor_mz']) for r in rows])
    scores = (q @ matrix[eligible].T).toarray()
    for j, row in enumerate(rows):
        mismatch = np.asarray([metadata[i][3] != row['ionization_mode'] for i in eligible])
        scores[j, mismatch] = 0.
    ranked_a = b.rank_candidates(scores, keys, set())
    shortlist = {key for key, _ in ranked_a[:shortlist_size]}
    b_evidence = {key: np.zeros(len(rows), dtype=float) for key in shortlist}
    qchannels = [exact_channels(row) for row in rows]
    for idx in eligible:
        key = metadata[idx][0]
        if key not in shortlist:
            continue
        ref_channels = exact_channels(raw[idx])
        for j, row in enumerate(rows):
            if row['ionization_mode'] == metadata[idx][3]:
                b_evidence[key][j] = max(b_evidence[key][j], exact_score(qchannels[j], ref_channels))
    ranked_b = sorted(((key, float(v.mean())) for key, v in b_evidence.items()),
                      key=lambda x: (-x[1], x[0]))
    return ([smiles[key] for key, _ in ranked_a[:25]],
            [smiles[key] for key, _ in ranked_b[:25]],
            {'mass':mass, 'candidate_structures':len(set(keys)),
             'shortlist':len(shortlist), 'shortlist_keys':list(shortlist),
             'mass_pool_keys':list(set(keys))})


def run(data, manifest_path, output, limit):
    started = time.monotonic()
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected = (manifest_path.parent/'split_manifest.sha256').read_text().split()[0]
    if manifest_hash != expected:
        raise ValueError('Frozen manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit 2026.03.3 required')
    keys = manifest['primary_fresh_dev_keys'][:limit]
    if set(keys) & (set(manifest['partitions']['legacy_dev']) | set(manifest['partitions']['sealed'])):
        raise AssertionError('Fresh evaluation overlaps exposed or sealed keys')
    output.mkdir(parents=True, exist_ok=False)
    record = {'status':'running', 'server':platform.node(), 'project':str(Path.cwd()),
              'entry':sys.argv, 'manifest_sha256':manifest_hash, 'evaluation_structures':len(keys),
              'cpu_threads':2, 'virtual_memory_limit_gib':12, 'timeout_seconds':900,
              'gpu':'none', 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_sha256':hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
              'greedy_helper_sha256':hashlib.sha256(Path(__file__).with_name('phase1.py').read_bytes()).hexdigest()}
    b.write_json(output/'run_record.json', record)
    try:
        groups, truth, query_spectra = load_fresh_queries(data, keys)
        qmass = [b.neutral_mass(r['precursor_mz'], r['adduct']) for rows in groups.values() for r in rows]
        train = b.read_metadata(data)
        metadata, matrix, raw, signatures, audit = build_reference(data, train, qmass)
        query_signatures = {spectrum_signature(r) for rows in groups.values() for r in rows}
        keep = np.asarray([i for i, sig in enumerate(signatures) if sig not in query_signatures], dtype=int)
        duplicate_refs = len(metadata)-len(keep)
        query_keys = set(keys)
        s2 = np.asarray([i for i in keep if metadata[i][0] not in query_keys], dtype=int)
        if any(metadata[i][0] in query_keys for i in s2):
            raise AssertionError('S2 reference contains a query structure')
        reference_keys = {metadata[i][0] for i in keep}
        predictions_a, predictions_b, diagnostics = {}, {}, {}
        for qid, rows in sorted(groups.items()):
            a, br, diagnostic = rank_one(rows, metadata, matrix, raw, keep)
            predictions_a[qid], predictions_b[qid] = a, br
            diagnostics[qid] = diagnostic
        supported = {qid: smiles for qid, smiles in truth.items() if b.canonical_key(smiles) in reference_keys}
        metrics = {
            'A_binned_all': b.evaluate(truth, predictions_a),
            'B_greedy_one_to_one_all': b.evaluate(truth, predictions_b),
            'A_binned_supported': b.evaluate(supported, predictions_a) if supported else None,
            'B_greedy_one_to_one_supported': b.evaluate(supported, predictions_b) if supported else None,
            'known_reference_coverage':len(supported)/len(truth),
            'mass_filtered_reference_coverage':sum(b.canonical_key(truth[qid]) in set(d['mass_pool_keys']) for qid,d in diagnostics.items())/len(truth),
            'shortlist_recall200':sum(b.canonical_key(truth[qid]) in set(d['shortlist_keys']) for qid,d in diagnostics.items())/len(truth),
            'S2_reference_audit_only':True,
            'S2_reason':'All query canonical keys are removed from every reference source. Pure spectral retrieval would have zero truth recall by construction; no S2 performance run was made.'}
        for d in diagnostics.values():
            d.pop('shortlist_keys')
            d.pop('mass_pool_keys')
        b.write_json(output/'metrics.json',metrics)
        b.write_json(output/'predictions.json',{'A_binned':predictions_a,'B_greedy_one_to_one':predictions_b})
        b.write_json(output/'diagnostics.json',diagnostics)
        b.write_json(output/'leakage_audit.json',{
            **audit, 'query_structures':len(keys), 'query_spectra':query_spectra,
            'exact_content_duplicate_reference_spectra_removed':duplicate_refs,
            'S1_reference_spectra':len(keep), 'S1_reference_structures':len(reference_keys),
            'S2_reference_spectra':len(s2),
            'S2_query_structure_reference_rows_removed':len(keep)-len(s2),
            'S2_query_key_intersection':len(query_keys & {metadata[i][0] for i in s2}),
            'measurement_id_available':False,
            'note':'Train has no spectrum_id or measurement_id; exact content duplicates are removed, but unseen altered copies cannot be disproved.'})
        record.update(status='complete', metrics=metrics, elapsed_seconds=time.monotonic()-started,
                      max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        b.write_json(output/'run_record.json',record)
    print(json.dumps({'status':record['status'],'metrics':metrics,'elapsed_seconds':record['elapsed_seconds'],
                      'max_rss_mib':record['max_rss_mib']}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=64)
    args=parser.parse_args()
    run(args.data,args.manifest,args.output,args.limit)



