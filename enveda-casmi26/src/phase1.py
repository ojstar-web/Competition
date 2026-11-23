"""Frozen phase-one split and leakage checks for the Enveda CASMI baseline."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import pyarrow.parquet as pq
import baseline as b

SALT = 'enveda-phase1-20261023-v1'
LEGACY_SPLIT = Path('outputs/baseline_20261023_03/split.json')


def split_keys(keys, legacy):
    keys = set(keys)
    legacy = set(legacy)
    if not legacy <= keys:
        raise ValueError(f'{len(legacy-keys)} legacy structures absent from NP source')
    ordered = sorted(keys-legacy, key=lambda key: (hashlib.sha256(f'{SALT}|{key}'.encode()).hexdigest(), key))
    sealed_count = len(ordered)//3
    return {'legacy_dev': sorted(legacy), 'sealed': ordered[:sealed_count],
            'fresh_dev': ordered[sealed_count:]}


def select_reference_rows(rows, excluded_keys):
    return [row for row in rows if row[0] not in excluded_keys]


def greedy_similarity(query, reference, floor_da=.005, ppm=10.):
    import math
    from bisect import bisect_left, bisect_right
    qnorm = math.sqrt(sum(w*w for _, w in query))
    rnorm = math.sqrt(sum(w*w for _, w in reference))
    if not qnorm or not rnorm:
        return 0.
    ordered_ref = sorted((float(mz), float(w), i) for i, (mz, w) in enumerate(reference))
    ref_mzs = [x[0] for x in ordered_ref]
    pairs = []
    for qi, (qmz, qw) in enumerate(query):
        relative = ppm*1e-6
        lower_tol = max(floor_da, relative*qmz)
        upper_tol = max(floor_da, relative*qmz/(1-relative))
        for j in range(bisect_left(ref_mzs, qmz-lower_tol), bisect_right(ref_mzs, qmz+upper_tol)):
            rmz, rw, ri = ordered_ref[j]
            if abs(qmz-rmz) <= max(floor_da, ppm*max(qmz, rmz)*1e-6):
                pairs.append((-qw*rw, abs(qmz-rmz), qi, ri))
    used_q, used_r = set(), set()
    dot = 0.
    for neg_score, _, qi, ri in sorted(pairs):
        if qi not in used_q and ri not in used_r:
            dot -= neg_score
            used_q.add(qi)
            used_r.add(ri)
    return dot/(qnorm*rnorm)


def prepare(data, output):
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError(f'Official RDKit version required: {b.RDKIT_VERSION}')
    train = b.read_metadata(data)
    np_smiles = sorted({s for s, src in zip(train['normalized_smiles'], train['ingest_lib'])
                        if src == b.CONFIG['validation_source'] and isinstance(s, str)})
    by_key = {}
    invalid = 0
    for smiles in np_smiles:
        info = b.structure_info(smiles)
        if info:
            by_key.setdefault(info[0], smiles)
        else:
            invalid += 1
    old = json.loads(LEGACY_SPLIT.read_text(encoding='utf-8'))
    legacy = set(old['canonical_structure_keys'])
    if legacy != {b.canonical_key(s) for s in old['validation'].values()}:
        raise ValueError('Legacy split keys and truth disagree')
    parts = split_keys(by_key, legacy)
    if len(parts['legacy_dev']) != 40:
        raise ValueError('Expected exactly 40 previously exposed legacy structures')
    manifest = {'protocol': 'phase1-20261023-v1', 'salt': SALT,
                'canonicalizer': 'RDKit 2026.03.3 remove stereo, canonical tautomer, InChIKey14',
                'source': b.CONFIG['validation_source'],
                'train_file': {'bytes': (data/'train.parquet').stat().st_size,
                               'mtime_ns': (data/'train.parquet').stat().st_mtime_ns},
                'np_unique_smiles': len(np_smiles), 'np_invalid_smiles': invalid,
                'np_unique_valid_keys': len(by_key), 'counts': {k: len(v) for k, v in parts.items()},
                'primary_fresh_dev_keys': parts['fresh_dev'][:64],
                'partitions': parts,
                'sealed_policy': 'No model inference, prediction reading, coverage scoring or parameter selection on sealed in phase one.'}
    output.mkdir(parents=True, exist_ok=False)
    path = output/'split_manifest.json'
    b.write_json(path, manifest)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'split_manifest.sha256').write_text(sha+'  split_manifest.json\n', encoding='utf-8')
    b.write_json(output/'run_record.json', {
        'status': 'manifest_frozen', 'server': platform.node(),
        'project': str(Path.cwd()), 'entry': sys.argv,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_source_sha256': hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
        'rdkit': b.RDKIT_VERSION, 'pyarrow': importlib.metadata.version('pyarrow'),
        'manifest_sha256': sha, 'counts': manifest['counts'],
        'gpu': 'none', 'cpu_threads': 2})
    print(json.dumps({'status':'manifest_frozen', 'counts':manifest['counts'], 'manifest_sha256':sha}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare'])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.data, args.output)
