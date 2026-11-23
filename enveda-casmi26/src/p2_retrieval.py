"""P2 frozen-pool A retrieval and fixed RRF60 development comparison."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

import numpy as np

import baseline as b
from phase1_eval import build_reference, load_fresh_queries, rank_a_keys_for_pool, spectrum_signature


PARENT_FILES = {
    'members': 'outputs/phase5_20261023_01/members.json',
    'pools': 'outputs/phase5_20261023_02/pools.json',
    'queries': 'outputs/phase5_20261023_02/queries.json',
    'fm_predictions': 'outputs/phase5_20261023_12/predictions.json',
    'fm_diagnostics': 'outputs/phase5_20261023_12/diagnostics.json',
    'fm_metrics': 'outputs/phase5_20261023_13/metrics.json',
}
CONFIG = {
    'pilot_count': 32, 'evaluation_count': 140, 'rrf_k': 60,
    's1_reference': 'non-NP train; cap8 effective spectra per key/adduct; exact-content duplicate exclusion after cap',
    's2_reference': 'S1 minus every fresh_dev target key before ranking',
    'a_score': 'original binned dot; per query max reference per key then mean spectra; positive evidence only',
    'mass_rule': 'frozen COCONUT pool and 15ppm/floor0.003Da on both reference and pool exact mass',
    'cpu_threads': 2, 'virtual_memory_gib': 12, 'job_timeout_seconds': 900,
    'cumulative_cpu_wall_seconds': 3600, 'gpu': 'none', 'seed': 'frozen inputs; no random selection',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    b.write_json(path, obj)
    return sha(path)


def parents(root):
    return {name: {'path': rel, 'sha256': sha(root / rel)}
            for name, rel in PARENT_FILES.items()}


def read_inputs(root):
    members = json.loads((root / PARENT_FILES['members']).read_text())
    pools = json.loads((root / PARENT_FILES['pools']).read_text())
    queries = json.loads((root / PARENT_FILES['queries']).read_text())
    predictions = json.loads((root / PARENT_FILES['fm_predictions']).read_text())
    diagnostics = json.loads((root / PARENT_FILES['fm_diagnostics']).read_text())
    qids = [f'q{i:03}' for i in range(140)]
    assert [m['qid'] for m in members['members']] == qids
    assert all(set(x) == set(qids) for x in (pools, queries, diagnostics, predictions['fm']))
    keys = [m['canonical_target_key'] for m in members['members']]
    assert len(set(keys)) == 140
    for qid in qids:
        pool = pools[qid]
        by_key = {c['key']: c for c in pool}
        order = diagnostics[qid]['full_fm_key_order']
        assert len(by_key) == len(pool) == len(order) and set(order) == set(by_key)
        assert [by_key[k]['smiles'] for k in order[:25]] == predictions['fm'][qid]
    return qids, keys, pools, queries, diagnostics


def rrf60(fm_order, a_order):
    assert len(fm_order) == len(set(fm_order))
    assert len(a_order) == len(set(a_order)) and set(a_order) <= set(fm_order)
    if not a_order:
        return fm_order[:25]
    score = {key: 1.0 / (60 + rank) for rank, key in enumerate(fm_order, 1)}
    for rank, key in enumerate(a_order, 1):
        score[key] += 1.0 / (60 + rank)
    return sorted(fm_order, key=lambda key: (-score[key], key))[:25]


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=False)
    qids, keys, pools, queries, diagnostics = read_inputs(root)
    parent = parents(root)
    source = {'p2_retrieval': sha(Path(__file__)),
              'phase1_eval': sha(root / 'src/phase1_eval.py'),
              'baseline': sha(root / 'src/baseline.py')}
    protocol = {'status': 'frozen_before_P2_reference_or_effect_evaluation',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'parents': parent, 'sources': source,
                'config': CONFIG, 'pilot_qids': qids[:32], 'full_qids': qids,
                'split_manifest_sha256': json.loads((root / PARENT_FILES['members']).read_text())['parents']['split_manifest_sha256'],
                'reference_exclusion_keys_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
                'truth_usage': 'Target keys supplied only to prespecified S2 reference builder; ranking receives no target key or target SMILES. Evaluate only after pilot and full predictions freeze.'}
    write(out / 'protocol.json', protocol)
    write(out / 'run_record.json', {'status': 'protocol_frozen', 'protocol_sha256': sha(out / 'protocol.json'),
                                   'server': platform.node(), 'project': str(root), 'entry': sys.argv,
                                   'python': sys.version, 'rdkit': b.RDKIT_VERSION})


def check_protocol(root, out, evaluation=False):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['config'] == CONFIG
    assert parents(root) == protocol['parents']
    current = {
        'p2_retrieval': sha(Path(__file__)),
        'phase1_eval': sha(root / 'src/phase1_eval.py'),
        'baseline': sha(root / 'src/baseline.py')}
    if evaluation:
        assert sha(out / 'p2_retrieval.predict_source.py') == protocol['sources']['p2_retrieval']
        assert all(current[k] == protocol['sources'][k] for k in ('phase1_eval', 'baseline'))
    else:
        assert current == protocol['sources']
    assert b.RDKIT_VERSION == '2026.03.3'
    return protocol


def reference_hash(metadata, matrix, signatures, allowed):
    h = hashlib.sha256()
    for i in allowed:
        h.update(json.dumps(metadata[int(i)], ensure_ascii=False, separators=(',', ':')).encode())
        h.update(signatures[int(i)])
        start, end = matrix.indptr[i:i+2]
        h.update(matrix.indices[start:end].tobytes())
        h.update(matrix.data[start:end].tobytes())
    return h.hexdigest()


def predict(root, out):
    started = time.monotonic()
    protocol = check_protocol(root, out)
    qids, keys, pools, frozen_queries, diagnostics = read_inputs(root)
    groups, truth_unused, spectra = load_fresh_queries(root / 'data', keys, include_truth=False)
    assert not truth_unused and set(groups) == set(qids)
    for qid in qids:
        rows = groups[qid]
        expected = frozen_queries[qid]
        assert len(rows) == expected['source_spectra']
        masses = [b.neutral_mass(r['precursor_mz'], r['adduct']) for r in rows]
        assert abs(float(np.median(masses)) - expected['neutral_mass_da']) < 1e-9
    query_masses = [b.neutral_mass(r['precursor_mz'], r['adduct'])
                    for rows in groups.values() for r in rows]
    metadata, matrix, raw, signatures, build_audit = build_reference(
        root / 'data', b.read_metadata(root / 'data'), query_masses)
    # The old A cap is applied within build_reference. Exact stored-content duplicates are removed afterwards.
    query_sigs = {spectrum_signature(r) for rows in groups.values() for r in rows}
    s1 = np.asarray([i for i, sig in enumerate(signatures) if sig not in query_sigs], dtype=int)
    target_keys = set(keys)
    s2 = np.asarray([i for i in s1 if metadata[i][0] not in target_keys], dtype=int)
    assert not ({metadata[i][0] for i in s2} & target_keys)
    ref_audit = {'built': build_audit, 'query_spectra': spectra,
                 'query_signature_count': len(query_sigs),
                 'signature_exclusion_stage': 'after_reference_cap',
                 'same_stored_content_removed_after_cap': len(metadata)-len(s1),
                 's1': {'spectra': len(s1), 'structures': len({metadata[i][0] for i in s1}),
                        'sha256': reference_hash(metadata, matrix, signatures, s1)},
                 's2': {'spectra': len(s2), 'structures': len({metadata[i][0] for i in s2}),
                        'all_140_target_keys_removed_spectra': len(s1)-len(s2),
                        'target_key_intersection': 0,
                        'sha256': reference_hash(metadata, matrix, signatures, s2)}}
    write(out / 'reference_audit.json', ref_audit)

    def rank(qid_list):
        result = {'S1': {}, 'S2': {}}
        for pos, qid in enumerate(qid_list, 1):
            pool = pools[qid]
            by_key = {c['key']: c for c in pool}
            fm_order = diagnostics[qid]['full_fm_key_order']
            item = {}
            for scenario, allowed in (('S1', s1), ('S2', s2)):
                ranked_a, audit = rank_a_keys_for_pool(groups[qid], metadata, matrix, allowed, pool)
                a_order = [key for key, _ in ranked_a]
                assert len(a_order) == len(set(a_order)) and set(a_order) <= set(by_key)
                fused = rrf60(fm_order, a_order)
                assert len(fused) == len(set(fused)) and set(fused) <= set(by_key)
                if not a_order:
                    assert fused == fm_order[:25]
                item[scenario] = {'a_positive_key_order': a_order,
                                  'a_positive_scores': [score for _, score in ranked_a],
                                  'a_only_top25': [by_key[key]['smiles'] for key in a_order[:25]],
                                  'rrf_top25': [by_key[key]['smiles'] for key in fused],
                                  'fm_top25': [by_key[key]['smiles'] for key in fm_order[:25]],
                                  'diagnostic': audit}
                result[scenario][qid] = item[scenario]
            if pos % 8 == 0 or pos == len(qid_list):
                print(f'P2 rank complete {pos}/{len(qid_list)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        return result

    pilot = rank(qids[:32])
    write(out / 'pilot_predictions.json', pilot)
    print(f'P2 pilot frozen sha256={sha(out / "pilot_predictions.json")}', flush=True)
    full = rank(qids)
    for scenario in ('S1', 'S2'):
        assert {qid: full[scenario][qid] for qid in qids[:32]} == pilot[scenario]
    write(out / 'full_predictions.json', full)
    record = json.loads((out / 'run_record.json').read_text())
    record.update(status='predictions_frozen_before_effect_evaluation',
                  reference_audit_sha256=sha(out / 'reference_audit.json'),
                  pilot_predictions_sha256=sha(out / 'pilot_predictions.json'),
                  full_predictions_sha256=sha(out / 'full_predictions.json'),
                  elapsed_seconds=time.monotonic()-started,
                  maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out / 'run_record.json', record)


def metrics(rrs):
    n = len(rrs)
    return {'molecules': n, 'mrr25': sum(rrs)/n,
            'top1_count': sum(x == 1 for x in rrs),
            'top5_count': sum(x >= .2 for x in rrs),
            'top25_count': sum(x > 0 for x in rrs)}


def evaluate(root, out):
    check_protocol(root, out, evaluation=True)
    record = json.loads((out / 'run_record.json').read_text())
    assert record['status'] == 'predictions_frozen_before_effect_evaluation'
    assert sha(out / 'full_predictions.json') == record['full_predictions_sha256']
    assert sha(out / 'pilot_predictions.json') == record['pilot_predictions_sha256']
    assert sha(out / 'reference_audit.json') == record['reference_audit_sha256']
    predictions = json.loads((out / 'full_predictions.json').read_text())
    audit = json.loads((out / 'reference_audit.json').read_text())
    qids, keys, pools, frozen_queries, diagnostics = read_inputs(root)
    truth = dict(zip(qids, keys))
    rows = {}
    for scenario in ('S1', 'S2'):
        assert set(predictions[scenario]) == set(qids)
        rows[scenario] = {}
        for qid in qids:
            entry = predictions[scenario][qid]
            ranks = {}
            for arm, field in (('A', 'a_only_top25'), ('FM', 'fm_top25'), ('RRF60', 'rrf_top25')):
                smiles = entry[field]
                predkeys = [b.canonical_key(s) for s in smiles]
                assert all(predkeys) and len(predkeys) == len(set(predkeys)) and len(smiles) <= 25
                ranks[arm] = next((i for i, key in enumerate(predkeys, 1) if key == truth[qid]), None)
            supported = truth[qid] in entry['diagnostic']['mode_qualified_reference_keys']
            rows[scenario][qid] = {
                'ranks': ranks, 'reciprocal_ranks': {k: (1/v if v else 0.) for k, v in ranks.items()},
                'truth_key_in_eligible_reference': supported,
                'any_positive_A_evidence': bool(entry['a_positive_key_order']),
                'truth_key_positive_A_evidence': truth[qid] in entry['a_positive_key_order'],
                'pool_size': len(pools[qid])}
    assert audit['s2']['target_key_intersection'] == 0
    assert all(rows['S2'][q]['ranks']['A'] is None for q in qids)
    assert all(rows['S2'][q]['reciprocal_ranks']['RRF60'] <= rows['S2'][q]['reciprocal_ranks']['FM']+1e-12 for q in qids)
    assert all(rows['S1'][q]['ranks']['FM'] == rows['S2'][q]['ranks']['FM'] for q in qids)
    old_fm = json.loads((root / PARENT_FILES['fm_metrics']).read_text())['all140']['fm']
    summary = {}
    for scenario in ('S1', 'S2'):
        subset = [q for q in qids if rows[scenario][q]['truth_key_in_eligible_reference']]
        summary[scenario] = {'all140': {}, 'supported_reference': {'n': len(subset)},
                             'positive_A_any_qids': sum(rows[scenario][q]['any_positive_A_evidence'] for q in qids),
                             'truth_positive_A_qids': sum(rows[scenario][q]['truth_key_positive_A_evidence'] for q in qids)}
        for arm in ('A', 'FM', 'RRF60'):
            summary[scenario]['all140'][arm] = metrics([rows[scenario][q]['reciprocal_ranks'][arm] for q in qids])
            summary[scenario]['supported_reference'][arm] = metrics([rows[scenario][q]['reciprocal_ranks'][arm] for q in subset]) if subset else None
        assert abs(summary[scenario]['all140']['FM']['mrr25'] - old_fm['mrr25']) < 1e-12
        assert all(summary[scenario]['all140']['FM'][k] == old_fm[k] for k in ('top1_count', 'top5_count', 'top25_count'))
        for arm in ('A', 'RRF60'):
            diffs = [rows[scenario][q]['reciprocal_ranks'][arm]-rows[scenario][q]['reciprocal_ranks']['FM'] for q in qids]
            summary[scenario][f'{arm}_versus_FM'] = dict(win=sum(d > 1e-12 for d in diffs),
                                                        tie=sum(abs(d) <= 1e-12 for d in diffs),
                                                        loss=sum(d < -1e-12 for d in diffs),
                                                        delta_mrr25=sum(diffs)/140)
    pilot_summary = {scenario: {arm: metrics([rows[scenario][q]['reciprocal_ranks'][arm] for q in qids[:32]])
                                for arm in ('A', 'FM', 'RRF60')} for scenario in ('S1', 'S2')}
    write(out / 'evaluation.json', {'prediction_sha256': record['full_predictions_sha256'],
                                    'reference_audit_sha256': record['reference_audit_sha256'],
                                    'summary': summary, 'pilot32': pilot_summary, 'per_qid': rows})
    record['status'] = 'complete'
    record['evaluation_sha256'] = sha(out / 'evaluation.json')
    record['evaluation_source_sha256'] = sha(Path(__file__))
    record['evaluation_diff_sha256'] = sha(out / 'evaluation_source.diff')
    write(out / 'run_record.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'predict', 'evaluate'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': prepare, 'predict': predict, 'evaluate': evaluate}[args.action](args.root.resolve(), args.output.resolve())




