"""Frozen P3 reliability router on P2 scores; no new spectrum or model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import baseline as b


FILES = {
    'members': 'outputs/phase5_20261023_01/members.json',
    'pools': 'outputs/phase5_20261023_02/pools.json',
    'fm_diagnostics': 'outputs/phase5_20261023_12/diagnostics.json',
    'fm_predictions': 'outputs/phase5_20261023_12/predictions.json',
    'p2_protocol': 'outputs/p2_20261025_02/protocol.json',
    'p2_reference': 'outputs/p2_20261025_02/reference_audit.json',
    'p2_predictions': 'outputs/p2_20261025_02/full_predictions.json',
    'p2_record': 'outputs/p2_20261025_02/run_record.json',
}
EXPECTED = {
    'members': '577fb395cdb8122b783b3f143f7af640d0871d89a01103fb865580fa909de943',
    'pools': 'ecbcfd92102c36c45618f48852eee6f0e24e7a8d3f7a4ecbf52c0b68941b17d3',
    'fm_diagnostics': '1999427927d7620a25cadde32f50fa84a73e8f04fc9915d01bf545ecb98eab88',
    'fm_predictions': '5ccbb209ee95ee964aadd652e0ff7208297865e661fa82223965e4851b1c4ce4',
    'p2_protocol': '535e24880359e09a44f8b1aa801de82613e3568fd380c2e2e2843dca87e77b45',
    'p2_reference': '752cce54854ea919cbecee63958dbf6da243168f72109504e43ea8c93165ce47',
    'p2_predictions': 'fab2d4af181317cf69dc4b53b37d34c32172a70300045a60cb031a62aa93690a',
}
TAUS = (.70, .80, .90, .95)
DELTAS = (.05, .10, .20)
GRID = [(tau, delta) for tau in TAUS for delta in DELTAS]
SENTINEL = 'FM'
CONFIG = {
    'fold_salt': 'enveda-p3-router-v1',
    'folds': 5,
    'train_size': 112,
    'holdout_size': 28,
    'taus': list(TAUS),
    'deltas': list(DELTAS),
    'train_s2_mrr_floor': 'same 112 FM MRR, tolerance 1e-12',
    'train_objective': 'maximize S1 MRR; tie fewer S2 routes then larger tau then larger delta; no gain => FM',
    'oof_candidate_gate': {'s1_delta_mrr_min': .01, 's2_delta_mrr_min': 0., 's1_routes_min': 20},
    'cpu_threads': 2, 'virtual_memory_gib': 12, 'job_timeout_seconds': 900,
    'cumulative_cpu_wall_seconds': 600, 'gpu': 'none',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    b.write_json(path, obj)
    return sha(path)


def parent_hashes(root):
    return {name: {'path': rel, 'sha256': sha(root / rel)} for name, rel in FILES.items()}


def threshold_name(tau, delta):
    return f'tau={tau:.2f},delta={delta:.2f}'


def route(scores, tau, delta):
    if not scores:
        return False
    second = scores[1] if len(scores) > 1 else 0.
    return scores[0] >= tau and scores[0]-second >= delta


def priority_order(fm, a):
    assert len(fm) == len(set(fm)) and len(a) == len(set(a)) and set(a) <= set(fm)
    aset = set(a)
    result = a + [key for key in fm if key not in aset]
    assert len(result) == len(fm) and set(result) == set(fm)
    return result


def folds_from_members(members):
    ordered = sorted(members, key=lambda m: (
        hashlib.sha256(f"enveda-p3-router-v1|{m['canonical_target_key']}".encode()).hexdigest(), m['canonical_target_key']))
    folds = {f'fold{i}': [m['qid'] for j, m in enumerate(ordered) if j % 5 == i] for i in range(5)}
    assert len(ordered) == 140 and all(len(qids) == 28 for qids in folds.values())
    assert len(set(q for qids in folds.values() for q in qids)) == 140
    return folds


def load_inputs(root):
    assert b.RDKIT_VERSION == '2026.03.3'
    actual = parent_hashes(root)
    assert all(actual[k]['sha256'] == expected for k, expected in EXPECTED.items())
    record = json.loads((root / FILES['p2_record']).read_text())
    assert record['status'] == 'complete'
    assert record['full_predictions_sha256'] == actual['p2_predictions']['sha256']
    assert record['reference_audit_sha256'] == actual['p2_reference']['sha256']
    members = json.loads((root / FILES['members']).read_text())['members']
    pools = json.loads((root / FILES['pools']).read_text())
    fm_diagnostics = json.loads((root / FILES['fm_diagnostics']).read_text())
    fm_predictions = json.loads((root / FILES['fm_predictions']).read_text())['fm']
    p2 = json.loads((root / FILES['p2_predictions']).read_text())
    ref = json.loads((root / FILES['p2_reference']).read_text())
    assert ref['s2']['target_key_intersection'] == 0
    qids = [f'q{i:03}' for i in range(140)]
    assert [m['qid'] for m in members] == qids
    assert all(set(x) == set(qids) for x in (pools, fm_diagnostics, fm_predictions, p2['S1'], p2['S2']))
    target_keys = {m['canonical_target_key'] for m in members}
    assert len(target_keys) == 140
    for qid in qids:
        pool = pools[qid]
        by_key = {c['key']: c['smiles'] for c in pool}
        fm = fm_diagnostics[qid]['full_fm_key_order']
        assert len(by_key) == len(pool) == len(fm) and set(fm) == set(by_key)
        assert [by_key[k] for k in fm[:25]] == fm_predictions[qid]
        for scenario in ('S1', 'S2'):
            item = p2[scenario][qid]
            a = item['a_positive_key_order']
            scores = item['a_positive_scores']
            assert len(a) == len(scores) == len(set(a)) and set(a) <= set(fm)
            assert all(math.isfinite(x) and x > 0 for x in scores)
            assert list(zip(a, scores)) == sorted(zip(a, scores), key=lambda x: (-x[1], x[0]))
            assert [by_key[k] for k in a[:25]] == item['a_only_top25']
            assert item['fm_top25'] == fm_predictions[qid]
            if scenario == 'S2':
                assert not (set(a) & target_keys)
    return members, pools, fm_diagnostics, p2


def freeze(root, out):
    members, pools, fm_diagnostics, p2 = load_inputs(root)
    out.mkdir(parents=True, exist_ok=False)
    folds = folds_from_members(members)
    protocol = {
        'status': 'frozen_before_P3_calibration_or_OOF_evaluation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parents': parent_hashes(root),
        'source_sha256': sha(Path(__file__)),
        'baseline_sha256': sha(root / 'src/baseline.py'),
        'config': CONFIG,
        'folds': folds,
        'truth_boundary': 'canonical keys assign folds and audit prior S2 exclusion; only 112 training keys enter score-based selection; holdout correctness is evaluated only after OOF predictions freeze; s1/gap are sole route features',
    }
    write(out / 'protocol.json', protocol)
    write(out / 'run_record.json', {'status': 'protocol_frozen',
        'protocol_sha256': sha(out / 'protocol.json'), 'server': platform.node(),
        'project': str(root), 'entry': sys.argv, 'python': sys.version,
        'rdkit': b.RDKIT_VERSION})


def check_protocol(root, out):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['config'] == CONFIG
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['baseline_sha256'] == sha(root / 'src/baseline.py')
    assert protocol['parents'] == parent_hashes(root)
    members, pools, fm_diagnostics, p2 = load_inputs(root)
    assert protocol['folds'] == folds_from_members(members)
    return protocol, members, pools, fm_diagnostics, p2


def rr(key_order, target):
    try:
        pos = key_order[:25].index(target) + 1
        return 1. / pos
    except ValueError:
        return 0.


def build_orders(pools, fm_diagnostics, p2):
    """Truth-free orders for the 12 frozen options and all comparison arms."""
    orders = {'S1': {}, 'S2': {}}
    for scenario in orders:
        for qid, item in p2[scenario].items():
            fm = fm_diagnostics[qid]['full_fm_key_order']
            a = item['a_positive_key_order']
            ap = priority_order(fm, a)
            choices = {SENTINEL: fm, 'A_priority': ap, 'A_only': a}
            for tau, delta in GRID:
                choices[threshold_name(tau, delta)] = ap if route(item['a_positive_scores'], tau, delta) else fm
            orders[scenario][qid] = choices
    return orders


def choose_on_train(train_qids, train_truth, orders, p2):
    assert len(train_qids) == len(train_truth) and set(train_qids) == set(train_truth)
    n = len(train_qids)
    base = {scenario: sum(rr(orders[scenario][q][SENTINEL], train_truth[q]) for q in train_qids)/n
            for scenario in ('S1', 'S2')}
    eligible = []
    for tau, delta in GRID:
        name = threshold_name(tau, delta)
        s1 = sum(rr(orders['S1'][q][name], train_truth[q]) for q in train_qids)/n
        s2 = sum(rr(orders['S2'][q][name], train_truth[q]) for q in train_qids)/n
        if s2 < base['S2']-1e-12 or s1 <= base['S1']+1e-12:
            continue
        s2_routes = sum(route(p2['S2'][q]['a_positive_scores'], tau, delta) for q in train_qids)
        eligible.append((s1, -s2_routes, tau, delta, name, s2))
    if not eligible:
        return {'name': SENTINEL, 'tau': None, 'delta': None, 'train_s1_mrr': base['S1'],
                'train_s2_mrr': base['S2'], 'train_s2_routes': 0,
                'train_fm_s1_mrr': base['S1'], 'train_fm_s2_mrr': base['S2']}
    best_s1 = max(item[0] for item in eligible)
    best = max((item for item in eligible if item[0] >= best_s1-1e-12),
               key=lambda item: (item[1], item[2], item[3]))
    return {'name': best[4], 'tau': best[2], 'delta': best[3],
            'train_s1_mrr': best[0], 'train_s2_mrr': best[5],
            'train_s2_routes': -best[1],
            'train_fm_s1_mrr': base['S1'], 'train_fm_s2_mrr': base['S2']}


def predict(root, out):
    started = time.monotonic()
    protocol, members, pools, fm_diagnostics, p2 = check_protocol(root, out)
    orders = build_orders(pools, fm_diagnostics, p2)
    by_qid = {m['qid']: m for m in members}
    qids = [m['qid'] for m in members]
    selections = {}
    oof = {scenario: {} for scenario in ('S1', 'S2')}
    for fold, holdout in protocol['folds'].items():
        train_qids = [q for q in qids if q not in set(holdout)]
        train_truth = {q: by_qid[q]['canonical_target_key'] for q in train_qids}
        selected = choose_on_train(train_qids, train_truth, orders, p2)
        selected['train_qids_sha256'] = hashlib.sha256('\n'.join(train_qids).encode()).hexdigest()
        selected['holdout_qids'] = holdout
        selections[fold] = selected
        for scenario in ('S1', 'S2'):
            for qid in holdout:
                pool = {c['key']: c['smiles'] for c in pools[qid]}
                choice = selected['name']
                keys = orders[scenario][qid][choice][:25]
                assert len(keys) == len(set(keys)) and set(keys) <= set(pool)
                routed = bool(choice != SENTINEL and
                    route(p2[scenario][qid]['a_positive_scores'], selected['tau'], selected['delta']))
                if scenario == 'S2':
                    assert not (set(p2[scenario][qid]['a_positive_key_order']) & {by_qid[q]['canonical_target_key'] for q in qids})
                oof[scenario][qid] = {
                    'fold': fold, 'selected': choice, 'routed_to_A': routed,
                    'router_top25': [pool[k] for k in keys],
                    'fm_top25': [pool[k] for k in orders[scenario][qid][SENTINEL][:25]],
                    'a_only_top25': [pool[k] for k in orders[scenario][qid]['A_only'][:25]],
                    'a_priority_top25': [pool[k] for k in orders[scenario][qid]['A_priority'][:25]],
                }
        print(f'P3 {fold} selected={selected["name"]} holdout={len(holdout)}', flush=True)
    assert all(set(oof[s]) == set(qids) for s in ('S1', 'S2'))
    assert all(oof['S1'][q]['fold'] == oof['S2'][q]['fold'] for q in qids)
    write(out / 'fold_selections.json', selections)
    write(out / 'oof_predictions.json', oof)
    record = json.loads((out / 'run_record.json').read_text())
    record.update(status='OOF_predictions_frozen_before_holdout_evaluation',
        fold_selections_sha256=sha(out / 'fold_selections.json'),
        oof_predictions_sha256=sha(out / 'oof_predictions.json'),
        elapsed_seconds=time.monotonic()-started,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out / 'run_record.json', record)


def summarize(rrs):
    n = len(rrs)
    return {'n': n, 'mrr25': sum(rrs)/n,
            'top1_count': sum(x == 1 for x in rrs),
            'top5_count': sum(x >= .2 for x in rrs),
            'top25_count': sum(x > 0 for x in rrs)}


def evaluate(root, out):
    started = time.monotonic()
    protocol, members, pools, fm_diagnostics, p2 = check_protocol(root, out)
    record = json.loads((out / 'run_record.json').read_text())
    assert record['status'] == 'OOF_predictions_frozen_before_holdout_evaluation'
    assert sha(out / 'fold_selections.json') == record['fold_selections_sha256']
    assert sha(out / 'oof_predictions.json') == record['oof_predictions_sha256']
    selections = json.loads((out / 'fold_selections.json').read_text())
    oof = json.loads((out / 'oof_predictions.json').read_text())
    qids = [m['qid'] for m in members]
    truth = {m['qid']: m['canonical_target_key'] for m in members}
    assert set(selections) == set(protocol['folds'])
    all_rows, summary = {}, {}
    for scenario in ('S1', 'S2'):
        assert set(oof[scenario]) == set(qids)
        all_rows[scenario] = {}
        summary[scenario] = {}
        for qid in qids:
            item = oof[scenario][qid]
            assert qid in protocol['folds'][item['fold']]
            assert item['selected'] == selections[item['fold']]['name']
            ranks = {}
            for arm, field in (('router', 'router_top25'), ('FM', 'fm_top25'),
                               ('A_only', 'a_only_top25'), ('A_priority', 'a_priority_top25')):
                predkeys = [b.canonical_key(s) for s in item[field]]
                assert all(predkeys) and len(predkeys) == len(set(predkeys)) and len(predkeys) <= 25
                ranks[arm] = next((i for i, k in enumerate(predkeys, 1) if k == truth[qid]), None)
            all_rows[scenario][qid] = {
                'ranks': ranks, 'reciprocal_ranks': {k: 1/v if v else 0. for k, v in ranks.items()},
                'routed_to_A': item['routed_to_A'], 'fold': item['fold']}
        for arm in ('router', 'FM', 'A_only', 'A_priority'):
            summary[scenario][arm] = summarize([all_rows[scenario][q]['reciprocal_ranks'][arm] for q in qids])
        diffs = [all_rows[scenario][q]['reciprocal_ranks']['router']-
                 all_rows[scenario][q]['reciprocal_ranks']['FM'] for q in qids]
        summary[scenario]['router_versus_FM'] = {
            'win': sum(d > 1e-12 for d in diffs), 'tie': sum(abs(d) <= 1e-12 for d in diffs),
            'loss': sum(d < -1e-12 for d in diffs), 'delta_mrr25': sum(diffs)/140,
            'route_count': sum(all_rows[scenario][q]['routed_to_A'] for q in qids)}
    assert all(all_rows['S2'][q]['reciprocal_ranks']['router'] <=
               all_rows['S2'][q]['reciprocal_ranks']['FM']+1e-12 for q in qids)
    old_fm = json.loads((root / 'outputs/phase5_20261023_13/metrics.json').read_text())['all140']['fm']
    for scenario in ('S1', 'S2'):
        assert abs(summary[scenario]['FM']['mrr25']-old_fm['mrr25']) < 1e-12
        assert all(summary[scenario]['FM'][k] == old_fm[k] for k in ('top1_count','top5_count','top25_count'))
    fold_results = {}
    for fold, holdout in protocol['folds'].items():
        fold_results[fold] = {}
        for scenario in ('S1', 'S2'):
            fold_results[fold][scenario] = {
                'FM': summarize([all_rows[scenario][q]['reciprocal_ranks']['FM'] for q in holdout]),
                'router': summarize([all_rows[scenario][q]['reciprocal_ranks']['router'] for q in holdout]),
                'route_count': sum(all_rows[scenario][q]['routed_to_A'] for q in holdout)}
            fold_results[fold][scenario]['delta_mrr25'] = (
                fold_results[fold][scenario]['router']['mrr25']-
                fold_results[fold][scenario]['FM']['mrr25'])
    gate = {'s1_delta_mrr_min': summary['S1']['router_versus_FM']['delta_mrr25'] >= .01-1e-12,
            's2_delta_mrr_min': summary['S2']['router_versus_FM']['delta_mrr25'] >= -1e-12,
            's1_routes_min20': summary['S1']['router_versus_FM']['route_count'] >= 20}
    result = {'protocol_sha256': sha(out / 'protocol.json'),
              'oof_predictions_sha256': record['oof_predictions_sha256'],
              'fold_selections_sha256': record['fold_selections_sha256'],
              'summary': summary, 'fold_results': fold_results,
              'gate': {'checks': gate, 'passed': all(gate.values())},
              'per_qid': all_rows}
    write(out / 'oof_evaluation.json', result)
    if all(gate.values()):
        orders = build_orders(pools, fm_diagnostics, p2)
        selected = choose_on_train(qids, truth, orders, p2)
        write(out / 'final_full140_threshold.json', {
            'development_fit_only': True, 'oof_evaluation_sha256': sha(out / 'oof_evaluation.json'),
            'selection': selected})
    record['status'] = 'complete'
    record['oof_evaluation_sha256'] = sha(out / 'oof_evaluation.json')
    record['gate_passed'] = all(gate.values())
    record['evaluation_elapsed_seconds'] = time.monotonic()-started
    write(out / 'run_record.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','predict','evaluate'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'predict': predict, 'evaluate': evaluate}[args.action](args.root.resolve(), args.output.resolve())




