"""Frozen hierarchical formula selection and candidate ranking on exposed development queries."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select(args):
    """Aggregate every predeclared formula/spectrum pair before selecting any formula."""
    protocol = read(args.protocol)
    metadata = read(args.metadata)
    frozen = read(args.candidates)
    inputs = read(args.inputs)
    cache = read(args.formula_cache)['candidate_formula_by_key']
    baseline = read(args.baseline)
    if protocol['status'] != 'frozen_before_subformula_explanation':
        raise ValueError('Unfrozen phase-four protocol')
    for field, path in [('formula_metadata_sha256', args.metadata),
                        ('phase3_baseline_predictions_sha256', args.baseline)]:
        if protocol[field] != digest(path):
            raise ValueError('Frozen input differs: ' + field)
    if protocol['source_candidate_sha256'] != digest(args.candidates):
        raise ValueError('Frozen candidate universe changed')
    qids = [x['qid'] for x in protocol['selected']]
    if len(qids) != 59 or qids != sorted(qids) or set(inputs['queries']) != set(baseline['hybrid']):
        raise ValueError('Query identity/order changed')
    batches = [read(p) for p in args.batches]
    if len(batches) != 3 or [b['qids'] for b in batches] != protocol['batches_qid_ascending']:
        raise ValueError('Incomplete fixed audit batches')
    results = {}
    for batch, batchpath in zip(batches, args.batches):
        if batch['protocol_sha256'] != digest(args.protocol) or batch['metadata_sha256'] != digest(args.metadata):
            raise ValueError('Batch parent identity changed')
        for item in batch['records']:
            label = item['qid'], item['formula_index'], item['spec']
            if label in results or item['qid'] not in batch['qids']:
                raise ValueError('Duplicate/unexpected formula-spectrum pair')
            base = args.pilot_results if item.get('reused_pilot') else args.new_results
            path = base / f"{item['qid']}_f{item['formula_index']:02}_{item['spec']}.json"
            if item['status'] == 'complete':
                if item['result_sha256'] != digest(path):
                    raise ValueError('Audit result SHA changed')
                obj = read(path)
                if (obj['status'] != 'complete' or (obj['qid'], obj['formula_index'], obj['spec']) != label or
                        obj['formula'] != metadata['queries'][item['qid']]['formulas'][item['formula_index']]['formula']):
                    raise ValueError('Audit result/formula identity changed')
                if obj['e_i'] != item['e_i'] or obj['e_n'] != item['e_n']:
                    raise ValueError('Recorded explanation changed')
                item = dict(item, processed_peaks_sha256=obj['processed_peaks_sha256'],
                            fragment_peak_count=obj['fragment_peak_count'],
                            matched_fragment_peak_count=obj['matched_fragment_peak_count'])
            results[label] = item
    all_candidates = {c['key']: c for c in frozen['candidates']}
    if len(all_candidates) != len(frozen['candidates']):
        raise ValueError('Candidate canonical key duplicated')
    selected = {}
    expected_labels = set()
    for row in protocol['selected']:
        qid = row['qid']
        qmeta = metadata['queries'][qid]
        qinput = inputs['queries'][qid]
        if qinput['candidate_count'] != qmeta['candidate_count'] or qinput['neutral_mass_da'] != qmeta['neutral_mass_da']:
            raise ValueError('Frozen mass pool changed')
        pool = qinput['candidate_keys']
        if len(pool) != len(set(pool)) or len(pool) != qmeta['candidate_count']:
            raise ValueError('Final pool has duplicate/missing keys')
        observed_counts = Counter(cache[k]['formula'] for k in pool)
        if len(qmeta['formulas']) < 2 or observed_counts != Counter({f['formula']: f['candidate_count'] for f in qmeta['formulas']}):
            raise ValueError('Complete formula candidate groups changed')
        for f in qmeta['formulas']:
            if not math.isclose(f['absolute_mass_error_da'],
                                abs(f['neutral_exact_mass_da'] - qinput['neutral_mass_da']),
                                rel_tol=0, abs_tol=1e-9):
                raise ValueError('Formula mass error differs from frozen all-adduct median query mass')
        hypotheses = []
        broken = []
        for fi, f in enumerate(qmeta['formulas']):
            if f['formula'] != row['formulas'][fi]:
                raise ValueError('Frozen formula order changed')
            readings = []
            for spec in row['hplus_positive_spec_ids']:
                label = qid, fi, spec
                expected_labels.add(label)
                item = results.get(label)
                if item is None or item['status'] != 'complete' or item['e_i'] is None or not math.isfinite(item['e_i']) or not -1e-12 <= item['e_i'] <= 1 + 1e-12:
                    broken.append({'formula': f['formula'], 'spec': spec,
                                   'status': 'missing' if item is None else item['status'] if item['status'] != 'complete' else 'invalid_or_empty_fragment_explanation'})
                    continue
                readings.append(item['e_i'])
            hypotheses.append({'formula': f['formula'], 'formula_index': fi,
                               'neutral_exact_mass_da': f['neutral_exact_mass_da'],
                               'absolute_mass_error_da': f['absolute_mass_error_da'],
                               'candidate_count': f['candidate_count'],
                               'hplus_e_i': readings,
                               'mean_e_i': None if len(readings) != len(row['hplus_positive_spec_ids']) else sum(readings)/len(readings)})
        if broken:
            status = 'audit_failure_or_na'
            order = []
        else:
            order = sorted(hypotheses, key=lambda f: (-f['mean_e_i'], f['absolute_mass_error_da'], f['formula']))
            if order[0]['mean_e_i'] < .50:
                status = 'low_explained_intensity'
            elif order[0]['mean_e_i'] - order[1]['mean_e_i'] < .05:
                status = 'formula_ambiguous_explanation'
            else:
                status = 'gate_pass'
        top = order[0] if order else None
        top_keys = [] if status != 'gate_pass' else [k for k in pool if cache[k]['formula'] == top['formula']]
        if status == 'gate_pass' and (len(top_keys) != len(set(top_keys)) or
                set(top_keys) != {k for k in pool if cache[k]['formula'] == top['formula']} or
                len(top_keys) != top['candidate_count']):
            raise ValueError('Top-formula candidate set changed')
        selected[qid] = {'status': status, 'broken_pairs': broken,
                         'ranked_formulas': order, 'all_formula_summaries': hypotheses,
                         'top_formula': None if status != 'gate_pass' else top['formula'],
                         'top_mean_e_i': None if top is None else top['mean_e_i'],
                         'top_second_gap_e_i': None if top is None else top['mean_e_i']-order[1]['mean_e_i'],
                         'selected_candidate_keys': top_keys,
                         'all_hplus_spec_ids': row['hplus_positive_spec_ids'],
                         'complete_candidate_count': len(pool)}
    if set(results) - expected_labels or len(expected_labels) != protocol['expected_formula_spectrum_pairs']:
        raise ValueError('Audit pair identity/denominator changed')
    attempted_pairs = len(results)
    for label in expected_labels - set(results):
        results[label] = {'qid': label[0], 'formula_index': label[1], 'spec': label[2],
                          'status': 'not_attempted_after_qid_failure'}
    if set(results) != expected_labels or len(results) != 759:
        raise ValueError('Complete frozen pair denominator changed')
    output = {'status': 'formula_gate_frozen_before_model_or_phase4_truth',
              'protocol_sha256': digest(args.protocol), 'metadata_sha256': digest(args.metadata),
              'candidate_sha256': digest(args.candidates), 'input_manifest_sha256': digest(args.inputs),
              'formula_cache_sha256': digest(args.formula_cache), 'phase3_baseline_sha256': digest(args.baseline),
              'batch_sha256': [digest(p) for p in args.batches], 'pilot_reuse_check_sha256': digest(args.reuse_check),
              'source_sha256': digest(__file__), 'queries': selected,
              'status_counts': dict(Counter(row['status'] for row in selected.values())),
              'truth_formula_used': False, 'mist_model_called': False, 'sealed_scored': False}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'selection.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    (args.output / 'selection.sha256').write_text(digest(path) + '  selection.json\n')
    print(json.dumps({'status': output['status'], 'selection_sha256': digest(path),
                      'status_counts': output['status_counts'], 'attempted_pairs': attempted_pairs, 'expected_pairs': len(expected_labels)}))



def prepare_model(args):
    """Freeze legal labels, original H+ spectra and representative-SMILES subset for gate-pass formulas."""
    import csv
    selection = read(args.selection)
    frozen = read(args.candidates)
    inputs = read(args.inputs)
    cache = read(args.formula_cache)['candidate_formula_by_key']
    map_rows = read(args.spectra_map)
    full_record = read(args.spectra_map.parent / 'run_record.json')
    if selection['status'] != 'formula_gate_frozen_before_model_or_phase4_truth' or (
            digest(args.selection) != (args.selection.parent / 'selection.sha256').read_text().split()[0]):
        raise ValueError('Formula selection is not frozen')
    for field,path in [('candidate_sha256',args.candidates),('input_manifest_sha256',args.inputs),
                       ('formula_cache_sha256',args.formula_cache)]:
        if selection[field] != digest(path):
            raise ValueError('Model input parent changed: ' + field)
    if full_record['mgf_sha256'] != digest(args.full_mgf) or full_record['protocol_sha256'] != selection['protocol_sha256']:
        raise ValueError('Original full H+ MGF changed')
    qids = sorted(q for q,row in selection['queries'].items() if row['status'] == 'gate_pass')
    if len(qids) != selection['status_counts'].get('gate_pass',0):
        raise ValueError('Gate-pass count changed')
    all_candidates = {c['key']:c for c in frozen['candidates']}
    if len(all_candidates) != len(frozen['candidates']):
        raise ValueError('Duplicate candidate key')
    subset = {}
    expected_specs = {}
    pool_sizes = {}
    for qid in qids:
        row = selection['queries'][qid]
        if row['all_hplus_spec_ids'] != inputs['queries'][qid]['hplus_positive_spec_ids']:
            raise ValueError('Complete H+ spectrum set changed')
        keys = row['selected_candidate_keys']
        if len(keys) != len(set(keys)) or not keys or set(keys) != {k for k in inputs['queries'][qid]['candidate_keys'] if cache[k]['formula'] == row['top_formula']}:
            raise ValueError('Selected formula does not include its complete candidate set')
        pool_sizes[qid] = len(keys)
        for key in keys:
            subset[key] = {'key':key,'smiles':all_candidates[key]['smiles']}
        for spec in row['all_hplus_spec_ids']:
            if spec in expected_specs:
                raise ValueError('Repeated H+ spectrum')
            expected_specs[spec] = (qid,row['top_formula'])
    by_spec = {x['spec']:x for x in map_rows}
    if len(by_spec) != len(map_rows) or set(expected_specs) - set(by_spec):
        raise ValueError('Frozen original spectrum map differs')
    blocks = {}
    for raw in args.full_mgf.read_text().split('BEGIN IONS\n')[1:]:
        block = 'BEGIN IONS\n' + raw.split('END IONS\n')[0] + 'END IONS\n\n'
        spec = next(x.split('=',1)[1] for x in block.splitlines() if x.startswith('FEATURE_ID='))
        if spec in blocks:
            raise ValueError('Duplicate original MGF spectrum')
        blocks[spec] = block
    if set(blocks) != set(by_spec) or set(expected_specs) - set(blocks):
        raise ValueError('Original MGF/map spectrum identity changed')
    chosen = [by_spec[s] for s in by_spec if s in expected_specs]
    if len(chosen) != len(expected_specs) or any(expected_specs[x['spec']][0] != x['qid'] for x in chosen):
        raise ValueError('Incomplete gate-pass H+ spectrum selection')
    args.output.mkdir(parents=True,exist_ok=False)
    mgf = args.output / 'model.mgf'
    mgf.write_text(''.join(blocks[x['spec']] for x in chosen))
    labels = args.output / 'model_labels.tsv'
    with labels.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['spec','formula','ionization','dataset','compound','parentmass','instrument'],delimiter='\t')
        writer.writeheader()
        for x in chosen:
            qid,formula=expected_specs[x['spec']]
            writer.writerow({'spec':x['spec'],'formula':formula,'ionization':'[M+H]+',
                             'dataset':'phase4_exposed_fresh_dev','compound':qid,
                             'parentmass':x['precursor_mz'],'instrument':x['instrument_type']})
    specmap = args.output / 'model_spectra_map.json'
    specmap.write_text(json.dumps([dict(x,neutral_formula=expected_specs[x['spec']][1]) for x in chosen],indent=2)+'\n')
    fp_subset = args.output / 'candidate_subset.json'
    fp_subset.write_text(json.dumps({'candidate_sha256':digest(args.candidates),
                                     'input_manifest_sha256':digest(args.selection),
                                     'selection_sha256':digest(args.selection),
                                     'pool_sizes':pool_sizes,
                                     'candidates':[subset[k] for k in sorted(subset)]},indent=2)+'\n')
    record = {'status':'model_inputs_frozen_before_inference','source_sha256':digest(__file__),
              'selection_sha256':digest(args.selection),'candidate_sha256':digest(args.candidates),
              'input_manifest_sha256':digest(args.inputs),'formula_cache_sha256':digest(args.formula_cache),
              'full_mgf_sha256':digest(args.full_mgf),'original_spectra_map_sha256':digest(args.spectra_map),
              'model_mgf_sha256':digest(mgf),'labels_sha256':digest(labels),'model_spectra_map_sha256':digest(specmap),
              'candidate_subset_sha256':digest(fp_subset),'model_qids':qids,
              'model_spectrum_count':len(chosen),'candidate_occurrences':sum(pool_sizes.values()),
              'unique_candidate_count':len(subset),'labels_have_smiles':False,
              'truth_formula_used':False,'sealed_scored':False}
    (args.output / 'run_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'qids':qids,'spectra':len(chosen),
                      'candidate_occurrences':record['candidate_occurrences'],
                      'unique_candidates':len(subset),
                      'candidate_subset_sha256':record['candidate_subset_sha256']}))



def rank(args):
    """Freeze all 76 F/FM lists while reusing unchanged phase-three hybrid fallbacks."""
    import numpy as np
    import pickle
    import baseline as b
    from mist_phase2 import rank_cosine
    selection=read(args.selection)
    frozen=read(args.candidates)
    inputs=read(args.inputs)
    cache=read(args.formula_cache)['candidate_formula_by_key']
    baseline=read(args.baseline)
    subset=read(args.fp_subset)
    fp=read(args.fp_model)
    comparison=read(args.fp_comparison)
    predcheck=read(args.prediction_check)
    subforms=read(args.subforms_audit)
    subidentity=read(args.subforms_identity)
    specmap=read(args.spectra_map)
    parent_members=read(args.phase3_members)['members']
    ambiguity_members=read(args.phase4_members)['members']
    qids=[m['qid'] for m in parent_members]
    ambiguous=[m['qid'] for m in ambiguity_members]
    if len(qids)!=76 or len(set(qids))!=76 or len(ambiguous)!=69 or set(selection['queries'])-set(ambiguous):
        raise ValueError('Frozen phase-three/four member denominator changed')
    if digest(args.selection)!=(args.selection.parent/'selection.sha256').read_text().split()[0]:
        raise ValueError('Gate selection is not frozen')
    for field,path in [('candidate_sha256',args.candidates),('input_manifest_sha256',args.inputs),
                       ('formula_cache_sha256',args.formula_cache),('phase3_baseline_sha256',args.baseline)]:
        if selection[field]!=digest(path):
            raise ValueError('Selection parent mismatch: '+field)
    if selection['status']!='formula_gate_frozen_before_model_or_phase4_truth':
        raise ValueError('Unfrozen gate')
    if set(baseline['hybrid'])!=set(qids) or set(baseline['mass_only'])!=set(qids):
        raise ValueError('Frozen phase-three baseline denominator changed')
    gate=[q for q in qids if q in selection['queries'] and selection['queries'][q]['status']=='gate_pass']
    if len(gate)!=selection['status_counts'].get('gate_pass',0):
        raise ValueError('Gate count changed')
    selected_union=set().union(*(set(selection['queries'][q]['selected_candidate_keys']) for q in gate))
    subset_keys={c['key'] for c in subset['candidates']}
    if subset['selection_sha256']!=digest(args.selection) or len(subset_keys)!=len(subset['candidates']) or subset_keys!=selected_union:
        raise ValueError('Selected fingerprint candidate union changed')
    if fp['candidate_subset_sha256']!=digest(args.fp_subset) or set(fp['vectors_hex_by_key'])!=selected_union:
        raise ValueError('Model fingerprint subset changed')
    if comparison['candidate_count']!=len(selected_union) or comparison['exact_equal_count']!=len(selected_union) or comparison['max_different_bits']!=0:
        raise ValueError('Cross-version selected candidate bitwise check failed')
    if predcheck['pred_sha256']!=digest(args.model_pred) or predcheck['shape']!=[len(predcheck['names']),4096] or predcheck['status']!='valid':
        raise ValueError('Model prediction identity/shape changed')
    if subidentity['selection_sha256']!=digest(args.selection) or subidentity['checked_spectra']!=len(specmap):
        raise ValueError('Official subformula identity check changed')
    if digest(args.checkpoint)!='a5a37e2b269f9a5ec483bd7fd2d3402257ce5f223270d534610fe41c7c4c5dea':
        raise ValueError('MIST checkpoint changed')
    if b.CONFIG['mass_ppm']!=15 or b.CONFIG['mass_floor_da']!=.003:
        raise ValueError('Frozen final mass tolerance changed')
    expected_spec={s:q for q in gate for s in selection['queries'][q]['all_hplus_spec_ids']}
    if len(expected_spec)!=sum(len(selection['queries'][q]['all_hplus_spec_ids']) for q in gate):
        raise ValueError('Duplicate selected H+ spectrum')
    mapped={x['spec']:x for x in specmap}
    if len(mapped)!=len(specmap) or set(mapped)!=set(expected_spec) or any(mapped[s]['qid']!=q for s,q in expected_spec.items()):
        raise ValueError('Model MGF spectrum mapping changed')
    audit={x['spec']:x for x in subforms}
    if len(audit)!=len(subforms) or set(audit)-set(expected_spec):
        raise ValueError('Unexpected or duplicate official subformula spectrum')
    with args.model_pred.open('rb') as f:
        model=pickle.load(f)
    names=list(model['names'])
    vectors=np.asarray(model['preds'])
    if len(names)!=len(set(names)) or set(names)-set(expected_spec) or vectors.shape!=(len(names),4096):
        raise ValueError('Unexpected or duplicate model prediction spectrum/shape')
    by_spec=dict(zip(names,vectors))
    by_key={c['key']:c for c in frozen['candidates']}
    if len(by_key)!=len(frozen['candidates']):
        raise ValueError('Duplicate frozen candidate key')
    F={q:list(baseline['hybrid'][q]) for q in qids}
    FM={q:list(baseline['hybrid'][q]) for q in qids}
    diagnostics={}
    failed={}
    for qid in gate:
        choice=selection['queries'][qid]
        mass=frozen['query_mass_da_by_id'][qid]
        poolkeys=inputs['queries'][qid]['candidate_keys']
        if mass!=inputs['queries'][qid]['neutral_mass_da'] or len(poolkeys)!=len(set(poolkeys)):
            raise ValueError('Frozen query mass/pool changed')
        tol=max(.003,mass*15e-6)
        pool=[by_key[k] for k in poolkeys]
        if len(pool)!=choice['complete_candidate_count'] or any(abs(c['exact_mass']-mass)>tol for c in pool):
            raise ValueError('Frozen final mass pool changed')
        mass_order=sorted(pool,key=lambda c:(abs(c['exact_mass']-mass),c['key']))
        if [c['smiles'] for c in mass_order[:25]]!=baseline['mass_only'][qid]:
            raise ValueError('Original phase-three mass tie/order changed')
        topkeys=set(choice['selected_candidate_keys'])
        if topkeys!={c['key'] for c in pool if cache[c['key']]['formula']==choice['top_formula']}:
            raise ValueError('Selected top formula is not the complete frozen formula group')
        top=[c for c in mass_order if c['key'] in topkeys]
        rest=[c for c in mass_order if c['key'] not in topkeys]
        if len(top)!=len(topkeys) or set(c['key'] for c in top)&set(c['key'] for c in rest):
            raise ValueError('Selected/rest candidate partition changed')
        reason=None
        ids=choice['all_hplus_spec_ids']
        if any(s not in audit or audit[s]['matched_subformulas']<=0 or audit[s]['root_formula_token']!=choice['top_formula'] or audit[s]['ion']!='[M+H]+' for s in ids):
            reason='invalid_or_empty_subformula_assignment'
        elif any(s not in by_spec or not np.isfinite(by_spec[s]).all() or np.linalg.norm(by_spec[s])<=0 for s in ids):
            reason='incomplete_model_spectrum_prediction'
        if reason is None:
            vectors=[]
            for c in top:
                packed=bytes.fromhex(fp['vectors_hex_by_key'][c['key']])
                if len(packed)!=512:
                    reason='invalid_candidate_fingerprint_bytes'
                    break
                bits=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='big')
                if bits.shape!=(4096,) or not np.isin(bits,[0,1]).all():
                    reason='invalid_candidate_fingerprint_bits'
                    break
                vectors.append(bits)
        if reason is not None:
            failed[qid]=reason
            diagnostics[qid]={'status':'whole_qid_phase3_hybrid_fallback','reason':reason,
                              'top_formula':choice['top_formula'],'complete_candidate_count':len(pool)}
            continue
        if len(top)==1:
            scored=[(top[0]['key'],None)]
        else:
            scored=rank_cosine([by_spec[s] for s in ids],vectors,[c['key'] for c in top])
        fkeys=[c['key'] for c in top+rest]
        fmkeys=[k for k,_ in scored]+[c['key'] for c in rest]
        if len(fkeys)!=len(pool) or len(fmkeys)!=len(pool) or set(fkeys)!=set(fmkeys) or set(fkeys)!=set(poolkeys):
            raise ValueError('F/FM complete candidate order changed')
        F[qid]=[by_key[k]['smiles'] for k in fkeys[:25]]
        FM[qid]=[by_key[k]['smiles'] for k in fmkeys[:25]]
        diagnostics[qid]={'status':'ranked','top_formula':choice['top_formula'],
                          'top_e_i':choice['top_mean_e_i'],'top_second_gap_e_i':choice['top_second_gap_e_i'],
                          'top_formula_candidate_count':len(top),'complete_candidate_count':len(pool),
                          'hplus_spec_ids':ids,'F_complete_order_keys':fkeys,
                          'FM_complete_order_keys':fmkeys,
                          'selected_formula_model_scores':{k:v for k,v in scored}}
    for qid in qids:
        if qid not in gate or qid in failed:
            if F[qid]!=baseline['hybrid'][qid] or FM[qid]!=baseline['hybrid'][qid]:
                raise ValueError('Old phase-three hybrid fallback changed')
    if set(F)!=set(qids) or set(FM)!=set(qids) or len(diagnostics)!=len(gate):
        raise ValueError('All76 predictions or gate diagnostics incomplete')
    args.output.mkdir(parents=True,exist_ok=False)
    predictions={'phase3_hybrid':baseline['hybrid'],'F':F,'FM':FM}
    path=args.output/'predictions.json'
    path.write_text(json.dumps(predictions,indent=2)+'\n')
    (args.output/'predictions.sha256').write_text(digest(path)+'  predictions.json\n')
    (args.output/'ranking_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    record={'status':'predictions_frozen_before_phase4_truth_scoring','source_sha256':digest(__file__),
            'selection_sha256':digest(args.selection),'candidate_sha256':digest(args.candidates),
            'input_manifest_sha256':digest(args.inputs),'formula_cache_sha256':digest(args.formula_cache),
            'phase3_baseline_sha256':digest(args.baseline),
            'phase3_members_sha256':digest(args.phase3_members),'phase4_members_sha256':digest(args.phase4_members),
            'fp_subset_sha256':digest(args.fp_subset),'fp_model_sha256':digest(args.fp_model),
            'fp_comparison_sha256':digest(args.fp_comparison),
            'model_prediction_sha256':digest(args.model_pred),'model_prediction_check_sha256':digest(args.prediction_check),
            'subforms_audit_sha256':digest(args.subforms_audit),'subforms_identity_sha256':digest(args.subforms_identity),
            'checkpoint_sha256':digest(args.checkpoint),'prediction_sha256':digest(path),
            'ranking_diagnostics_sha256':digest(args.output/'ranking_diagnostics.json'),
            'gate_pass_qids':gate,'ranked_qids':[q for q in gate if q not in failed],
            'model_or_input_failure_by_qid':failed,
            'phase3_hybrid_fallback_qids':[q for q in qids if q not in gate or q in failed],
            'truth_used_for_ranking':False,'sealed_scored':False}
    (args.output/'run_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'prediction_sha256':record['prediction_sha256'],
                      'ranked_qids':len(record['ranked_qids']),
                      'fallback_qids':len(record['phase3_hybrid_fallback_qids']),
                      'failures':failed}))



def evaluate(args):
    """Use already verified canonical target keys only after all76 rankings are frozen."""
    import numpy as np
    import baseline as b
    selection=read(args.selection)
    members=read(args.phase3_members)['members']
    ambiguity=read(args.phase4_members)['members']
    archived=read(args.phase3_metrics)
    archived_rows=read(args.phase3_rows)
    baseline=read(args.baseline)
    preds=read(args.predictions)
    run=read(args.predictions.parent/'run_record.json')
    if run['status']!='predictions_frozen_before_phase4_truth_scoring' or (
            digest(args.predictions)!=(args.predictions.parent/'predictions.sha256').read_text().split()[0]):
        raise ValueError('Phase-four predictions were not frozen')
    if run['prediction_sha256']!=digest(args.predictions) or run['source_sha256']!=digest(args.ranking_source_snapshot):
        raise ValueError('Exact executed ranking source snapshot changed')
    for field,path in [('selection_sha256',args.selection),('phase3_baseline_sha256',args.baseline),
                       ('phase3_members_sha256',args.phase3_members),('phase4_members_sha256',args.phase4_members)]:
        if run[field]!=digest(path):
            raise ValueError('Frozen prediction/evaluation identity changed: '+field)
    qids=[x['qid'] for x in members]
    truthkeys={x['qid']:x['canonical_target_key'] for x in members}
    if len(qids)!=len(set(qids)) or len(qids)!=len(set(truthkeys.values())) or len(qids)!=76 or set(qids)!=set(preds['F']) or set(qids)!=set(preds['FM']) or set(qids)!=set(preds['phase3_hybrid']):
        raise ValueError('Frozen 76 target/prediction key denominator changed')
    q59=[q for q in qids if q in selection['queries']]
    q69=[x['qid'] for x in ambiguity]
    gate=run['gate_pass_qids']
    if len(q59)!=59 or len(q69)!=69 or len(gate)!=selection['status_counts'].get('gate_pass',0) or set(gate)-set(q59):
        raise ValueError('Frozen subset denominator changed')
    if any(preds['phase3_hybrid'][q]!=baseline['hybrid'][q] for q in qids):
        raise ValueError('Old hybrid baseline changed')
    if any(preds['F'][q]!=baseline['hybrid'][q] or preds['FM'][q]!=baseline['hybrid'][q] for q in qids if q not in gate or q in run['model_or_input_failure_by_qid']):
        raise ValueError('Fallback top25 list changed')
    if set(run['phase3_hybrid_fallback_qids'])!={q for q in qids if q not in gate or q in run['model_or_input_failure_by_qid']}:
        raise ValueError('Fallback membership changed')
    key_cache={}
    def rank(target, guesses):
        if len(guesses)>25:
            raise ValueError('Prediction exceeds top25')
        keys=[]
        for smiles in guesses:
            if smiles not in key_cache:
                key_cache[smiles]=b.canonical_key(smiles)
            key=key_cache[smiles]
            if not key:
                raise ValueError('Invalid candidate prediction SMILES')
            keys.append(key)
        if len(keys)!=len(set(keys)):
            raise ValueError('Duplicate canonical candidate prediction')
        return next((i for i,k in enumerate(keys,1) if k==target),None)
    ranks={}
    for arm, predictions in [('mass_only',baseline['mass_only']),
                             ('phase3_hybrid',preds['phase3_hybrid']),('F',preds['F']),('FM',preds['FM'])]:
        if set(predictions)!=set(qids):
            raise ValueError('Prediction qid set changed: '+arm)
        ranks[arm]={q:rank(truthkeys[q],predictions[q]) for q in qids}
    def metrics(arm,selected):
        rr=np.asarray([0 if ranks[arm][q] is None else 1/ranks[arm][q] for q in selected],dtype=float)
        return {'molecules':len(selected),'mrr25':float(np.mean(rr)),
                'top1':float(np.mean(rr==1)),'top5':float(np.mean(rr>=.2)),
                'top25':float(np.mean(rr>0))}
    for arm,oldname in [('mass_only','mass_only_all76'),('phase3_hybrid','hybrid_all76')]:
        new=metrics(arm,qids)
        prior=archived[oldname]
        if any(not np.isclose(new[k],prior[k],rtol=0,atol=1e-12) for k in ['mrr25','top1','top5','top25']) or prior['molecules']!=76:
            raise ValueError('Frozen-target-key scoring differs from archived official phase-three score')
    oldrows={x['qid']:x for x in archived_rows}
    if set(oldrows)!=set(qids) or any(
            not np.isclose(oldrows[q]['hybrid_rr'],0 if ranks['phase3_hybrid'][q] is None else 1/ranks['phase3_hybrid'][q],rtol=0,atol=1e-12)
            for q in qids):
        raise ValueError('Old phase-three per-molecule truth scoring differs')
    def outcomes(a,z,selected):
        diffs=[(0 if ranks[z][q] is None else 1/ranks[z][q])-
               (0 if ranks[a][q] is None else 1/ranks[a][q]) for q in selected]
        return {'win':sum(x>0 for x in diffs),'tie':sum(x==0 for x in diffs),'loss':sum(x<0 for x in diffs)}
    def table(selected):
        return {'count':len(selected),'phase3_hybrid':metrics('phase3_hybrid',selected),
                'F':metrics('F',selected),'FM':metrics('FM',selected),
                'F_vs_phase3_hybrid':outcomes('phase3_hybrid','F',selected),
                'FM_vs_F':outcomes('F','FM',selected),
                'FM_vs_phase3_hybrid':outcomes('phase3_hybrid','FM',selected)}
    per=[]
    for q in qids:
        r={arm:ranks[arm][q] for arm in ['phase3_hybrid','F','FM']}
        rr={arm:0 if value is None else 1/value for arm,value in r.items()}
        gatepass=q in gate
        chosen=selection['queries'][q] if gatepass else None
        per.append({'qid':q,'fixed59':q in q59,'ambiguous69':q in q69,
                    'gate_pass':gatepass,'model_success':q in run['ranked_qids'],
                    'runtime_failure':run['model_or_input_failure_by_qid'].get(q),
                    'top_formula':None if chosen is None else chosen['top_formula'],
                    'top_formula_contains_truth_key':None if chosen is None else truthkeys[q] in chosen['selected_candidate_keys'],
                    'phase3_rank':r['phase3_hybrid'] if r['phase3_hybrid'] is not None else '>25',
                    'F_rank':r['F'] if r['F'] is not None else '>25',
                    'FM_rank':r['FM'] if r['FM'] is not None else '>25',
                    'phase3_rr':rr['phase3_hybrid'],'F_rr':rr['F'],'FM_rr':rr['FM'],
                    'formula_layer_rr_delta':rr['F']-rr['phase3_hybrid'],
                    'within_formula_mist_rr_delta':rr['FM']-rr['F'],
                    'total_rr_delta':rr['FM']-rr['phase3_hybrid']})
    results={'main_all76':table(qids),'secondary_fixed59':table(q59),
             'diagnostic_complete_ambiguous69':table(q69),
             'diagnostic_gate_pass':table(gate),
             'gate_status_counts':selection['status_counts'],
             'gate_truth_formula_contains_key_count':sum(x['top_formula_contains_truth_key'] for x in per if x['gate_pass']),
             'model_success_count':len(run['ranked_qids']),
             'phase3_mass_metric_identity_check':metrics('mass_only',qids),
             'phase3_hybrid_metric_identity_check':metrics('phase3_hybrid',qids)}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'per_molecule.json').write_text(json.dumps(per,indent=2)+'\n')
    (args.output/'run_record.json').write_text(json.dumps({
        'status':'complete','prediction_sha256':digest(args.predictions),
        'ranking_source_sha256':run['source_sha256'],'evaluation_source_sha256':digest(__file__),
        'phase3_members_sha256':digest(args.phase3_members),
        'phase4_members_sha256':digest(args.phase4_members),
        'phase3_archived_metrics_sha256':digest(args.phase3_metrics),
        'phase3_archived_per_molecule_sha256':digest(args.phase3_rows),
        'formula_selection_sha256':digest(args.selection),
        'truth_source':'frozen phase3 canonical_target_key verified against original truth SMILES in phase3',
        'truth_used_only_after_prediction_freeze':True,'sealed_scored':False},indent=2)+'\n')
    print(json.dumps({'status':'complete','main_all76':results['main_all76'],
                      'secondary_fixed59':results['secondary_fixed59'],
                      'gate_truth_formula_contains_key_count':results['gate_truth_formula_contains_key_count']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('select')
    for name in ['protocol','metadata','candidates','inputs','formula_cache','baseline','reuse_check']:
        a.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a.add_argument('--batches',type=Path,nargs=3,required=True)
    a.add_argument('--pilot-results',type=Path,required=True)
    a.add_argument('--new-results',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('prepare-model')
    for name in ['selection','candidates','inputs','formula_cache','spectra_map','full_mgf','output']:
        a.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=sub.add_parser('rank')
    for name in ['selection','candidates','inputs','formula_cache','baseline','fp_subset','fp_model',
                 'fp_comparison','model_pred','prediction_check','spectra_map','subforms_audit',
                 'subforms_identity','phase3_members','phase4_members','checkpoint','output']:
        a.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=sub.add_parser('evaluate')
    for name in ['selection','phase3_members','phase4_members','phase3_metrics','phase3_rows',
                 'baseline','predictions','ranking_source_snapshot','output']:
        a.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=p.parse_args()
    {'select':select,'prepare-model':prepare_model,'rank':rank,'evaluate':evaluate}[args.action](args)
