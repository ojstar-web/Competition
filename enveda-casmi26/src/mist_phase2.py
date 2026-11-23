"""Frozen, truth-free MIST CPU inputs and prediction wrapper for phase-two pilot."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    import pyarrow.parquet as pq
    import baseline as b
    from coconut_candidates import frozen_manifest, frozen_remaining_members
    if args.members is None:
        keys, split_hash = frozen_manifest(args.manifest)
        qid_start = 0
    else:
        keys, split_hash, member_hash, protocol_hash, addendum_hash = frozen_remaining_members(args.manifest, args.members)
        qid_start = 64
    inputs = json.loads(args.inputs.read_text())
    frozen = json.loads(args.candidates.read_text())
    candidate_hash = digest(args.candidates)
    if candidate_hash != (args.candidates.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate SHA mismatch')
    if inputs['candidate_sha256'] != candidate_hash or inputs['split_manifest_sha256'] != split_hash:
        raise ValueError('Input manifest and frozen universe differ')
    if args.members is not None and any(inputs[field] != value for field,value in [
        ('members_sha256',member_hash),('protocol_sha256',protocol_hash),
        ('evaluation_addendum_sha256',addendum_hash)]):
        raise ValueError('Remaining-76 input protocol changed')
    selected = inputs['selected_query_ids']
    expected = [f'q{i+qid_start:03}' for i in range(len(keys))
                if inputs['queries'][f'q{i+qid_start:03}']['eligible']]
    if args.members is None:
        if len(selected) != 9 or set(selected) - {f'q{i:03}' for i in range(len(keys))}:
            raise ValueError('Unexpected phase-two eligible list')
    elif selected != expected:
        raise ValueError('Frozen remaining-76 eligibility changed')
    qid_by_key = {key:f'q{i+qid_start:03}' for i,key in enumerate(keys)
                  if f'q{i+qid_start:03}' in selected}
    rows = defaultdict(list)
    columns = ['ingest_lib','normalized_smiles','ms2_mzs','ms2_normalized_intensities',
               'adduct','ionization_mode','instrument_type','precursor_mz']
    failures = {}
    for batch in pq.ParquetFile(args.data/'train.parquet').iter_batches(batch_size=4096, columns=columns):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i,source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            # Structure is read only to associate existing spectra with pre-frozen qids.
            qid = qid_by_key.get(b.canonical_key(row['normalized_smiles']))
            if qid is None or row['adduct'] != '[M+H]+' or row['ionization_mode'] != 'positive':
                continue
            try:
                precursor = float(row['precursor_mz'])
                peaks = [(float(m),float(v)) for m,v in zip(row['ms2_mzs'],row['ms2_normalized_intensities'])]
                if not math.isfinite(precursor) or precursor <= 0 or not peaks or any(
                        not math.isfinite(m) or not math.isfinite(v) or m <= 0 or v < 0 for m,v in peaks):
                    raise ValueError('invalid_precursor_or_raw_peaks')
                rows[qid].append({'peaks':peaks, 'precursor_mz':precursor,
                                  'instrument':row['instrument_type'] or 'unknown'})
            except (TypeError, ValueError) as exc:
                if args.members is None:
                    raise ValueError(f'Invalid phase-two spectrum for {qid}') from exc
                rows[qid].append(None)
                failures[qid] = 'invalid_precursor_or_raw_peaks'
    for qid in selected:
        expected = inputs['queries'][qid]
        if expected['reason'] != 'selected_mh_positive_unique_formula' or len(rows[qid]) != expected['spectrum_count_mh_positive_used']:
            raise ValueError(f'Frozen eligibility/count mismatch for {qid}')
        if args.members is not None and expected['hplus_positive_spec_ids'] != [f'{qid}_s{i:03}' for i in range(len(rows[qid]))]:
            raise ValueError(f'Frozen H+ spectrum IDs changed for {qid}')
        if not expected['neutral_formula'] or expected['subformula_cartesian_upper_bound'] > 2_000_000:
            raise ValueError(f'Unsupported mother formula for {qid}')
    runnable = [qid for qid in selected if qid not in failures]
    args.output.mkdir(parents=True,exist_ok=False)
    map_rows = []
    for qid in runnable:
        for i,row in enumerate(rows[qid]):
            map_rows.append({'spec':f'{qid}_s{i:03}', 'qid':qid,
                             'neutral_formula':inputs['queries'][qid]['neutral_formula'],
                             'ionization':'[M+H]+', 'input_peaks':len(row['peaks']),
                             'precursor_mz':row['precursor_mz'], 'instrument':row['instrument']})
    stages = [('smoke',runnable[:2]),('timing',runnable[:4]),('full',runnable)] if args.members is None else [('smoke',runnable[:2]),('full',runnable)]
    for label, qids in stages:
        allowed = set(qids)
        chosen = [r for r in map_rows if r['qid'] in allowed and (label != 'smoke' or args.members is None or r['spec'].endswith('_s000'))]
        with (args.output/f'{label}_labels.tsv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['spec','formula','ionization','dataset','compound','parentmass','instrument'],delimiter='\t')
            writer.writeheader()
            for r in chosen:
                writer.writerow({'spec':r['spec'],'formula':r['neutral_formula'], 'ionization':'[M+H]+',
                                 'dataset':'phase2_fresh_dev' if args.members is None else 'phase3_remaining_fresh_dev','compound':r['qid'],
                                 'parentmass':r['precursor_mz'],'instrument':r['instrument']})
        with (args.output/f'{label}.mgf').open('w',encoding='utf-8') as f:
            for r in chosen:
                spec_idx = int(r['spec'].split('_s')[1])
                source = rows[r['qid']][spec_idx]
                f.write(f"BEGIN IONS\nFEATURE_ID={r['spec']}\nPEPMASS={r['precursor_mz']:.12g}\nCHARGE=1+\nMSLEVEL=2\n")
                for mz,inten in source['peaks']:
                    f.write(f'{mz:.12g} {inten:.12g}\n')
                f.write('END IONS\n\n')
    (args.output/'spectra_map.json').write_text(json.dumps(map_rows,indent=2),encoding='utf-8')
    record={'status':'inputs_frozen_before_model_prediction','source_sha256':digest(__file__),
            'split_sha256':split_hash,'candidate_sha256':candidate_hash,
            'input_manifest_sha256':digest(args.inputs),'spectra':len(map_rows),
            'eligible_qids':selected,'runnable_qids':runnable,'input_failure_by_qid':failures,
            'members_sha256':None if args.members is None else member_hash,
            'protocol_sha256':None if args.members is None else protocol_hash,
            'evaluation_addendum_sha256':None if args.members is None else addendum_hash,
            'smiles_in_model_labels':False,
            'truth_formula_used':False,'sealed_scored':False,
            'preprocessing':'official MIST MGF process_spec_file, default top60/max_inten0.001; subformula 20ppm; neutral formula and [M+H]+ only'}
    (args.output/'run_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'status':record['status'],'spectra':len(map_rows),'by_qid':{q:len(rows[q]) for q in selected},'input_failures':failures}))


def infer(args):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CPU-only inference required')
    # Official pred_fp.run_fp_pred: strict state_dict load, no_grad spectra encoder,
    # allow_none_smiles=True; only worker, I/O, and CPU settings override checkpoint.
    sys.argv=['mist.pred_fp','--model-ckpt',str(args.model),'--labels-file',str(args.labels),
              '--subform-folder',str(args.subforms),'--dataset-name',args.name,
              '--save-dir',str(args.save_dir),'--num-workers','0']
    from mist.pred_fp import run_fp_pred
    run_fp_pred()



def pad_official_batch_fixed61(batch):
    """Pad only the official MIST collated peak axis to top60 plus CLS."""
    import torch.nn.functional as F

    expected = {'types', 'form_vec', 'ion_vec', 'intens', 'names',
                'num_peaks', 'instruments'}
    if set(batch) != expected:
        raise ValueError(f'Unexpected official spectrum collate fields: {set(batch)}')
    current = batch['form_vec'].shape[1]
    if current > 61 or current < int(batch['num_peaks'].max()):
        raise ValueError(f'Unsupported peak axis length {current} for fixed61')
    if any(batch[key].shape[1] != current for key in ('types', 'ion_vec', 'intens')):
        raise ValueError('Official peak axes disagree')
    amount = 61 - current
    if amount:
        batch['form_vec'] = F.pad(batch['form_vec'], (0, 0, 0, amount))
        for key in ('types', 'ion_vec', 'intens'):
            batch[key] = F.pad(batch[key], (0, amount))
    if batch['form_vec'].shape[1] != 61:
        raise ValueError('Fixed peak axis was not applied')
    return batch


@contextmanager
def fixed61_collate_scope():
    """Temporarily wrap the official static collator and restore its descriptor."""
    from mist.data import featurizers

    descriptor = featurizers.PeakFormula.__dict__['collate_fn']
    original_callable = featurizers.PeakFormula.collate_fn
    featurizers.PeakFormula.collate_fn = staticmethod(
        lambda input_list: pad_official_batch_fixed61(original_callable(input_list)))
    try:
        yield
    finally:
        featurizers.PeakFormula.collate_fn = descriptor


def infer_fixed61(args):
    """Official MIST inference with fixed all-zero peak-axis padding."""
    with fixed61_collate_scope():
        infer(args)


def inspect(args):
    import numpy as np
    import pickle
    with args.pred.open('rb') as f:
        obj=pickle.load(f)
    preds=np.asarray(obj['preds'])
    names=list(obj['names'])
    if preds.shape != (len(names),4096) or len(names) != len(set(names)) or not np.isfinite(preds).all():
        raise ValueError(f'Invalid model prediction: {preds.shape}')
    norms=np.linalg.norm(preds,axis=1)
    if not np.all(norms > 0):
        raise ValueError('Zero fingerprint prediction')
    result={'status':'valid','spectra':len(names),'shape':list(preds.shape),
            'min_norm':float(norms.min()),'max_norm':float(norms.max()),
            'pred_sha256':digest(args.pred),'names':names}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


def audit_subforms(args):
    from mist import utils
    parsed=utils.parse_spectra_mgf(args.mgf)
    result=[]
    for meta,tuples in parsed:
        name=meta['FEATURE_ID']
        processed=utils.process_spec_file(meta,tuples)  # Official MGF defaults: top60, 0.001.
        tree=json.loads((args.subforms/f'{name}.json').read_text())
        table=tree['output_tbl']
        matched=0 if table is None else len(table['formula'])
        if processed is None or matched == 0 or matched > len(processed):
            raise ValueError(f'Empty/invalid subformula assignment for {name}')
        result.append({'spec':name,'processed_peaks':len(processed),
                       'matched_subformulas':matched,'root_formula_token':tree['cand_form'],
                       'ion':tree['cand_ion']})
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


def rank_cosine(query_vectors, candidate_vectors, keys):
    import numpy as np
    q=np.asarray(query_vectors,dtype=np.float64)
    c=np.asarray(candidate_vectors,dtype=np.float64)
    if q.ndim != 2 or c.shape != (len(keys),q.shape[1]) or not len(q):
        raise ValueError('Fingerprint shape mismatch')
    qnorm=np.linalg.norm(q,axis=1)
    cnorm=np.linalg.norm(c,axis=1)
    if not np.isfinite(q).all() or not np.isfinite(c).all() or np.any(qnorm<=0) or np.any(cnorm<=0):
        raise ValueError('Invalid fingerprint values')
    # MIST official retrieval cosine, then mean across this molecule's supported spectra.
    scores=((q/qnorm[:,None]) @ (c/cnorm[:,None]).T).mean(axis=0)
    return sorted(zip(keys,map(float,scores)),key=lambda item:(-item[1],item[0]))


def rank(args):
    import numpy as np
    import pickle
    import baseline as b
    from coconut_candidates import frozen_manifest
    keys,split_hash=frozen_manifest(args.manifest)
    inputs=json.loads(args.inputs.read_text())
    frozen=json.loads(args.candidates.read_text())
    c_hash=digest(args.candidates)
    if c_hash!=(args.candidates.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate SHA mismatch')
    if inputs['candidate_sha256']!=c_hash or inputs['split_manifest_sha256']!=split_hash:
        raise ValueError('Frozen source mismatch')
    subset=json.loads(args.fp_subset.read_text())
    fp=json.loads(args.fingerprints.read_text())
    if subset['candidate_sha256']!=c_hash or subset['input_manifest_sha256']!=digest(args.inputs) or fp['candidate_subset_sha256']!=digest(args.fp_subset):
        raise ValueError('Candidate fingerprint provenance mismatch')
    with args.pred.open('rb') as f:
        model=pickle.load(f)
    names=list(model['names'])
    pred=np.asarray(model['preds'])
    specmap=json.loads(args.spectra_map.read_text())
    if pred.shape!=(len(specmap),4096) or len(names)!=len(set(names)) or set(names)!={r['spec'] for r in specmap}:
        raise ValueError('Missing/extra MIST spectra')
    qids=inputs['selected_query_ids']
    by_spec={row['spec']:row['qid'] for row in specmap}
    by_qid=defaultdict(list)
    for name,vector in zip(names,pred):
        by_qid[by_spec[name]].append(vector)
    if set(by_qid)!=set(qids):
        raise ValueError('Eligible query mismatch')
    for qid in qids:
        if len(by_qid[qid])!=inputs['queries'][qid]['spectrum_count_mh_positive_used']:
            raise ValueError(f'Model spectrum count changed for {qid}')
    if b.CONFIG['mass_ppm']!=15 or b.CONFIG['mass_floor_da']!=.003:
        raise ValueError('Frozen mass filter changed')
    mass_only,hybrid,diagnostics={},{},{} 
    expected_subset={c['key']:c['smiles'] for c in subset['candidates']}
    observed_model_pool_keys=set()
    for i in range(64):
        qid=f'q{i:03}'
        mass=frozen['query_mass_da_by_id'][qid]
        tol=max(.003,mass*15e-6)
        pool=[c for c in frozen['candidates'] if abs(c['exact_mass']-mass)<=tol]
        mass_order=sorted(pool,key=lambda c:(abs(c['exact_mass']-mass),c['key']))
        mass_only[qid]=[c['smiles'] for c in mass_order[:25]]
        if qid in by_qid:
            if len(pool)!=subset['pool_sizes'][qid] or any(expected_subset.get(c['key'])!=c['smiles'] for c in pool):
                raise ValueError(f'Frozen fingerprint pool changed for {qid}')
            observed_model_pool_keys.update(c['key'] for c in pool)
            vectors=[]
            for c in pool:
                bits=bytes.fromhex(fp['vectors_hex_by_key'][c['key']])
                if len(bits)!=512:
                    raise ValueError('Candidate fingerprint must be exactly 4096 bits')
                vector=np.unpackbits(np.frombuffer(bits,dtype=np.uint8),bitorder='big')
                if vector.shape!=(4096,) or not np.isin(vector,[0,1]).all():
                    raise ValueError('Invalid candidate binary fingerprint')
                vectors.append(vector)
            scored=rank_cosine(by_qid[qid],vectors,[c['key'] for c in pool])
            smiles={c['key']:c['smiles'] for c in pool}
            hybrid[qid]=[smiles[key] for key,_ in scored[:25]]
            diagnostics[qid]={'method':'MIST_cosine_mean_Hplus','candidate_count':len(pool),
                              'model_spectra':len(by_qid[qid])}
        else:
            hybrid[qid]=mass_only[qid]
            diagnostics[qid]={'method':'mass_only_fallback','candidate_count':len(pool),
                              'model_spectra':0,'reason':inputs['queries'][qid]['reason']}
        if len(pool)!=inputs['queries'][qid]['candidate_count']:
            raise ValueError(f'Candidate pool changed for {qid}')
    if observed_model_pool_keys!=set(expected_subset) or observed_model_pool_keys!=set(fp['vectors_hex_by_key']):
        raise ValueError('Model pool key union differs from frozen fingerprint subset')
    args.output.mkdir(parents=True,exist_ok=False)
    predictions={'mass_only':mass_only,'hybrid':hybrid,
                 'mist_on_eligible':{q:hybrid[q] for q in qids}}
    path=args.output/'predictions.json'
    path.write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    (args.output/'predictions.sha256').write_text(digest(path)+'  predictions.json\n',encoding='utf-8')
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    record={'status':'predictions_frozen_before_truth_scoring','split_sha256':split_hash,
            'candidate_sha256':c_hash,'input_manifest_sha256':digest(args.inputs),
            'fingerprints_sha256':digest(args.fingerprints),'model_predictions_sha256':digest(args.pred),
            'source_sha256':digest(__file__),'prediction_sha256':digest(path),
            'eligible_qids':qids,'candidate_method':'original representative SMILES morgan4096; cosine per spectrum then mean; stable canonical key tie break',
            'fallback_qids':64-len(qids),'truth_used_for_ranking':False,'sealed_scored':False}
    (args.output/'run_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'status':record['status'],'prediction_sha256':record['prediction_sha256'],'eligible':len(qids),'fallback':64-len(qids)}))



def rank_remaining(args):
    """Freeze all 76 predictions from the original candidate pool before reading truth."""
    import numpy as np
    import pickle
    import baseline as b
    from coconut_candidates import frozen_remaining_members
    keys, split_hash, member_hash, protocol_hash, addendum_hash = frozen_remaining_members(args.manifest, args.members)
    members = json.loads(args.members.read_text())['members']
    qids = [m['qid'] for m in members]
    inputs = json.loads(args.inputs.read_text())
    frozen = json.loads(args.candidates.read_text())
    candidate_hash = digest(args.candidates)
    if candidate_hash != (args.candidates.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate SHA changed')
    for field,value in [('split_manifest_sha256',split_hash),('members_sha256',member_hash),
                        ('protocol_sha256',protocol_hash),('evaluation_addendum_sha256',addendum_hash),
                        ('candidate_sha256',candidate_hash)]:
        if inputs[field] != value:
            raise ValueError(f'Frozen input mismatch: {field}')
    if set(inputs['queries']) != set(qids):
        raise ValueError('Input query set changed')
    eligible = inputs['selected_query_ids']
    if eligible != [q for q in qids if inputs['queries'][q]['eligible']]:
        raise ValueError('Predeclared eligible set changed')
    subset = json.loads(args.fp_subset.read_text())
    fp = json.loads(args.fingerprints.read_text())
    comparison = json.loads(args.fp_comparison.read_text())
    if subset['candidate_sha256'] != candidate_hash or subset['input_manifest_sha256'] != digest(args.inputs) or fp['candidate_subset_sha256'] != digest(args.fp_subset):
        raise ValueError('Fingerprint provenance changed')
    if (comparison['candidate_count'] != comparison['exact_equal_count'] or
            comparison['candidate_count'] != len(subset['candidates']) or
            comparison['candidate_sha256'] != candidate_hash or
            comparison['input_manifest_sha256'] != digest(args.inputs)):
        raise ValueError('Cross-version candidate fingerprint check failed')
    expected_subset = {c['key']:c['smiles'] for c in subset['candidates']}
    expected_union = set().union(*(set(inputs['queries'][q]['candidate_keys']) for q in eligible))
    if set(expected_subset) != expected_union or set(fp['vectors_hex_by_key']) != expected_union:
        raise ValueError('Eligible candidate key union changed')
    frozen_by_key = {c['key']:c for c in frozen['candidates']}
    if any(frozen_by_key[k]['smiles'] != smiles for k,smiles in expected_subset.items()):
        raise ValueError('Original candidate representative changed')
    specmap = json.loads(args.spectra_map.read_text())
    input_record = json.loads((args.spectra_map.parent/'run_record.json').read_text())
    if input_record['input_manifest_sha256'] != digest(args.inputs) or input_record['candidate_sha256'] != candidate_hash:
        raise ValueError('Frozen spectrum input provenance changed')
    expected_spec = {s:q for q in eligible for s in inputs['queries'][q]['hplus_positive_spec_ids']}
    mapped_spec = {r['spec']:r['qid'] for r in specmap}
    if len(mapped_spec) != len(specmap) or any(expected_spec.get(s) != q for s,q in mapped_spec.items()):
        raise ValueError('Unexpected or repeated MIST input spectrum')
    if set(mapped_spec) != expected_spec.keys() - {s for q in input_record['input_failure_by_qid'] for s in inputs['queries'][q]['hplus_positive_spec_ids']}:
        raise ValueError('Input failures are not whole-molecule fallbacks')
    audit = json.loads(args.subforms_audit.read_text())
    audit_by_spec = {r['spec']:r for r in audit}
    if len(audit_by_spec) != len(audit) or set(audit_by_spec) != set(mapped_spec):
        raise ValueError('Subformula audit/input spectrum set changed')
    with args.pred.open('rb') as f:
        model = pickle.load(f)
    names = list(model['names'])
    vectors = np.asarray(model['preds'])
    if len(names) != len(set(names)) or set(names) - set(mapped_spec) or vectors.shape != (len(names),4096) or not np.isfinite(vectors).all():
        raise ValueError('Invalid or unexpected MIST prediction spectra')
    by_spec = dict(zip(names,vectors))
    if b.CONFIG['mass_ppm'] != 15 or b.CONFIG['mass_floor_da'] != .003:
        raise ValueError('Final mass filter changed')
    mass_only, hybrid, diagnostics = {}, {}, {}
    runtime_failures = dict(input_record['input_failure_by_qid'])
    for qid in eligible:
        ids = inputs['queries'][qid]['hplus_positive_spec_ids']
        if qid not in runtime_failures and any(audit_by_spec[s]['matched_subformulas'] <= 0 or audit_by_spec[s]['root_formula_token'] != inputs['queries'][qid]['neutral_formula'] or audit_by_spec[s]['ion'] != '[M+H]+' for s in ids):
            runtime_failures[qid] = 'invalid_or_empty_subformula_assignment'
        if qid not in runtime_failures and any(s not in by_spec for s in ids):
            runtime_failures[qid] = 'incomplete_model_prediction'
    for qid in qids:
        row = inputs['queries'][qid]
        mass = frozen['query_mass_da_by_id'][qid]
        if mass != row['neutral_mass_da']:
            raise ValueError(f'Query mass changed for {qid}')
        tol = None if mass is None else max(.003, mass*15e-6)
        pool = [] if mass is None else [c for c in frozen['candidates'] if abs(c['exact_mass']-mass) <= tol]
        if set(c['key'] for c in pool) != set(row['candidate_keys']) or len(pool) != row['candidate_count']:
            raise ValueError(f'Final candidate pool changed for {qid}')
        mass_order = sorted(pool,key=lambda c:(abs(c['exact_mass']-mass),c['key']))
        mass_only[qid] = [c['smiles'] for c in mass_order[:25]]
        if qid in eligible and qid not in runtime_failures:
            if len(pool) != subset['pool_sizes'][qid] or any(expected_subset[c['key']] != c['smiles'] for c in pool):
                raise ValueError(f'Eligible original candidate pool changed for {qid}')
            candidate_vectors = []
            for c in pool:
                packed = bytes.fromhex(fp['vectors_hex_by_key'][c['key']])
                if len(packed) != 512:
                    raise ValueError('Candidate fingerprint must contain exactly 4096 bits')
                bits = np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='big')
                if bits.shape != (4096,) or not np.isin(bits,[0,1]).all():
                    raise ValueError('Invalid candidate fingerprint')
                candidate_vectors.append(bits)
            scored = rank_cosine([by_spec[s] for s in row['hplus_positive_spec_ids']], candidate_vectors,
                                 [c['key'] for c in pool])
            smiles = {c['key']:c['smiles'] for c in pool}
            hybrid[qid] = [smiles[key] for key,_ in scored[:25]]
            method = 'MIST_cosine_mean_Hplus'
        else:
            hybrid[qid] = mass_only[qid]
            method = 'mass_only_fallback'
        diagnostics[qid] = {'method':method,'predeclared_eligible':qid in eligible,
                            'runtime_failure':runtime_failures.get(qid),
                            'fallback_reason':runtime_failures.get(qid) or (None if qid in eligible else row['reason']),
                            'candidate_count':len(pool),'hplus_spectra':row['spectrum_count_mh_positive_used']}
    if set(mass_only) != set(qids) or set(hybrid) != set(qids):
        raise ValueError('The 76-member prediction denominator changed')
    if any(hybrid[q] != mass_only[q] for q in qids if q not in eligible or q in runtime_failures):
        raise ValueError('Fallback predictions changed')
    args.output.mkdir(parents=True,exist_ok=False)
    predictions = {'mass_only':mass_only,'hybrid':hybrid,
                   'eligible_policy':{q:hybrid[q] for q in eligible}}
    path = args.output/'predictions.json'
    path.write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    (args.output/'predictions.sha256').write_text(digest(path)+'  predictions.json\n',encoding='utf-8')
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    record = {'status':'predictions_frozen_before_truth_scoring','source_sha256':digest(__file__),
              'split_sha256':split_hash,'members_sha256':member_hash,'protocol_sha256':protocol_hash,
              'evaluation_addendum_sha256':addendum_hash,'candidate_sha256':candidate_hash,
              'input_manifest_sha256':digest(args.inputs),'fingerprint_subset_sha256':digest(args.fp_subset),
              'fingerprints_sha256':digest(args.fingerprints),'fingerprint_comparison_sha256':digest(args.fp_comparison),
              'subformula_audit_sha256':digest(args.subforms_audit),'model_predictions_sha256':digest(args.pred),
              'prediction_sha256':digest(path),'eligible_qids':eligible,'runtime_failures':runtime_failures,
              'model_success_qids':[q for q in eligible if q not in runtime_failures],
              'fallback_qids':len(qids)-len(eligible)+len(runtime_failures),'truth_used_for_ranking':False,
              'sealed_scored':False}
    (args.output/'run_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'status':record['status'],'prediction_sha256':record['prediction_sha256'],
                      'eligible':len(eligible),'model_success':len(record['model_success_qids']),
                      'fallback':record['fallback_qids']}))


def evaluate(args):
    import baseline as b
    from coconut_candidates import frozen_manifest
    from phase1_eval import load_fresh_queries
    keys,split_hash=frozen_manifest(args.manifest)
    pred_hash=digest(args.predictions)
    if pred_hash!=(args.predictions.parent/'predictions.sha256').read_text().split()[0]:
        raise ValueError('Prediction file was not frozen')
    record=json.loads((args.predictions.parent/'run_record.json').read_text())
    if record['split_sha256']!=split_hash or record['prediction_sha256']!=pred_hash:
        raise ValueError('Prediction/split identity mismatch')
    preds=json.loads(args.predictions.read_text())
    if set(preds['mass_only'])!={f'q{i:03}' for i in range(64)} or set(preds['hybrid'])!=set(preds['mass_only']):
        raise ValueError('Prediction coverage changed')
    # Truth enters only here, after the independent ranking file and SHA are frozen.
    _,truth,_=load_fresh_queries(args.data,keys)
    eligible=record['eligible_qids']
    selected_truth={q:truth[q] for q in eligible}
    metrics={'mass_only_all64':b.evaluate(truth,preds['mass_only']),
             'hybrid_all64':b.evaluate(truth,preds['hybrid']),
             'mass_only_same9':b.evaluate(selected_truth,preds['mass_only']),
             'mist_same9':b.evaluate(selected_truth,preds['mist_on_eligible'])}
    rankrows=[]
    for qid in eligible:
        rankrows.append({'qid':qid,'mass_only_rr':b.reciprocal_rank(truth[qid],preds['mass_only'][qid]),
                         'mist_rr':b.reciprocal_rank(truth[qid],preds['mist_on_eligible'][qid])})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (args.output/'eligible_per_molecule.json').write_text(json.dumps(rankrows,indent=2),encoding='utf-8')
    (args.output/'run_record.json').write_text(json.dumps({'status':'complete','split_sha256':split_hash,
        'prediction_sha256':pred_hash,'source_sha256':digest(__file__),
        'truth_loaded_after_predictions_frozen':True,'sealed_scored':False},indent=2),encoding='utf-8')
    print(json.dumps(metrics))



def evaluate_remaining(args):
    """Read truth only after verifying the frozen 76-member prediction artifact."""
    import numpy as np
    import pyarrow.parquet as pq
    import baseline as b
    from coconut_candidates import frozen_remaining_members
    keys, split_hash, member_hash, protocol_hash, addendum_hash = frozen_remaining_members(args.manifest,args.members)
    members = json.loads(args.members.read_text())['members']
    qids = [m['qid'] for m in members]
    pred_hash = digest(args.predictions)
    if pred_hash != (args.predictions.parent/'predictions.sha256').read_text().split()[0]:
        raise ValueError('Predictions were not frozen')
    record = json.loads((args.predictions.parent/'run_record.json').read_text())
    for field,value in [('prediction_sha256',pred_hash),('source_sha256',digest(__file__)),
                        ('split_sha256',split_hash),('members_sha256',member_hash),
                        ('protocol_sha256',protocol_hash),('evaluation_addendum_sha256',addendum_hash)]:
        if record[field] != value:
            raise ValueError(f'Prediction/evaluation identity mismatch: {field}')
    preds = json.loads(args.predictions.read_text())
    eligible = record['eligible_qids']
    if set(preds['mass_only']) != set(qids) or set(preds['hybrid']) != set(qids) or set(preds['eligible_policy']) != set(eligible):
        raise ValueError('Prediction denominator changed')
    if any(preds['eligible_policy'][q] != preds['hybrid'][q] for q in eligible):
        raise ValueError('Eligible policy changed')
    if any(preds['hybrid'][q] != preds['mass_only'][q] for q in qids if q not in record['model_success_qids']):
        raise ValueError('Fallback prediction list changed')
    # This is the first truth load in this stage. The ranking file and its SHA already exist.
    key_to_qid = {m['canonical_target_key']:m['qid'] for m in members}
    truth = {}
    columns = ['ingest_lib','normalized_smiles']
    for batch in pq.ParquetFile(args.data/'train.parquet').iter_batches(batch_size=4096,columns=columns):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i,source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            key = b.canonical_key(row['normalized_smiles'])
            qid = key_to_qid.get(key)
            if qid is not None:
                truth.setdefault(qid,row['normalized_smiles'])
    if set(truth) != set(qids) or any(b.canonical_key(truth[q]) != m['canonical_target_key'] for q,m in zip(qids,members)):
        raise ValueError('Frozen truth membership changed')
    inputs = json.loads(args.inputs.read_text())
    frozen = json.loads(args.candidates.read_text())
    if record['input_manifest_sha256'] != digest(args.inputs) or record['candidate_sha256'] != digest(args.candidates):
        raise ValueError('Frozen input/candidate hashes changed')
    candidate_keys = {c['key'] for c in frozen['candidates']}
    rows = []
    for qid in qids:
        key = b.canonical_key(truth[qid])
        qrow = inputs['queries'][qid]
        mass_rank = next((i for i,s in enumerate(preds['mass_only'][qid][:25],1) if b.canonical_key(s)==key),None)
        hybrid_rank = next((i for i,s in enumerate(preds['hybrid'][qid][:25],1) if b.canonical_key(s)==key),None)
        wide = key in candidate_keys
        final = key in qrow['candidate_keys']
        if final and not wide:
            raise ValueError('Final pool key missing from wide-window indexed universe')
        rows.append({'qid':qid,'predeclared_eligible':qid in eligible,
                     'model_success':qid in record['model_success_qids'],
                     'runtime_failure':record['runtime_failures'].get(qid),
                     'fallback_reason':record['runtime_failures'].get(qid) or (None if qid in eligible else qrow['reason']),
                     'mass_rank':mass_rank if mass_rank is not None else '>25',
                     'hybrid_rank':hybrid_rank if hybrid_rank is not None else '>25',
                     'mass_rr':0 if mass_rank is None else 1/mass_rank,
                     'hybrid_rr':0 if hybrid_rank is None else 1/hybrid_rank,
                     'rr_difference':(0 if hybrid_rank is None else 1/hybrid_rank)-(0 if mass_rank is None else 1/mass_rank),
                     'wide_window_indexed_truth_coverage':wide,'final_pool_truth_coverage':final,
                     'candidate_count':qrow['candidate_count'],'formula_count':qrow['formula_count'],
                     'hplus_spectrum_count':qrow['spectrum_count_mh_positive_used']})
    eligible_truth = {q:truth[q] for q in eligible}
    successful = record['model_success_qids']
    successful_truth = {q:truth[q] for q in successful}
    def outcomes(selected):
        diffs = [r['rr_difference'] for r in rows if r['qid'] in selected]
        return {'win':sum(d>0 for d in diffs),'tie':sum(d==0 for d in diffs),'loss':sum(d<0 for d in diffs)}
    counts = np.array([r['candidate_count'] for r in rows],dtype=float)
    metrics = {'mass_only_all76':b.evaluate(truth,preds['mass_only']),
               'hybrid_all76':b.evaluate(truth,preds['hybrid']),
               'mass_only_predeclared_eligible':b.evaluate(eligible_truth,preds['mass_only']),
               'hybrid_predeclared_eligible':b.evaluate(eligible_truth,preds['hybrid']),
               'successful_model_diagnostic':None if not successful else {
                   'mass_only':b.evaluate(successful_truth,preds['mass_only']),
                   'hybrid':b.evaluate(successful_truth,preds['hybrid'])},
               'predeclared_eligible_count':len(eligible),'model_success_count':len(successful),
               'all76_win_tie_loss':outcomes(set(qids)),'eligible_win_tie_loss':outcomes(set(eligible)),
               'candidate_count_quantiles':{str(q):float(np.quantile(counts,q)) for q in [0,.25,.5,.75,1]},
               'wide_window_indexed_truth_coverage':sum(r['wide_window_indexed_truth_coverage'] for r in rows),
               'final_pool_truth_coverage':sum(r['final_pool_truth_coverage'] for r in rows),
               'coverage_decomposition':{
                   'not_in_wide_window_indexed_universe':sum(not r['wide_window_indexed_truth_coverage'] for r in rows),
                   'indexed_but_lost_at_final_mass_filter':sum(r['wide_window_indexed_truth_coverage'] and not r['final_pool_truth_coverage'] for r in rows),
                   'in_final_pool_but_mass_rank_beyond25':sum(r['final_pool_truth_coverage'] and r['mass_rr']==0 for r in rows),
                   'in_final_pool_but_hybrid_rank_beyond25':sum(r['final_pool_truth_coverage'] and r['hybrid_rr']==0 for r in rows)}}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (args.output/'per_molecule.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (args.output/'run_record.json').write_text(json.dumps({
        'status':'complete','prediction_sha256':pred_hash,'prediction_source_sha256':record['source_sha256'],
        'evaluation_source_sha256':digest(__file__),'members_sha256':member_hash,
        'protocol_sha256':protocol_hash,'evaluation_addendum_sha256':addendum_hash,
        'eligible_count':len(eligible),'truth_loaded_after_predictions_frozen':True,
        'sealed_scored':False},indent=2),encoding='utf-8')
    print(json.dumps(metrics))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare')
    a.add_argument('--data',type=Path,default=Path('data'))
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--members',type=Path)
    a.add_argument('--inputs',type=Path,required=True)
    a.add_argument('--candidates',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('infer')
    a.add_argument('--model',type=Path,required=True)
    a.add_argument('--labels',type=Path,required=True)
    a.add_argument('--subforms',type=Path,required=True)
    a.add_argument('--name',required=True)
    a.add_argument('--save-dir',type=Path,required=True)
    a=sub.add_parser('infer-fixed61')
    a.add_argument('--model',type=Path,required=True)
    a.add_argument('--labels',type=Path,required=True)
    a.add_argument('--subforms',type=Path,required=True)
    a.add_argument('--name',required=True)
    a.add_argument('--save-dir',type=Path,required=True)
    a=sub.add_parser('inspect')
    a.add_argument('--pred',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('audit-subforms')
    a.add_argument('--mgf',type=Path,required=True)
    a.add_argument('--subforms',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('rank')
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--members',type=Path)
    a.add_argument('--inputs',type=Path,required=True)
    a.add_argument('--candidates',type=Path,required=True)
    a.add_argument('--fp-subset',type=Path,required=True)
    a.add_argument('--fingerprints',type=Path,required=True)
    a.add_argument('--fp-comparison',type=Path)
    a.add_argument('--subforms-audit',type=Path)
    a.add_argument('--pred',type=Path,required=True)
    a.add_argument('--spectra-map',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('evaluate')
    a.add_argument('--data',type=Path,default=Path('data'))
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--members',type=Path)
    a.add_argument('--inputs',type=Path)
    a.add_argument('--candidates',type=Path)
    a.add_argument('--predictions',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    {'prepare':prepare,'infer':infer,'infer-fixed61':infer_fixed61,'inspect':inspect,'audit-subforms':audit_subforms,
     'rank':rank_remaining if args.action == 'rank' and args.members is not None else rank,
     'evaluate':evaluate_remaining if args.action == 'evaluate' and args.members is not None else evaluate}[args.action](args)

