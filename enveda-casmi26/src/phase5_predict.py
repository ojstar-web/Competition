"""Unified fresh-dev predictor; the prepare action only groups NP spectra offline."""
import argparse, hashlib, json, math, time, resource
from collections import Counter
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def block(s,mz,peaks):
    return f'BEGIN IONS\nFEATURE_ID={s}\nPEPMASS={mz:.12g}\nCHARGE=1+\nMSLEVEL=2\n'+''.join(f'{m:.12g} {v:.12g}\n' for m,v in peaks)+'END IONS\n\n'
def prepare(a):
    import numpy as np, pyarrow.parquet as pq, baseline as b
    from formula_audit import candidate_formula
    t=time.monotonic(); prot=json.loads(a.protocol.read_text()); members=json.loads(a.members.read_text())['members']
    assert prot['status']=='frozen_before_phase5_prediction_or_truth_evaluation' and prot['members_sha256']==sha(a.members)
    assert [m['qid'] for m in members]==[f'q{i:03}' for i in range(140)]
    for field,p in [('phase1_candidates_sha256',a.primary_candidates),('phase3_candidates_sha256',a.remaining_candidates)]:
        assert prot['parents'][field]==sha(p),field
    snaps=[json.loads(a.primary_candidates.read_text()),json.loads(a.remaining_candidates.read_text())]
    lookup={m['canonical_target_key']:m['qid'] for m in members}; assert len(lookup)==140
    groups={m['qid']:{'source_spectra':0,'adduct':Counter(),'polarity':Counter(),'masses':[],'hplus':[],'failure':None} for m in members}
    cols=['ingest_lib','normalized_smiles','ms2_mzs','ms2_normalized_intensities','adduct','ionization_mode','precursor_mz','instrument_type']
    for batch in pq.ParquetFile(a.data/'train.parquet').iter_batches(batch_size=4096,columns=cols):
        ids=[i for i,v in enumerate(batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()) if v==b.CONFIG['validation_source']]
        if not ids: continue
        for r in batch.take(ids).to_pylist():
            q=lookup.get(b.canonical_key(r['normalized_smiles']))
            if q is None: continue
            g=groups[q]; g['source_spectra']+=1; g['adduct'][str(r['adduct'])]+=1; g['polarity'][str(r['ionization_mode'])]+=1
            try: mz=float(r['precursor_mz'])
            except (TypeError,ValueError): mz=float('nan')
            if r['adduct'] in b.ADDUCTS and math.isfinite(mz) and mz>0:
                mass=b.neutral_mass(mz,r['adduct'])
                if math.isfinite(mass) and mass>0: g['masses'].append(mass)
            if r['adduct']!='[M+H]+' or r['ionization_mode']!='positive': continue
            spec=f"{q}_s{len(g['hplus']):03}"
            try:
                peaks=[(float(m),float(v)) for m,v in zip(r['ms2_mzs'],r['ms2_normalized_intensities'])]
                if not math.isfinite(mz) or mz<=0 or not peaks or any(not math.isfinite(m) or not math.isfinite(v) or m<=0 or v<0 for m,v in peaks): raise ValueError('raw')
                raw=block(spec,mz,peaks)
                g['hplus'].append({'spec':spec,'precursor_mz':mz,'instrument':r['instrument_type'] or 'unknown','input_peaks':len(peaks),'raw_block_sha256':hashlib.sha256(raw.encode()).hexdigest(),'block':raw})
            except (TypeError,ValueError):
                g['hplus'].append({'spec':spec,'invalid':True}); g['failure']='invalid_precursor_or_raw_peaks'
    assert all(g['source_spectra'] for g in groups.values())
    pools={}; unions=[{},{}]
    for i,(q,g) in enumerate(groups.items()):
        j=int(i>=64); mass=float(np.median(g['masses'])) if g['masses'] else None; old=snaps[j]['query_mass_da_by_id'][q]
        if (mass is None)!=(old is None) or (mass is not None and abs(mass-old)>1e-9): raise ValueError('mass drift '+q)
        tol=None if mass is None else max(b.CONFIG['mass_floor_da'],mass*b.CONFIG['mass_ppm']*1e-6)
        pool=[c for c in snaps[j]['candidates'] if tol is not None and abs(c['exact_mass']-mass)<=tol]
        if len(pool)!=len({c['key'] for c in pool}): raise ValueError('duplicate key '+q)
        pools[q]=pool; g['neutral_mass_da']=mass; unions[j].update((c['key'],c) for c in pool)
    formulas0={}; maxerr=0.
    for k,c in sorted(unions[0].items()):
        f,e,iso=candidate_formula(c)
        if abs(e)>1e-6: raise ValueError('formula mass drift '+k)
        maxerr=max(maxerr,abs(e)); formulas0[k]={'formula':f,'isotopes':iso}
    formulas1=json.loads(a.remaining_formula_audit.read_text())['candidate_formula_by_key']
    olds=[json.loads(a.primary_formula_audit.read_text())['queries'],json.loads(a.remaining_formula_audit.read_text())['queries']]
    query={}
    for i,(q,g) in enumerate(groups.items()):
        j=int(i>=64); fs=Counter((formulas0 if j==0 else formulas1)[c['key']]['formula'] for c in pools[q]); old=olds[j][q]
        if len(pools[q])!=old['candidate_count'] or len(fs)!=old['formula_count']: raise ValueError('pool/formula drift '+q)
        if j==0 and dict(sorted(fs.items()))!=old['candidates_by_formula']: raise ValueError('primary formula drift '+q)
        if j==1 and sorted(fs)!=old['formulas']: raise ValueError('remaining formula drift '+q)
        if dict(g['adduct'])!=old['adduct_spectra'] or dict(g['polarity'])!=old['polarity_spectra']: raise ValueError('adduct drift '+q)
        ids=[s['spec'] for s in g['hplus']]
        if j==1 and ids!=old['hplus_positive_spec_ids']: raise ValueError('H+ ID drift '+q)
        query[q]={'neutral_mass_da':g['neutral_mass_da'],'source_spectra':g['source_spectra'],'legal_mass_spectra':len(g['masses']),
                  'adduct_spectra':dict(sorted(g['adduct'].items())),'polarity_spectra':dict(sorted(g['polarity'].items())),
                  'hplus_positive_spec_ids':ids,'hplus_input_failure':g['failure'],
                  'hplus_spectra':[{k:v for k,v in s.items() if k!='block'} for s in g['hplus']],
                  'candidate_count':len(pools[q]),'candidate_keys':sorted(c['key'] for c in pools[q]),
                  'formula_count':len(fs),'formulas':sorted(fs)}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'hplus.mgf').write_text(''.join(s['block'] for g in groups.values() for s in g['hplus'] if 'block' in s))
    dump(a.output/'queries.json',query); dump(a.output/'pools.json',pools); dump(a.output/'formulas_primary.json',formulas0)
    record={'status':'truth_free_all140_inputs_frozen','source_sha256':sha(__file__),'members_sha256':sha(a.members),'protocol_sha256':sha(a.protocol),
            'primary_candidates_sha256':sha(a.primary_candidates),'remaining_candidates_sha256':sha(a.remaining_candidates),
            'primary_formula_audit_sha256':sha(a.primary_formula_audit),'remaining_formula_audit_sha256':sha(a.remaining_formula_audit),
            'queries_sha256':sha(a.output/'queries.json'),'pools_sha256':sha(a.output/'pools.json'),
            'formulas_primary_sha256':sha(a.output/'formulas_primary.json'),'mgf_sha256':sha(a.output/'hplus.mgf'),
            'queries':len(query),'hplus_spectra':sum(len(g['hplus']) for g in groups.values()),
            'primary_final_pool_union':len(unions[0]),'remaining_final_pool_union':len(unions[1]),
            'max_primary_formula_mass_error_da':maxerr,'elapsed_seconds':time.monotonic()-t,
            'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'truth_fields_in_predictor_input':False,'sealed_scored':False}
    dump(a.output/'run_record.json',record); print(json.dumps(record),flush=True)

def freeze_model_inputs(a):
    """Apply one truth-free branch policy to all 140 queries and freeze model inputs."""
    import ast,csv,re
    from collections import Counter
    t=time.monotonic()
    protocol=json.loads(a.protocol.read_text())
    if protocol['members_sha256']!=sha(a.members) or protocol['status']!='frozen_before_phase5_prediction_or_truth_evaluation':
        raise ValueError('Phase-five protocol identity changed')
    query=json.loads(a.queries.read_text()); pools=json.loads(a.pools.read_text())
    formulas0=json.loads(a.formulas_primary.read_text())
    formulas1=json.loads(a.formulas_remaining.read_text())['candidate_formula_by_key']
    gate=json.loads(a.gate.read_text())
    if gate['status']!='all140_formula_gate_frozen_before_prediction_or_truth' or gate['phase5_input_queries_sha256']!=sha(a.queries) or gate['phase5_input_pools_sha256']!=sha(a.pools):
        raise ValueError('Gate/input identity changed')
    if list(query)!=[f'q{i:03}' for i in range(140)] or set(query)!=set(pools):
        raise ValueError('All140 query/pool denominator changed')
    valid_source=Path('references/mist_main_v2/src/mist/utils/chem_utils.py')
    valid=None
    for node in ast.parse(valid_source.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='VALID_ELEMENTS' for x in node.targets):
            valid=set(ast.literal_eval(node.value)); break
    if not valid: raise ValueError('Official valid elements unavailable')
    token=re.compile(r'(?:\[(\d+)([A-Z][a-z]*)\]|([A-Z][a-z]*))(\d*)')
    def support(formula):
        pos=0; counts=Counter(); isotope=False
        while pos<len(formula):
            m=token.match(formula,pos)
            if not m: return False,None,'formula_syntax'
            iso,sym,plain,n=m.groups()
            counts[sym or plain]+=int(n or 1); isotope|=iso is not None; pos=m.end()
        if isotope: return False,None,'formula_isotope'
        if set(counts)-valid: return False,None,'unsupported_element'
        bound=math.prod(n+1 for n in counts.values())
        if bound>2_000_000: return False,bound,'formula_budget'
        return True,bound,None
    raw={}
    for part in a.mgf.read_text().split('BEGIN IONS\n')[1:]:
        content='BEGIN IONS\n'+part.split('END IONS\n')[0]+'END IONS\n\n'
        ids=[line.split('=',1)[1] for line in content.splitlines() if line.startswith('FEATURE_ID=')]
        if len(ids)!=1 or ids[0] in raw: raise ValueError('MGF duplicate name')
        raw[ids[0]]=content
    if set(raw)!={x['spec'] for row in query.values() for x in row['hplus_spectra'] if not x.get('invalid')}:
        raise ValueError('Raw MGF/query map mismatch')
    policy={}; chosen={}; all_candidates={}
    for i,(qid,row) in enumerate(query.items()):
        pool=pools[qid]; fm=formulas0 if i<64 else formulas1
        if len(pool)!=row['candidate_count'] or sorted(c['key'] for c in pool)!=row['candidate_keys']:
            raise ValueError('Full candidate pool changed '+qid)
        formula_groups=Counter(fm[c['key']]['formula'] for c in pool)
        if sorted(formula_groups)!=row['formulas'] or len(formula_groups)!=row['formula_count']:
            raise ValueError('Formula grouping changed '+qid)
        ids=row['hplus_positive_spec_ids']
        if ids!=[x['spec'] for x in row['hplus_spectra']]:
            raise ValueError('H+ order changed '+qid)
        status='mass_only_fallback'; reason=None; formula=None; keys=[]
        if row['neutral_mass_da'] is None: reason='no_supported_mass'
        elif not pool: reason='empty_pool'
        elif len(pool)==1: reason='single_candidate'
        elif not ids: reason='no_supported_hplus'
        elif row['hplus_input_failure']: reason=row['hplus_input_failure']
        elif len(formula_groups)==1:
            formula=next(iter(formula_groups)); good,bound,why=support(formula)
            if any(fm[c['key']]['isotopes'] for c in pool): good=False; why='candidate_isotope'
            if good: status='single_formula_model'; keys=[c['key'] for c in pool]
            else: reason=why
        else:
            choice=gate['queries'].get(qid)
            if choice is None: reason='incomplete_multiformula_support_or_budget'
            elif choice['status']!='gate_pass': reason=choice['status']
            else:
                formula=choice['top_formula']; keys=[c['key'] for c in pool if fm[c['key']]['formula']==formula]
                if set(keys)!=set(choice['selected_candidate_keys']) or len(keys)!=len(choice['selected_candidate_keys']) or ids!=choice['all_hplus_spec_ids']:
                    raise ValueError('Selected complete formula pool/H+ changed '+qid)
                if len(keys)==1: status='multi_gate_single_candidate'
                else: status='multi_gate_model'
        if status in ('single_formula_model','multi_gate_model'):
            if any(x.get('invalid') for x in row['hplus_spectra']): raise ValueError('Invalid selected H+ raw spectrum '+qid)
            for spec in row['hplus_spectra']:
                if sha_bytes(raw[spec['spec']].encode())!=spec['raw_block_sha256']:
                    raise ValueError('Raw H+ block drift '+spec['spec'])
            for c in pool:
                if c['key'] not in keys: continue
                prior=all_candidates.get(c['key'])
                if prior is not None and (prior['smiles'],prior['exact_mass'])!=(c['smiles'],c['exact_mass']):
                    raise ValueError('Shared canonical key has conflicting representative '+c['key'])
                all_candidates[c['key']]=c
            chosen[qid]=(formula,ids,keys)
        policy[qid]={'status':status,'fallback_reason':reason,'formula':formula,
                     'selected_candidate_keys':keys,'hplus_positive_spec_ids':ids,
                     'complete_candidate_count':len(pool),'formula_count':len(formula_groups)}
    old_single=set(json.loads(a.phase2_inputs.read_text())['selected_query_ids'])|set(json.loads(a.phase3_inputs.read_text())['selected_query_ids'])
    selected_single={q for q,row in policy.items() if row['status']=='single_formula_model'}
    if selected_single!=old_single: raise ValueError('Frozen old9+6 single-formula policy drift')
    a.output.mkdir(parents=True,exist_ok=False)
    dump(a.output/'policy.json',policy)
    subset={'candidate_sha256':sha(a.pools),'input_manifest_sha256':sha(a.output/'policy.json'),
            'selection_sha256':sha(a.gate),
            'pool_sizes':{q:len(keys) for q,(_,_,keys) in chosen.items()},
            'candidates':[{'key':key,'smiles':all_candidates[key]['smiles']} for key in sorted(all_candidates)]}
    dump(a.output/'candidate_subset.json',subset)
    mapping=[]
    with (a.output/'model.mgf').open('w') as mgf, (a.output/'model_labels.tsv').open('w',newline='') as labels:
        writer=csv.DictWriter(labels,fieldnames=['spec','formula','ionization','dataset','compound','parentmass','instrument'],delimiter='\t')
        writer.writeheader()
        for qid,(formula,ids,keys) in chosen.items():
            by_spec={x['spec']:x for x in query[qid]['hplus_spectra']}
            for spec in ids:
                row=by_spec[spec]; mgf.write(raw[spec])
                writer.writerow({'spec':spec,'formula':formula,'ionization':'[M+H]+',
                    'dataset':'phase5_fresh_dev','compound':qid,'parentmass':row['precursor_mz'],
                    'instrument':row['instrument']})
                mapping.append({'qid':qid,'spec':spec,'neutral_formula':formula,
                    'raw_block_sha256':row['raw_block_sha256'],'input_peaks':row['input_peaks'],
                    'precursor_mz':row['precursor_mz'],'instrument':row['instrument']})
    dump(a.output/'model_spectra_map.json',mapping)
    if len(mapping)!=len({x['spec'] for x in mapping}) or set(chosen)!={x['qid'] for x in mapping}:
        raise ValueError('Selected model spec mapping incomplete')
    record={'status':'all140_model_inputs_frozen_before_model_or_truth',
            'source_sha256':sha(__file__),'phase5_protocol_sha256':sha(a.protocol),
            'gate_sha256':sha(a.gate),'queries_sha256':sha(a.queries),
            'pools_sha256':sha(a.pools),'raw_mgf_sha256':sha(a.mgf),
            'official_element_source_sha256':sha(valid_source),
            'policy_sha256':sha(a.output/'policy.json'),
            'candidate_subset_sha256':sha(a.output/'candidate_subset.json'),
            'model_mgf_sha256':sha(a.output/'model.mgf'),
            'labels_sha256':sha(a.output/'model_labels.tsv'),
            'spectra_map_sha256':sha(a.output/'model_spectra_map.json'),
            'selected_model_qids':list(chosen),'selected_model_spectra':len(mapping),
            'unique_selected_candidates':len(all_candidates),
            'policy_status_counts':dict(Counter(x['status'] for x in policy.values())),
            'elapsed_seconds':time.monotonic()-t,
            'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'truth_fields_in_model_labels':False,'sealed_scored':False}
    dump(a.output/'run_record.json',record)
    print(json.dumps(record),flush=True)

def sha_bytes(data): return hashlib.sha256(data).hexdigest()

def predict_query(qid, spectrum_inputs, candidates, candidate_formula_by_key,
                  fixed_policy, query_vectors_by_spec, candidate_fingerprints_hex):
    """Rank one frozen mass pool using only inference-visible and candidate fields."""
    import numpy as np
    from mist_phase2 import rank_cosine

    mass=spectrum_inputs['neutral_mass_da']
    if mass is None and candidates: raise ValueError(f'Candidate pool without legal mass: {qid}')
    if mass is not None and (not math.isfinite(mass) or mass<=0): raise ValueError(f'Invalid neutral mass: {qid}')
    if len(candidates)!=len({c['key'] for c in candidates}): raise ValueError(f'Duplicate pool key: {qid}')
    for c in candidates:
        if c['key'] not in candidate_formula_by_key or not c['smiles'] or not math.isfinite(c['exact_mass']):
            raise ValueError(f'Incomplete candidate: {qid}')
    mass_order=[] if mass is None else sorted(candidates,key=lambda c:(abs(c['exact_mass']-mass),c['key']))
    mass_keys=[c['key'] for c in mass_order]
    by_key={c['key']:c for c in candidates}
    status=fixed_policy['status']
    selected_keys=fixed_policy['selected_candidate_keys']
    if status not in {'mass_only_fallback','single_formula_model','multi_gate_model','multi_gate_single_candidate'}:
        raise ValueError(f'Unknown fixed policy: {qid}')
    if status=='mass_only_fallback':
        if selected_keys: raise ValueError(f'Fallback selected a formula: {qid}')
        fm_keys=mass_keys
        failure=fixed_policy['fallback_reason']
    else:
        formula=fixed_policy['formula']
        selected_set={c['key'] for c in candidates if candidate_formula_by_key[c['key']]['formula']==formula}
        if not selected_set or selected_set!=set(selected_keys) or len(selected_set)!=len(selected_keys):
            raise ValueError(f'Incomplete selected formula candidate group: {qid}')
        if status=='single_formula_model' and selected_set!=set(mass_keys):
            raise ValueError(f'Single-formula branch omitted candidate: {qid}')
        if status=='multi_gate_single_candidate' and len(selected_set)!=1:
            raise ValueError(f'Unexpected direct-rank formula size: {qid}')
        ids=fixed_policy['hplus_positive_spec_ids']
        if ids!=spectrum_inputs['hplus_positive_spec_ids'] or len(ids)!=len(set(ids)) or not ids:
            raise ValueError(f'Incomplete H+ spectrum selection: {qid}')
        if status=='multi_gate_single_candidate':
            selected_rank=list(selected_set)
            failure=None
        elif any(spec not in query_vectors_by_spec for spec in ids) or any(key not in candidate_fingerprints_hex for key in selected_set):
            selected_rank=[]
            failure='incomplete_model_input_or_fingerprint'
        else:
            selected=[c for c in mass_order if c['key'] in selected_set]
            vectors=[]
            for c in selected:
                packed=bytes.fromhex(candidate_fingerprints_hex[c['key']])
                if len(packed)!=512: raise ValueError(f'Fingerprint length changed: {c["key"]}')
                bits=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='big')
                if bits.shape!=(4096,): raise ValueError(f'Fingerprint bit count changed: {c["key"]}')
                vectors.append(bits)
            qvec=[query_vectors_by_spec[spec] for spec in ids]
            selected_rank=[key for key,_ in rank_cosine(qvec,vectors,[c['key'] for c in selected])]
            failure=None
        if failure is not None:
            fm_keys=mass_keys  # Whole-query fallback; never omit a failed H+ spectrum.
        else:
            fm_keys=selected_rank+[key for key in mass_keys if key not in selected_set]
    if len(fm_keys)!=len(mass_keys) or set(fm_keys)!=set(mass_keys):
        raise ValueError(f'Predicted order lost a candidate: {qid}')
    single_keys=fm_keys if status=='single_formula_model' and failure is None else mass_keys
    return {'mass_only':[by_key[key]['smiles'] for key in mass_keys[:25]],
            'single_formula_hybrid':[by_key[key]['smiles'] for key in single_keys[:25]],
            'fm':[by_key[key]['smiles'] for key in fm_keys[:25]],
            'diagnostic':{'policy_status':status,'selected_formula':fixed_policy['formula'],
                          'runtime_failure':failure if status!='mass_only_fallback' else None,
                          'fallback_reason':failure or fixed_policy['fallback_reason'],
                          'candidate_count':len(candidates),'selected_formula_candidate_count':len(selected_keys),
                          'hplus_spectrum_count':len(fixed_policy['hplus_positive_spec_ids']),
                          'full_mass_key_order':mass_keys,'full_fm_key_order':fm_keys}}


def rank_fixed_candidate(a):
    """Freeze all140 truth-free lists under the separate fixed61 candidate protocol."""
    import numpy as np,pickle
    t=time.monotonic()
    protocol=json.loads(a.candidate_protocol.read_text())
    parent=json.loads(a.phase5_protocol.read_text())
    if protocol['status']!='frozen_candidate_protocol_before_fixed61_inference_or_truth' or protocol['parent_phase5_protocol_sha256']!=sha(a.phase5_protocol):
        raise ValueError('Fixed61 candidate protocol identity changed')
    if parent['status']!='frozen_before_phase5_prediction_or_truth_evaluation':
        raise ValueError('Parent phase5 protocol identity changed')
    if protocol['source_sha256']!=sha(a.infer_source):
        raise ValueError('Fixed61 inference wrapper changed')
    for field,path in [('labels',a.labels),('mgf',a.mgf),('candidate_subset',a.subset),('policy',a.policy)]:
        if protocol['input_sha256'][field]!=sha(path): raise ValueError(f'Frozen fixed61 input changed: {field}')
    q=json.loads(a.queries.read_text()); pools=json.loads(a.pools.read_text())
    policy=json.loads(a.policy.read_text()); f0=json.loads(a.formulas_primary.read_text())
    f1=json.loads(a.formulas_remaining.read_text())['candidate_formula_by_key']
    fingerprints=json.loads(a.fingerprints.read_text()); comparison=json.loads(a.fingerprint_comparison.read_text())
    specs=json.loads(a.spectra_map.read_text()); audits=json.loads(a.subforms_audit.read_text())
    with a.pred.open('rb') as f: model=pickle.load(f)
    names=list(model['names']); vectors=np.asarray(model['preds'])
    if vectors.shape!=(66,4096) or len(names)!=len(set(names)) or set(names)!={x['spec'] for x in specs} or not np.isfinite(vectors).all():
        raise ValueError('Fixed61 model prediction incomplete')
    if comparison['max_different_bits']!=0 or comparison['exact_equal_count']!=832 or comparison['candidate_count']!=832:
        raise ValueError('Candidate fingerprint cross-version check changed')
    vector_check=json.loads(a.vector_check.read_text())
    if vector_check['status']!='valid' or vector_check['candidate_vector_sha256']!=sha(a.pred) or vector_check['protocol_sha256']!=sha(a.candidate_protocol) or vector_check['wrapper_sha256']!=sha(a.infer_source):
        raise ValueError('Fixed61 vector provenance changed')
    if fingerprints['candidate_subset_sha256']!=sha(a.subset):
        raise ValueError('Candidate fingerprint input changed')
    if len(audits)!=66 or {x['spec'] for x in audits}!=set(names) or any(x['matched_subformulas']<=0 for x in audits):
        raise ValueError('Official subformula audit incomplete')
    if list(q)!=[f'q{i:03}' for i in range(140)] or set(q)!=set(pools)!=set(policy):
        raise ValueError('Fresh-dev denominator changed')
    by_spec=dict(zip(names,vectors)); by_audit={x['spec']:x for x in audits}
    for x in specs:
        p=policy[x['qid']]
        if x['spec'] not in p['hplus_positive_spec_ids'] or x['neutral_formula']!=p['formula'] or by_audit[x['spec']]['root_formula_token']!=p['formula'] or by_audit[x['spec']]['ion']!='[M+H]+':
            raise ValueError('Model spectrum/formula assignment changed')
    mass_out={}; single_out={}; fm_out={}; detail={}
    for i,qid in enumerate(q):
        formula_map=f0 if i<64 else f1
        result=predict_query(qid,q[qid],pools[qid],formula_map,policy[qid],by_spec,fingerprints['vectors_hex_by_key'])
        mass_out[qid]=result['mass_only']; single_out[qid]=result['single_formula_hybrid']; fm_out[qid]=result['fm']
        detail[qid]=result['diagnostic']
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'predictions.json'
    dump(path,{'mass_only':mass_out,'single_formula_hybrid':single_out,'fm':fm_out})
    (a.output/'predictions.sha256').write_text(sha(path)+'  predictions.json\n')
    dump(a.output/'diagnostics.json',detail)
    record={'status':'fixed61_candidate_predictions_frozen_before_history_comparison_or_truth',
            'candidate_protocol_sha256':sha(a.candidate_protocol),'parent_protocol_sha256':sha(a.phase5_protocol),
            'source_sha256':sha(__file__),'wrapper_source_sha256':sha(a.infer_source),
            'queries_sha256':sha(a.queries),'pools_sha256':sha(a.pools),
            'formulas_primary_sha256':sha(a.formulas_primary),'formulas_remaining_sha256':sha(a.formulas_remaining),
            'policy_sha256':sha(a.policy),'fingerprints_sha256':sha(a.fingerprints),
            'fingerprint_comparison_sha256':sha(a.fingerprint_comparison),
            'model_prediction_sha256':sha(a.pred),'spectra_map_sha256':sha(a.spectra_map),
            'subformula_audit_sha256':sha(a.subforms_audit),'vector_check_sha256':sha(a.vector_check),
            'predictions_sha256':sha(path),
            'diagnostics_sha256':sha(a.output/'diagnostics.json'),
            'runtime_failure_qids':[qid for qid,x in detail.items() if x['runtime_failure']],
            'qid_count':140,'model_spectra':66,'truth_used':False,'sealed_scored':False,
            'elapsed_seconds':time.monotonic()-t,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    dump(a.output/'run_record.json',record)
    print(json.dumps({'status':record['status'],'predictions_sha256':record['predictions_sha256'],
                      'runtime_failures':record['runtime_failure_qids'],'qid_count':140}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='action',required=True); s=sub.add_parser('prepare')
    for n in ['data','members','protocol','primary_candidates','remaining_candidates','primary_formula_audit','remaining_formula_audit','output']:
        s.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    s=sub.add_parser('freeze-model-inputs')
    for n in ['members','protocol','queries','pools','formulas_primary','formulas_remaining','gate','mgf','phase2_inputs','phase3_inputs','output']:
        s.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    s=sub.add_parser('rank-fixed-candidate')
    for n in ['candidate_protocol','phase5_protocol','infer_source','queries','pools','policy','formulas_primary','formulas_remaining','fingerprints','fingerprint_comparison','subset','spectra_map','subforms_audit','vector_check','pred','labels','mgf','output']:
        s.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    a=p.parse_args(); {'prepare':prepare,'freeze-model-inputs':freeze_model_inputs,'rank-fixed-candidate':rank_fixed_candidate}[a.action](a)
