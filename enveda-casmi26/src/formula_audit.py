"""Audit neutral formula ambiguity in frozen COCONUT mass pools."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import baseline as b
from coconut_candidates import frozen_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_formula(candidate):
    mol = Chem.MolFromSmiles(candidate['smiles'])
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        raise ValueError(f"Invalid frozen candidate {candidate['key']}")
    isotopes = sorted({f"{atom.GetSymbol()}-{atom.GetIsotope()}" for atom in mol.GetAtoms()
                       if atom.GetIsotope()})
    Chem.RemoveStereochemistry(mol)
    canonical = b.TAUTOMERS.Canonicalize(mol)
    key = Chem.MolToInchiKey(canonical)[:14]
    if key != candidate['key']:
        raise ValueError(f"Frozen candidate key changed: {candidate['key']} -> {key}")
    mass_error = Descriptors.ExactMolWt(canonical) - candidate['exact_mass']
    return rdMolDescriptors.CalcMolFormula(canonical, separateIsotopes=True, abbreviateHIsotopes=False), mass_error, isotopes


def query_metadata(data, keys):
    key_to_id = {key: f'q{i:03}' for i, key in enumerate(keys)}
    groups = defaultdict(list)
    columns = ['ingest_lib', 'normalized_smiles', 'precursor_mz', 'adduct', 'ionization_mode']
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096, columns=columns):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i, source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            qid = key_to_id.get(b.canonical_key(row['normalized_smiles']))
            if qid is None or row['adduct'] not in b.ADDUCTS:
                continue
            groups[qid].append({'neutral_mass': b.neutral_mass(row['precursor_mz'], row['adduct']),
                                'adduct': row['adduct'], 'polarity': row['ionization_mode']})
    if len(groups) != len(keys):
        raise ValueError(f'Only {len(groups)}/{len(keys)} frozen query structures found')
    return groups


def audit(data, manifest_path, candidates_path, output):
    start = time.monotonic()
    keys, manifest_hash = frozen_manifest(manifest_path)
    candidate_hash = sha256(candidates_path)
    if candidate_hash != (candidates_path.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Frozen candidate SHA256 mismatch')
    frozen = json.loads(candidates_path.read_text())
    if frozen['manifest_sha256'] != manifest_hash:
        raise ValueError('Candidate/manifest binding changed')
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit version required')
    groups = query_metadata(data, keys)
    formulas = {}
    isotope_candidates = {}
    max_abs_mass_error = 0.
    for candidate in frozen['candidates']:
        formula, mass_error, isotopes = candidate_formula(candidate)
        formulas[candidate['key']] = formula
        if isotopes:
            isotope_candidates[candidate['key']] = isotopes
        max_abs_mass_error = max(max_abs_mass_error, abs(mass_error))
    if max_abs_mass_error > 1e-6:
        raise ValueError(f'Canonical formula mass mismatch: max {max_abs_mass_error:.9g} Da')
    rows = {}
    for qid, spectra in sorted(groups.items()):
        mass = float(np.median([row['neutral_mass'] for row in spectra]))
        if abs(mass-frozen['query_mass_da_by_id'][qid]) > 1e-9:
            raise ValueError(f'Query mass changed: {qid}')
        tolerance = max(b.CONFIG['mass_floor_da'], mass*b.CONFIG['mass_ppm']*1e-6)
        pool = [c for c in frozen['candidates'] if abs(c['exact_mass']-mass) <= tolerance]
        counts = Counter(formulas[c['key']] for c in pool)
        rows[qid] = {'neutral_mass_da': mass, 'candidate_count': len(pool),
                     'formula_count': len(counts), 'candidates_by_formula': dict(sorted(counts.items())),
                     'adduct_spectra': dict(sorted(Counter(row['adduct'] for row in spectra).items())),
                     'polarity_spectra': dict(sorted(Counter(str(row['polarity']) for row in spectra).items())),
                     'isotope_candidate_count': sum(c['key'] in isotope_candidates for c in pool)}
    buckets = Counter('4+' if r['formula_count'] >= 4 else str(r['formula_count']) for r in rows.values())
    summary = {'formula_group_counts': {label: buckets[label] for label in ['0', '1', '2', '3', '4+']},
               'single_formula_two_or_more_candidates': sum(r['formula_count'] == 1 and
                      r['candidate_count'] >= 2 for r in rows.values()),
               'empty_pools': buckets['0'],
               'single_formula_mh_positive': sum(r['formula_count'] == 1 and
                      r['adduct_spectra'].get('[M+H]+', 0) > 0 and
                      r['polarity_spectra'].get('positive', 0) > 0 for r in rows.values()),
               'isotope_candidates_in_frozen_universe': len(isotope_candidates),
               'queries_with_isotope_candidates': sum(r['isotope_candidate_count'] > 0 for r in rows.values()),
               'max_canonical_mass_difference_da': max_abs_mass_error}
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'complete', 'server': 'net-4090d', 'hostname': platform.node(),
              'user': 'yuanyn', 'python': platform.python_version(), 'rdkit': b.RDKIT_VERSION,
              'manifest_sha256': manifest_hash, 'candidate_sha256': candidate_hash,
              'source_sha256': sha256(Path(__file__)), 'query_count': len(rows),
              'query_spectra': sum(len(v) for v in groups.values()),
              'mass_ppm': b.CONFIG['mass_ppm'], 'mass_floor_da': b.CONFIG['mass_floor_da'],
              'cpu_threads_limit': 2, 'virtual_memory_limit_gib': 12, 'single_job_timeout_s': 900,
              'elapsed_s': time.monotonic()-start,
              'maxrss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'sealed_scored': False,
              'note': 'Formulas come only from frozen candidate SMILES. Query truth formula is unused.'}
    b.write_json(output/'formula_audit.json', {'summary': summary, 'queries': rows,
                                               'isotope_candidates': isotope_candidates})
    b.write_json(output/'run_record.json', record)
    print(json.dumps({'summary': summary, 'record': record}, indent=2))


def audit_remaining(data, manifest_path, members_path, candidates_path, output):
    """Freeze candidate-derived formulas and branch eligibility for the remaining 76."""
    import ast
    import math
    import re
    from coconut_candidates import frozen_remaining_members
    started = time.monotonic()
    keys, split_hash, member_hash, protocol_hash, addendum_hash = frozen_remaining_members(manifest_path, members_path)
    content = candidates_path.read_bytes()
    candidate_hash = hashlib.sha256(content).hexdigest()
    if candidate_hash != (candidates_path.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate SHA mismatch')
    frozen = json.loads(content)
    if any(frozen[field] != value for field,value in [
        ('manifest_sha256',split_hash),('members_sha256',member_hash),
        ('protocol_sha256',protocol_hash),('evaluation_addendum_sha256',addendum_hash)]):
        raise ValueError('Candidate protocol binding changed')
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit version required')
    members = json.loads(members_path.read_text())['members']
    key_to_id = {m['canonical_target_key']:m['qid'] for m in members}
    groups = {m['qid']:{'source_spectra':0,'adduct_spectra':Counter(),
                        'polarity_spectra':Counter(),'hplus_positive_ids':[],
                        'legal_mass_spectra':0}
              for m in members}
    columns=['ingest_lib','normalized_smiles','precursor_mz','adduct','ionization_mode']
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096, columns=columns):
        sources=batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices=[i for i,source in enumerate(sources) if source==b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            qid=key_to_id.get(b.canonical_key(row['normalized_smiles']))
            if qid is None:
                continue
            g=groups[qid]
            g['source_spectra']+=1
            g['adduct_spectra'][str(row['adduct'])]+=1
            g['polarity_spectra'][str(row['ionization_mode'])]+=1
            if row['adduct']=='[M+H]+' and row['ionization_mode']=='positive':
                g['hplus_positive_ids'].append(f"{qid}_s{len(g['hplus_positive_ids']):03}")
            mz=row['precursor_mz']
            if row['adduct'] in b.ADDUCTS and mz is not None and math.isfinite(mz) and mz>0:
                mass=b.neutral_mass(mz,row['adduct'])
                if math.isfinite(mass) and mass>0:
                    g['legal_mass_spectra']+=1
    query_mass_audit=json.loads((candidates_path.parent/'query_mass_audit.json').read_text())
    for qid,g in groups.items():
        if g['source_spectra']==0 or g['source_spectra']!=query_mass_audit[qid]['source_spectra'] or g['legal_mass_spectra']!=query_mass_audit[qid]['legal_mass_spectra']:
            raise ValueError(f'Query metadata disagrees with candidate build: {qid}')
    candidate_by_key={c['key']:c for c in frozen['candidates']}
    pools={}
    union=set()
    for qid in groups:
        mass=frozen['query_mass_da_by_id'][qid]
        tolerance=None if mass is None else max(b.CONFIG['mass_floor_da'],mass*b.CONFIG['mass_ppm']*1e-6)
        pool=sorted([c['key'] for c in frozen['candidates'] if mass is not None and abs(c['exact_mass']-mass)<=tolerance])
        pools[qid]=pool
        union.update(pool)
    source=Path('references/mist_main_v2/src/mist/utils/chem_utils.py')
    tree=ast.parse(source.read_text())
    valid=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALID_ELEMENTS' for t in node.targets):
            valid=set(ast.literal_eval(node.value))
            break
    if not valid:
        raise ValueError('Cannot read official MIST valid elements')
    formula_by_key={}
    max_mass_error=0.
    for i,key in enumerate(sorted(union),1):
        formula,mass_error,isotopes=candidate_formula(candidate_by_key[key])
        formula_by_key[key]={'formula':formula,'isotopes':isotopes}
        max_mass_error=max(max_mass_error,abs(mass_error))
        if i%1000==0:
            print(json.dumps({'stage':'formula_union','done':i,'total':len(union),'elapsed_seconds':time.monotonic()-started}),flush=True)
    if max_mass_error>1e-6:
        raise ValueError(f'Canonical mass/formula mismatch: {max_mass_error:.9g} Da')
    priority=['no_supported_mass','empty_pool','single_candidate','formula_ambiguous',
              'isotope','unsupported_element','formula_budget','no_supported_hplus']
    query_rows={}
    selected=[]
    for qid,g in groups.items():
        pool=pools[qid]
        formulas=sorted({formula_by_key[k]['formula'] for k in pool})
        isotope_keys=[k for k in pool if formula_by_key[k]['isotopes']]
        unsupported=set()
        bound=None
        if len(formulas)==1:
            tokens=re.findall(r'([A-Z][a-z]*)([0-9]*)',formulas[0])
            if ''.join(el+n for el,n in tokens)!=formulas[0]:
                unsupported.add('formula_syntax')
            unsupported.update(el for el,_ in tokens if el not in valid)
            if not unsupported and not isotope_keys:
                bound=math.prod(int(n or 1)+1 for _,n in tokens)
        reasons=[]
        mass=frozen['query_mass_da_by_id'][qid]
        if mass is None: reasons.append('no_supported_mass')
        if not pool: reasons.append('empty_pool')
        if len(pool)==1: reasons.append('single_candidate')
        if len(formulas)>1: reasons.append('formula_ambiguous')
        if isotope_keys: reasons.append('isotope')
        if unsupported: reasons.append('unsupported_element')
        if bound is not None and bound>2_000_000: reasons.append('formula_budget')
        if not g['hplus_positive_ids']: reasons.append('no_supported_hplus')
        eligible=not reasons
        if eligible:
            selected.append(qid)
        query_rows[qid]={'neutral_mass_da':mass,'candidate_count':len(pool),
            'candidate_keys':pool,'formula_count':len(formulas),'formulas':formulas,
            'neutral_formula':formulas[0] if len(formulas)==1 else None,
            'isotope_candidate_keys':isotope_keys,'unsupported_elements':sorted(unsupported),
            'subformula_cartesian_upper_bound':bound,
            'source_spectra':g['source_spectra'],'legal_mass_spectra':g['legal_mass_spectra'],
            'adduct_spectra':dict(sorted(g['adduct_spectra'].items())),
            'polarity_spectra':dict(sorted(g['polarity_spectra'].items())),
            'hplus_positive_spec_ids':g['hplus_positive_ids'],
            'spectrum_count_mh_positive_used':len(g['hplus_positive_ids']),
            'eligible':eligible,'reason':'selected_mh_positive_unique_formula' if eligible else next(r for r in priority if r in reasons),
            'all_reasons':reasons}
    buckets=Counter('4+' if r['formula_count']>=4 else str(r['formula_count']) for r in query_rows.values())
    summary={'formula_group_counts':{k:buckets[k] for k in ['0','1','2','3','4+']},
             'queries':len(query_rows),'final_pool_unique_candidates':len(union),
             'eligible_structures':len(selected),'eligible_hplus_spectra':sum(query_rows[q]['spectrum_count_mh_positive_used'] for q in selected),
             'primary_reason_counts':dict(Counter(r['reason'] for r in query_rows.values())),
             'max_canonical_mass_difference_da':max_mass_error}
    output.mkdir(parents=True,exist_ok=False)
    b.write_json(output/'formula_audit.json',{'summary':summary,'queries':query_rows,
                 'candidate_formula_by_key':formula_by_key,
                 'official_valid_elements_source_sha256':sha256(source)})
    manifest={'status':'frozen_before_model_predictions_or_truth_evaluation',
              'split_manifest_sha256':split_hash,'members_sha256':member_hash,
              'protocol_sha256':protocol_hash,'evaluation_addendum_sha256':addendum_hash,
              'candidate_sha256':candidate_hash,'selected_query_ids':selected,
              'queries':query_rows,'truth_formula_used':False,'sealed_scored':False}
    b.write_json(output/'model_input_manifest.json',manifest)
    record={'status':'complete','server':platform.node(),'source_sha256':sha256(Path(__file__)),
            'candidate_sha256':candidate_hash,'members_sha256':member_hash,
            'protocol_sha256':protocol_hash,'evaluation_addendum_sha256':addendum_hash,
            'model_input_manifest_sha256':sha256(output/'model_input_manifest.json'),
            'official_valid_elements_source_sha256':sha256(source),
            'elapsed_seconds':time.monotonic()-started,
            'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'cpu_threads':2,'virtual_memory_limit_gib':12,'timeout_seconds':900,'gpu':'none',
            'sealed_scored':False}
    b.write_json(output/'run_record.json',record)
    print(json.dumps({'summary':summary,'manifest_sha256':record['model_input_manifest_sha256']}),flush=True)



def audit_ambiguous_metadata(manifest_path, phase3_members_path, phase4_members_path,
                             candidates_path, input_path, formula_cache_path, output):
    """Truth-free per-formula metadata from frozen phase-three final mass pools."""
    import ast
    import math
    import re
    from coconut_candidates import frozen_remaining_members
    started = time.monotonic()
    _, split_hash, parent_member_hash, parent_protocol_hash, addendum_hash = frozen_remaining_members(
        manifest_path, phase3_members_path)
    candidates_hash, inputs_hash = sha256(candidates_path), sha256(input_path)
    frozen = json.loads(candidates_path.read_text())
    inputs = json.loads(input_path.read_text())
    members_hash = sha256(phase4_members_path)
    members_data = json.loads(phase4_members_path.read_text())
    protocol_path = phase4_members_path.parent/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    parents = {'parent_split_sha256':split_hash,'phase3_members_sha256':parent_member_hash,
               'phase3_protocol_sha256':parent_protocol_hash,
               'phase3_evaluation_addendum_sha256':addendum_hash,
               'phase3_candidates_sha256':candidates_hash,
               'phase3_model_input_manifest_sha256':inputs_hash}
    if any(members_data[k] != v or protocol[k] != v for k,v in parents.items()) or protocol['members_sha256'] != members_hash:
        raise ValueError('Frozen phase-four parent identity changed')
    if candidates_hash != (candidates_path.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate universe SHA changed')
    if frozen['members_sha256'] != parent_member_hash or inputs['candidate_sha256'] != candidates_hash:
        raise ValueError('Candidate/formula manifest linkage changed')
    parent_members = json.loads(phase3_members_path.read_text())['members']
    expected = [m for m in parent_members if inputs['queries'][m['qid']]['reason'] == 'formula_ambiguous']
    members = members_data['members']
    if len(members) != 69 or [m['qid'] for m in members] != [m['qid'] for m in expected] or any(
            m['canonical_target_key'] != e['canonical_target_key'] for m,e in zip(members,expected)):
        raise ValueError('Phase-four ambiguity membership changed')
    cache = json.loads(formula_cache_path.read_text())
    formula_by_key = cache['candidate_formula_by_key']
    valid_source = Path('references/mist_main_v2/src/mist/utils/chem_utils.py')
    if cache['official_valid_elements_source_sha256'] != sha256(valid_source):
        raise ValueError('Official element-list source changed')
    valid = None
    for node in ast.parse(valid_source.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALID_ELEMENTS' for t in node.targets):
            valid = set(ast.literal_eval(node.value))
            break
    if valid is None:
        raise ValueError('Official valid elements missing')
    candidates = {c['key']:c for c in frozen['candidates']}
    token = re.compile(r'(?:\[(\d+)([A-Z][a-z]*)\]|([A-Z][a-z]*))(\d*)')
    def parse_formula(formula):
        pos = 0
        counts = Counter()
        explicit_isotope = False
        while pos < len(formula):
            match = token.match(formula,pos)
            if not match:
                return None,True
            isotope,symbol,normal_symbol,count = match.groups()
            counts[symbol or normal_symbol] += int(count or 1)
            explicit_isotope |= isotope is not None
            pos = match.end()
        return counts,explicit_isotope
    results = {}
    all_formula_groups = 0
    all_candidate_occurrences = 0
    max_mass_span = 0.
    for member in members:
        qid = member['qid']
        q = inputs['queries'][qid]
        if q['reason'] != 'formula_ambiguous' or not q['formula_count'] > 1:
            raise ValueError(f'Unexpected query class: {qid}')
        by_formula = defaultdict(list)
        for key in q['candidate_keys']:
            if key not in candidates or key not in formula_by_key:
                raise ValueError(f'Unfrozen candidate formula: {qid}/{key}')
            by_formula[formula_by_key[key]['formula']].append(key)
        if sorted(by_formula) != q['formulas'] or sum(map(len,by_formula.values())) != q['candidate_count']:
            raise ValueError(f'Complete formula/pool count changed: {qid}')
        mass = q['neutral_mass_da']
        if mass is None:
            raise ValueError('Ambiguous formula query unexpectedly lacks supported mass')
        formulas = []
        for formula, keys in sorted(by_formula.items()):
            counts, parsed_isotope = parse_formula(formula)
            isotopes = sorted({iso for key in keys for iso in formula_by_key[key]['isotopes']})
            if counts is None:
                symbols = []
                bound = None
                unsupported = ['formula_syntax']
            else:
                symbols = sorted(counts)
                bound = math.prod(n+1 for n in counts.values())
                unsupported = sorted(set(symbols)-valid)
            if parsed_isotope and not isotopes:
                raise ValueError(f'Formula isotope/cache disagreement: {qid}/{formula}')
            masses = [float(candidates[key]['exact_mass']) for key in keys]
            min_mass,max_mass = min(masses),max(masses)
            max_mass_span = max(max_mass_span,max_mass-min_mass)
            neutral_mass = float(np.median(masses))
            formulas.append({'formula':formula,'candidate_count':len(keys),
                             'neutral_exact_mass_da':neutral_mass,
                             'neutral_exact_mass_span_da':max_mass-min_mass,
                             'mass_error_da':neutral_mass-mass,
                             'absolute_mass_error_da':abs(neutral_mass-mass),
                             'elements':symbols,'element_counts':dict(sorted(counts.items())) if counts is not None else None,
                             'isotopes':isotopes,
                             'cartesian_upper_bound':bound,
                             'is_chno':bool(counts) and set(counts)<=set('CHNO'),
                             'upstream_supported':counts is not None and not isotopes and not unsupported,
                             'unsupported_elements_or_syntax':unsupported})
        hplus = q['hplus_positive_spec_ids']
        pairing_count = len(hplus)*len(formulas)
        full_w = None if any(f['cartesian_upper_bound'] is None for f in formulas) else len(hplus)*sum(f['cartesian_upper_bound'] for f in formulas)
        all_supported = all(f['upstream_supported'] for f in formulas)
        all_within = all(f['cartesian_upper_bound'] is not None and f['cartesian_upper_bound']<=2_000_000 for f in formulas)
        results[qid] = {'neutral_mass_da':mass,'candidate_count':q['candidate_count'],
                        'formula_count':len(formulas),'formulas':formulas,
                        'hplus_positive_spec_ids':hplus,'hplus_positive_spectra':len(hplus),
                        'source_spectra':q['source_spectra'],
                        'all_formulas_upstream_supported':all_supported,
                        'all_formulas_within_2m_combinations':all_within,
                        'formula_spectrum_pairs':pairing_count,'cost_proxy_w':full_w,
                        'cost_pilot_eligible':bool(hplus) and all_supported and all_within and pairing_count<=200 and full_w is not None and full_w<=20_000_000}
        all_formula_groups += len(formulas)
        all_candidate_occurrences += q['candidate_count']
    summary = {'queries':len(results),'formula_groups':all_formula_groups,
               'candidate_occurrences_in_final_pools':all_candidate_occurrences,
               'hplus_queries':sum(bool(r['hplus_positive_spectra']) for r in results.values()),
               'all_formulas_upstream_supported_queries':sum(r['all_formulas_upstream_supported'] for r in results.values()),
               'cost_pilot_eligible_queries':sum(r['cost_pilot_eligible'] for r in results.values()),
               'isotope_formula_groups':sum(bool(f['isotopes']) for r in results.values() for f in r['formulas']),
               'chno_formula_groups':sum(f['is_chno'] for r in results.values() for f in r['formulas']),
               'max_same_formula_exact_mass_span_da':max_mass_span}
    output.mkdir(parents=True,exist_ok=False)
    b.write_json(output/'formula_metadata.json',{'summary':summary,'queries':results})
    record={'status':'truth_free_metadata_complete','source_sha256':sha256(Path(__file__)),
            'phase4_members_sha256':members_hash,'phase4_protocol_sha256':sha256(protocol_path),
            'phase3_formula_cache_sha256':sha256(formula_cache_path),
            'official_elements_source_sha256':sha256(valid_source),
            'candidate_sha256':candidates_hash,'input_manifest_sha256':inputs_hash,
            'metadata_sha256':sha256(output/'formula_metadata.json'),
            'elapsed_seconds':time.monotonic()-started,
            'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'cpu_threads':2,'virtual_memory_limit_gib':12,'timeout_seconds':900,
            'truth_formula_used':False,'sealed_scored':False}
    b.write_json(output/'run_record.json',record)
    print(json.dumps({'summary':summary,'record':record},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--members', type=Path)
    parser.add_argument('--phase4-members', type=Path)
    parser.add_argument('--input-manifest', type=Path)
    parser.add_argument('--formula-cache', type=Path)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase4_members is not None:
        audit_ambiguous_metadata(args.manifest, args.members, args.phase4_members,
                                 args.candidates, args.input_manifest, args.formula_cache, args.output)
    elif args.members is None:
        audit(args.data, args.manifest, args.candidates, args.output)
    else:
        audit_remaining(args.data, args.manifest, args.members, args.candidates, args.output)
