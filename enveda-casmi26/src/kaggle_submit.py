"""Test-only inputs for the fixed61 COCONUT/MIST submission path."""
import argparse
from collections import Counter
import hashlib
import json
import math
import time
from pathlib import Path

from phase5_predict import block


TEST_COLUMNS = ('molecule_id', 'spectrum_id', 'ms2_mzs',
                'ms2_normalized_intensities', 'adduct', 'ionization_mode',
                'instrument_type', 'precursor_mz')


def prepare_test(test, archive, output, limit=None, id_list=None):
    """Only test-visible fields enter the mass windows and model input."""
    import numpy as np
    import pyarrow.parquet as pq
    import baseline as b
    from coconut_candidates import build as build_coconut, SOURCE_SHA256, sha256
    from formula_audit import candidate_formula
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pf = pq.ParquetFile(test)
    if not set(TEST_COLUMNS) <= set(pf.schema_arrow.names):
        raise ValueError('Test spectrum schema lacks required columns')
    if id_list is not None:
        if limit is not None:
            raise ValueError('Do not combine --limit and --id-list')
        requested_ids = json.loads(Path(id_list).read_text())
        if not isinstance(requested_ids, list) or not requested_ids or len(requested_ids) != len(set(requested_ids)):
            raise ValueError('Invalid frozen test ID list')
        requested_set = set(requested_ids)
    else:
        requested_set = None
    ids = {}
    all_test_ids = set()
    groups = {}
    masses = {}
    audits = {}
    raw = []
    for batch in pf.iter_batches(batch_size=4096, columns=TEST_COLUMNS):
        for row in batch.to_pylist():
            mid = row['molecule_id']
            if mid is None:
                raise ValueError('Null molecule_id')
            all_test_ids.add(mid)
            if requested_set is not None and mid not in requested_set:
                continue
            if mid not in ids:
                if limit is not None and len(ids) >= limit:
                    continue
                qid = f'q{len(ids):03}'
                ids[mid] = qid
                groups[qid] = {'molecule_id': mid, 'source_spectra': 0,
                               'adduct_spectra': Counter(), 'polarity_spectra': Counter(),
                               'hplus_spectra': [], 'hplus_input_failure': None}
                masses[qid] = []
                audits[qid] = {'source_spectra': 0, 'legal_mass_spectra': 0,
                               'unsupported_adduct': 0, 'invalid_precursor_or_mass': 0}
            qid = ids.get(mid)
            if qid is None:
                continue
            g = groups[qid]
            g['source_spectra'] += 1
            g['adduct_spectra'][str(row['adduct'])] += 1
            g['polarity_spectra'][str(row['ionization_mode'])] += 1
            audits[qid]['source_spectra'] += 1
            try:
                precursor = float(row['precursor_mz'])
            except (TypeError, ValueError):
                precursor = float('nan')
            if row['adduct'] in b.ADDUCTS:
                if math.isfinite(precursor) and precursor > 0:
                    mass = b.neutral_mass(precursor, row['adduct'])
                    if math.isfinite(mass) and mass > 0:
                        masses[qid].append(float(mass))
                        audits[qid]['legal_mass_spectra'] += 1
                    else:
                        audits[qid]['invalid_precursor_or_mass'] += 1
                else:
                    audits[qid]['invalid_precursor_or_mass'] += 1
            else:
                audits[qid]['unsupported_adduct'] += 1
            if row['adduct'] != '[M+H]+' or row['ionization_mode'] != 'positive':
                continue
            spec = f"{qid}_s{len(g['hplus_spectra']):03}"
            try:
                mzs, intensities = row['ms2_mzs'], row['ms2_normalized_intensities']
                if mzs is None or intensities is None or len(mzs) != len(intensities):
                    raise ValueError('unaligned_raw_peaks')
                peaks = [(float(m), float(v)) for m, v in zip(mzs, intensities)]
                if not math.isfinite(precursor) or precursor <= 0 or not peaks or any(
                    not math.isfinite(m) or not math.isfinite(v) or m <= 0 or v < 0
                    for m, v in peaks
                ):
                    raise ValueError('invalid_raw_peaks')
                text = block(spec, precursor, peaks)
                g['hplus_spectra'].append({
                    'spec': spec, 'precursor_mz': precursor,
                    'instrument': row['instrument_type'] or 'unknown',
                    'input_peaks': len(peaks),
                    'raw_block_sha256': hashlib.sha256(text.encode()).hexdigest()})
                raw.append(text)
            except (TypeError, ValueError):
                g['hplus_spectra'].append({'spec': spec, 'invalid': True})
                g['hplus_input_failure'] = 'invalid_precursor_or_raw_peaks'
    if not groups:
        raise ValueError('No test molecule IDs')
    if requested_set is not None and set(ids) != requested_set:
        raise ValueError('Frozen test ID list incomplete')
    if limit is None and requested_set is None and set(ids) != all_test_ids:
        raise ValueError('Full test molecule-ID set incomplete')
    for qid, g in groups.items():
        g['neutral_mass_da'] = float(np.median(masses[qid])) if masses[qid] else None
        g['legal_mass_spectra'] = len(masses[qid])
        g['hplus_positive_spec_ids'] = [x['spec'] for x in g['hplus_spectra']]
        g['adduct_spectra'] = dict(sorted(g['adduct_spectra'].items()))
        g['polarity_spectra'] = dict(sorted(g['polarity_spectra'].items()))
    b.write_json(output/'queries.json', groups)
    b.write_json(output/'query_spectrum_masses.json', masses)
    (output/'hplus.mgf').write_text(''.join(raw), encoding='utf-8')
    build_coconut(None, archive, None, output/'candidates',
                  test_mass_groups=masses, test_query_audit=audits)
    frozen = json.loads((output/'candidates'/'candidates.json').read_text())
    candidates = frozen['candidates']
    pools = {}
    union = {}
    for qid, g in groups.items():
        mass = g['neutral_mass_da']
        tol = None if mass is None else max(b.CONFIG['mass_floor_da'],
                                             mass*b.CONFIG['mass_ppm']*1e-6)
        pool = [c for c in candidates if tol is not None and
                abs(c['exact_mass']-mass) <= tol]
        pools[qid] = pool
        union.update((c['key'], c) for c in pool)
    formulas = {}
    for key, candidate in sorted(union.items()):
        formula, error, isotopes = candidate_formula(candidate)
        if abs(error) > 1e-6:
            raise ValueError(f'Candidate formula mass drift: {key}')
        formulas[key] = {'formula': formula, 'isotopes': isotopes}
    b.write_json(output/'pools.json', pools)
    b.write_json(output/'candidate_formula_by_key.json', formulas)
    record = {'status': 'test_only_candidates_prepared', 'test_sha256': sha256(test),
              'archive_sha256': SOURCE_SHA256,
              'source_sha256': sha256(Path(__file__)),
              'queries_sha256': sha256(output/'queries.json'),
              'hplus_mgf_sha256': sha256(output/'hplus.mgf'),
              'candidate_sha256': sha256(output/'candidates'/'candidates.json'),
              'pools_sha256': sha256(output/'pools.json'),
              'formulas_sha256': sha256(output/'candidate_formula_by_key.json'),
              'query_count': len(groups), 'source_test_molecule_count': len(all_test_ids),
              'smoke_limit': limit, 'selected_id_list_sha256': sha256(Path(id_list)) if id_list is not None else None,
               'legal_spectra': sum(map(len, masses.values())),
              'hplus_spectra': sum(len(g['hplus_spectra']) for g in groups.values()),
              'candidate_union': len(candidates), 'final_pool_union': len(union),
              'empty_final_pool_count': sum(not x for x in pools.values()),
              'truth_fields_used': False}
    b.write_json(output/'run_record.json', record)
    print(json.dumps(record), flush=True)


def model_stage(prepared, checkpoint, output_dir=None, strategy='fixed61'):
    """Official MIST subformula assignment, frozen gate, then fixed61 inference."""
    stage_started = time.monotonic()
    import ast
    import csv
    import re
    from collections import defaultdict
    import numpy as np
    from mist.data.featurizers import FingerprintFeaturizer
    from mist import utils
    from mist.utils.chem_utils import VALID_ELEMENTS
    from phase4_explain import explanation_from_output
    from mist_phase2 import infer_fixed61
    from types import SimpleNamespace
    def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if strategy not in {'fixed61', 'mass2'}:
        raise ValueError('Unknown test prediction strategy')

    if sha256(checkpoint) != 'a5a37e2b269f9a5ec483bd7fd2d3402257ce5f223270d534610fe41c7c4c5dea':
        raise ValueError('Official checkpoint identity mismatch')
    out = output_dir or prepared/'model'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    queries = json.loads((prepared/'queries.json').read_text())
    pools = json.loads((prepared/'pools.json').read_text())
    formulas = json.loads((prepared/'candidate_formula_by_key.json').read_text())
    if set(queries) != set(pools):
        raise ValueError('Query/pool ID mismatch')
    parsed = utils.parse_spectra_mgf(prepared/'hplus.mgf')
    raw = {meta['FEATURE_ID']: (meta, tuples) for meta, tuples in parsed}
    if len(raw) != len(parsed):
        raise ValueError('Duplicate H+ spectrum name')
    expected = {x['spec'] for q in queries.values() for x in q['hplus_spectra']
                if not x.get('invalid')}
    if set(raw) != expected:
        raise ValueError('H+ MGF/input set mismatch')
    token = re.compile(r'(?:\[(\d+)([A-Z][a-z]*)\]|([A-Z][a-z]*))(\d*)')
    def support(formula):
        pos = 0
        counts = Counter()
        isotope = False
        while pos < len(formula):
            match = token.match(formula, pos)
            if not match:
                return False, 'formula_syntax', None
            isotope_token, symbol, plain, number = match.groups()
            counts[symbol or plain] += int(number or 1)
            isotope |= isotope_token is not None
            pos = match.end()
        bound = math.prod(v+1 for v in counts.values())
        if isotope:
            return False, 'formula_isotope', bound
        if set(counts)-set(VALID_ELEMENTS):
            return False, 'unsupported_element', bound
        if bound > 2_000_000:
            return False, 'formula_budget', bound
        return True, None, bound
    policy = {}
    model_specs = []
    model_outputs = {}
    mass2_specs = []
    mass2_outputs = {}
    audit = {}
    selected_candidates = {}
    for qid, query in queries.items():
        query_started = time.monotonic()
        bounds = {}
        pool = pools[qid]
        groups = defaultdict(list)
        for candidate in pool:
            groups[formulas[candidate['key']]['formula']].append(candidate)
        ids = query['hplus_positive_spec_ids']
        status = 'mass_only_fallback'
        reason = None
        chosen = None
        selected = []
        mass2_formulas = []
        if query['neutral_mass_da'] is None:
            reason = 'no_supported_mass'
        elif not pool:
            reason = 'empty_pool'
        elif len(pool) == 1:
            reason = 'single_candidate'
        elif not ids:
            reason = 'no_supported_hplus'
        elif query['hplus_input_failure']:
            reason = query['hplus_input_failure']
        else:
            bad = []
            bounds = {}
            for form, candidates in groups.items():
                good, why, bound = support(form)
                bounds[form] = bound
                if any(formulas[c['key']]['isotopes'] for c in candidates):
                    good, why = False, 'candidate_isotope'
                if not good:
                    bad.append({'formula': form, 'reason': why, 'cartesian_upper_bound': bound})
            if bad:
                reason = 'incomplete_multiformula_support_or_budget'
                audit[qid] = {'unsupported_formulas': bad}
            elif len(groups) == 1:
                chosen = next(iter(groups))
                selected = [c['key'] for c in pool]
                status = 'single_formula_model'
            elif len(groups)*len(ids) > 200 or len(ids)*sum(bounds.values()) > 20_000_000:
                reason = 'multiformula_pair_or_cost_budget'
                audit[qid] = {'formula_spectrum_pairs': len(groups)*len(ids),
                              'cost_proxy_w': len(ids)*sum(bounds.values())}
            else:
                hypotheses = []
                broken = False
                for form, candidates in sorted(groups.items()):
                    readings = []
                    for spec in ids:
                        processed = utils.process_spec_file(*raw[spec]) if spec in raw else None
                        if processed is None or not len(processed):
                            broken = True
                            break
                        output = utils.get_output_dict(spec_name=spec, spec=processed.copy(),
                                form=form, mass_diff_type='ppm', mass_diff_thresh=20,
                                ion_type='[M+H]+')
                        measure = explanation_from_output(
                            processed.copy(),
                            next(s['precursor_mz'] for s in query['hplus_spectra'] if s['spec'] == spec),
                            output)
                        if measure['e_i'] is None or not math.isfinite(measure['e_i']):
                            broken = True
                            break
                        readings.append(measure['e_i'])
                    if broken:
                        break
                    hypotheses.append({'formula': form,
                        'mean_e_i': sum(readings)/len(readings),
                        'absolute_mass_error_da': abs(candidates[0]['exact_mass']-query['neutral_mass_da']),
                        'candidate_count': len(candidates)})
                if broken or len(hypotheses) != len(groups):
                    reason = 'audit_failure_or_na'
                else:
                    order = sorted(hypotheses, key=lambda h: (
                        -h['mean_e_i'], h['absolute_mass_error_da'], h['formula']))
                    audit[qid] = {'ranked_formulas': order}
                    if order[0]['mean_e_i'] < .50:
                        reason = 'low_explained_intensity'
                    elif order[0]['mean_e_i']-order[1]['mean_e_i'] < .05:
                        reason = 'formula_ambiguous_explanation'
                        if strategy == 'mass2':
                            for form, candidates in groups.items():
                                if max(c['exact_mass'] for c in candidates)-min(c['exact_mass'] for c in candidates) > 1e-9:
                                    raise ValueError(f'Inconsistent exact mass within formula: {qid} {form}')
                            mass2_formulas = sorted(groups, key=lambda form: (
                                abs(groups[form][0]['exact_mass']-query['neutral_mass_da']), form))[:2]
                            selected = [c['key'] for form in mass2_formulas for c in groups[form]]
                            status, reason = 'mass2_model', None
                    else:
                        chosen = order[0]['formula']
                        selected = [c['key'] for c in groups[chosen]]
                        status = 'multi_gate_single_candidate' if len(selected) == 1 else 'multi_gate_model'
        if status in {'single_formula_model', 'multi_gate_model'}:
            assignments = {}
            try:
                for spec in ids:
                    if spec not in raw:
                        raise ValueError('missing_raw_spectrum')
                    processed = utils.process_spec_file(*raw[spec])
                    if processed is None or not len(processed):
                        raise ValueError('empty_processed_spectrum')
                    result = utils.get_output_dict(spec_name=spec, spec=processed.copy(),
                                form=chosen, mass_diff_type='ppm', mass_diff_thresh=20,
                                ion_type='[M+H]+')
                    table = result['output_tbl']
                    if table is None or not len(table['formula']):
                        raise ValueError('empty_subformula_assignment')
                    assignments[spec] = result
            except (ValueError, KeyError) as exc:
                status, reason, chosen, selected = 'mass_only_fallback', f'model_input_failure:{exc}', None, []
            else:
                model_outputs.update(assignments)
                for spec in ids:
                    row = next(s for s in query['hplus_spectra'] if s['spec'] == spec)
                    model_specs.append({'qid': qid, 'spec': spec, 'formula': chosen,
                                        'precursor_mz': row['precursor_mz'],
                                        'instrument': row['instrument']})
                for candidate in pool:
                    if candidate['key'] in selected:
                        prior = selected_candidates.get(candidate['key'])
                        if prior is not None and prior['smiles'] != candidate['smiles']:
                            raise ValueError('Conflicting original candidate representative')
                        selected_candidates[candidate['key']] = candidate
        if status == 'mass2_model':
            assignments = {}
            try:
                for fi, form in enumerate(mass2_formulas):
                    for spec in ids:
                        if spec not in raw:
                            raise ValueError('missing_raw_spectrum')
                        processed = utils.process_spec_file(*raw[spec])
                        if processed is None or not len(processed):
                            raise ValueError('empty_processed_spectrum')
                        name = f'{spec}__mass2_f{fi}'
                        if name in raw or name in model_outputs or name in mass2_outputs:
                            raise ValueError('mass2_spectrum_name_collision')
                        result = utils.get_output_dict(spec_name=name, spec=processed.copy(),
                                    form=form, mass_diff_type='ppm', mass_diff_thresh=20,
                                    ion_type='[M+H]+')
                        table = result['output_tbl']
                        if table is None or not len(table['formula']):
                            raise ValueError('empty_subformula_assignment')
                        assignments[name] = result
            except (ValueError, KeyError) as exc:
                status, reason, selected, mass2_formulas = (
                    'mass_only_fallback', f'mass2_input_failure:{exc}', [], [])
            else:
                mass2_outputs.update(assignments)
                for fi, form in enumerate(mass2_formulas):
                    for spec in ids:
                        row = next(s for s in query['hplus_spectra'] if s['spec'] == spec)
                        mass2_specs.append({'qid': qid, 'spec': f'{spec}__mass2_f{fi}',
                                            'raw_spec': spec, 'formula': form,
                                            'precursor_mz': row['precursor_mz'],
                                            'instrument': row['instrument']})
                for candidate in pool:
                    if candidate['key'] in selected:
                        prior = selected_candidates.get(candidate['key'])
                        if prior is not None and prior['smiles'] != candidate['smiles']:
                            raise ValueError('Conflicting original candidate representative')
                        selected_candidates[candidate['key']] = candidate
        policy[qid] = {'status': status, 'fallback_reason': reason, 'formula': chosen,
                       'selected_candidate_keys': selected,
                       'hplus_positive_spec_ids': ids,
                       'complete_candidate_count': len(pool),
                       'formula_count': len(groups)}
        if strategy == 'mass2' and status == 'mass2_model':
            policy[qid]['mass2_formulas'] = mass2_formulas
        print(json.dumps({'stage':'model_query_complete', 'qid':qid, 'status':status,
            'candidate_count':len(pool), 'formula_count':len(groups), 'hplus_spectra':len(ids),
            'formula_spectrum_pairs':len(groups)*len(ids),
            'cost_proxy_w':len(ids)*sum(bounds.values()) if len(bounds)==len(groups) and all(v is not None for v in bounds.values()) else None,
            'elapsed_seconds':time.monotonic()-query_started}), flush=True)
    subforms = out/'subforms'
    subforms.mkdir()
    for spec, result in model_outputs.items():
        (subforms/f'{spec}.json').write_text(json.dumps(result, indent=4))
    with (out/'model_labels.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['spec','formula','ionization','dataset',
                    'compound','parentmass','instrument'], delimiter='\t')
        writer.writeheader()
        for row in model_specs:
            writer.writerow({'spec': row['spec'], 'formula': row['formula'],
                'ionization': '[M+H]+', 'dataset': 'kaggle_test',
                'compound': row['qid'], 'parentmass': row['precursor_mz'],
                'instrument': row['instrument']})
    by_block = {}
    for chunk in (prepared/'hplus.mgf').read_text().split('BEGIN IONS\n')[1:]:
        content = 'BEGIN IONS\n'+chunk.split('END IONS\n')[0]+'END IONS\n\n'
        names = [line.split('=',1)[1] for line in content.splitlines()
                 if line.startswith('FEATURE_ID=')]
        if len(names) != 1 or names[0] in by_block:
            raise ValueError('Invalid raw MGF block')
        by_block[names[0]] = content
    (out/'model.mgf').write_text(''.join(by_block[x['spec']] for x in model_specs))
    if strategy == 'mass2':
        mass2_subforms = out/'mass2_subforms'
        mass2_subforms.mkdir()
        for name, result in mass2_outputs.items():
            (mass2_subforms/f'{name}.json').write_text(json.dumps(result, indent=4))
        with (out/'mass2_labels.tsv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['spec','formula','ionization','dataset',
                        'compound','parentmass','instrument'], delimiter='	')
            writer.writeheader()
            for row in mass2_specs:
                writer.writerow({'spec': row['spec'], 'formula': row['formula'],
                    'ionization': '[M+H]+', 'dataset': 'kaggle_test',
                    'compound': row['qid'], 'parentmass': row['precursor_mz'],
                    'instrument': row['instrument']})
        (out/'mass2.mgf').write_text(''.join(
            by_block[x['raw_spec']].replace(
                f"FEATURE_ID={x['raw_spec']}\n", f"FEATURE_ID={x['spec']}\n", 1)
            for x in mass2_specs))
        (out/'mass2_spectra_map.json').write_text(json.dumps(mass2_specs, indent=2))
    fingerprint = {}
    featurizer = FingerprintFeaturizer(fp_names=['morgan4096'])
    for key, candidate in sorted(selected_candidates.items()):
        bits = np.asarray(featurizer.featurize_smiles(candidate['smiles']))
        if bits.shape != (4096,) or not np.isin(bits, [0, 1]).all():
            raise ValueError('Invalid official candidate fingerprint')
        fingerprint[key] = np.packbits(bits.astype(np.uint8), bitorder='big').tobytes().hex()
    (out/'policy.json').write_text(json.dumps(policy, indent=2))
    (out/'explanation_audit.json').write_text(json.dumps(audit, indent=2))
    (out/'selected_fingerprints_model.json').write_text(json.dumps(fingerprint, indent=2))
    (out/'model_spectra_map.json').write_text(json.dumps(model_specs, indent=2))
    if model_specs:
        save_dir = out/'preds'
        save_dir.mkdir()
        infer_fixed61(SimpleNamespace(model=checkpoint, labels=out/'model_labels.tsv',
                      subforms=subforms, name='kaggle_test', save_dir=save_dir))
        pred = save_dir/'fp_preds_kaggle_test.p'
        if not pred.exists():
            raise FileNotFoundError(pred)
    if mass2_specs:
        import os, resource, signal, subprocess, sys
        save_dir = out/'mass2_preds'
        save_dir.mkdir()
        remaining = (float(os.environ['CASMI_MODEL_DEADLINE_MONOTONIC']) - time.monotonic() - 2
                     if 'CASMI_MODEL_DEADLINE_MONOTONIC' in os.environ
                     else 900 - (time.monotonic()-stage_started) - 2)
        if remaining <= 0:
            raise TimeoutError('No time remains in the bounded model stage')
        command = [sys.executable, str(Path(__file__).with_name('mist_phase2.py')),
                   'infer-fixed61', '--model', str(checkpoint),
                   '--labels', str(out/'mass2_labels.tsv'),
                   '--subforms', str(out/'mass2_subforms'),
                   '--name', 'kaggle_test_mass2', '--save-dir', str(save_dir)]
        child = subprocess.Popen(command, start_new_session=True)
        try:
            returncode = child.wait(timeout=remaining)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        pred = save_dir/'fp_preds_kaggle_test_mass2.p'
        if not pred.exists():
            raise FileNotFoundError(pred)
    (out/'model_run_record.json').write_text(json.dumps({
        'status':'test_model_stage_complete', 'strategy':strategy,
        'source_sha256':sha256(Path(__file__)),
        'checkpoint_sha256':sha256(checkpoint),
        'queries_sha256':sha256(prepared/'queries.json'),
        'pools_sha256':sha256(prepared/'pools.json'),
        'formula_sha256':sha256(prepared/'candidate_formula_by_key.json'),
        'hplus_mgf_sha256':sha256(prepared/'hplus.mgf'),
        'policy_sha256':sha256(out/'policy.json'),
        'old_model_prediction_sha256':sha256(out/'preds'/'fp_preds_kaggle_test.p') if model_specs else None,
        'mass2_model_prediction_sha256':sha256(out/'mass2_preds'/'fp_preds_kaggle_test_mass2.p') if mass2_specs else None,
        'old_model_spectra':len(model_specs), 'mass2_model_spectra':len(mass2_specs),
        'mass2_inference_process':'separate_existing_infer_fixed61_cli' if mass2_specs else None,
        'stage_elapsed_seconds':time.monotonic()-stage_started,
        'max_parent_plus_child_rss_kib_upper_bound':(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            if mass2_specs else None
    }, indent=2)+'\n')
    print(json.dumps({'status':'test_model_stage_complete',
        'strategy':strategy, 'mass2_model_queries':len({x['qid'] for x in mass2_specs}),
        'mass2_model_spectra':len(mass2_specs),
        'queries':len(queries), 'selected_model_queries':len({x['qid'] for x in model_specs}),
        'model_spectra':len(model_specs), 'selected_candidate_union':len(fingerprint),
        'status_counts':dict(Counter(v['status'] for v in policy.values()))}), flush=True)

def finalize_test(prepared, model_dir, output_dir=None, strategy='fixed61'):
    """Rank every test molecule once, then emit the exact Kaggle CSV schema."""
    import csv
    import pickle
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import baseline as b
    from phase5_predict import predict_query
    from coconut_candidates import sha256
    if strategy not in {'fixed61', 'mass2'}:
        raise ValueError('Unknown test prediction strategy')

    queries = json.loads((prepared/'queries.json').read_text())
    pools = json.loads((prepared/'pools.json').read_text())
    formulas = json.loads((prepared/'candidate_formula_by_key.json').read_text())
    universe = json.loads((prepared/'candidates'/'candidates.json').read_text())['candidates']
    policy = json.loads((model_dir/'policy.json').read_text())
    explanation_audit = json.loads((model_dir/'explanation_audit.json').read_text()) if strategy == 'mass2' else {}
    fingerprints = json.loads((model_dir/'selected_fingerprints_model.json').read_text())
    spectra = json.loads((model_dir/'model_spectra_map.json').read_text())
    if set(queries) != set(pools) or set(queries) != set(policy):
        raise ValueError('Test prediction ID denominator changed')
    model_record = model_dir/'model_run_record.json'
    if model_record.exists():
        record = json.loads(model_record.read_text())
        if (record['strategy'] != strategy or
                record['queries_sha256'] != sha256(prepared/'queries.json') or
                record['pools_sha256'] != sha256(prepared/'pools.json') or
                record['policy_sha256'] != sha256(model_dir/'policy.json')):
            raise ValueError('Model/finalize strategy or input identity mismatch')
    elif strategy != 'fixed61':
        raise ValueError('Missing mass2 model strategy record')
    if strategy == 'fixed61' and any(v['status'] == 'mass2_model' for v in policy.values()):
        raise ValueError('Mass2 policy requires explicit mass2 strategy')
    prepared_record = json.loads((prepared/'run_record.json').read_text())
    if 'smoke_limit' not in prepared_record or 'source_test_molecule_count' not in prepared_record:
        raise ValueError('Test input-set identity not frozen')
    if prepared_record['query_count'] != len(queries) or (prepared_record['smoke_limit'] is None and prepared_record.get('selected_id_list_sha256') is None and prepared_record['source_test_molecule_count'] != len(queries)):
        raise ValueError('Submission omits a source test molecule ID')
    if len({g['molecule_id'] for g in queries.values()}) != len(queries):
        raise ValueError('Duplicate test molecule_id')
    model_pred = model_dir/'preds'/'fp_preds_kaggle_test.p'
    if spectra and not model_pred.exists():
        raise FileNotFoundError(model_pred)
    by_spec = {}
    if spectra:
        with model_pred.open('rb') as f:
            raw = pickle.load(f)
        names = list(raw['names'])
        vectors = np.asarray(raw['preds'])
        if len(names) != len(set(names)) or set(names) != {x['spec'] for x in spectra}:
            raise ValueError('Incomplete MIST spectrum predictions')
        if vectors.shape != (len(names), 4096) or not np.isfinite(vectors).all():
            raise ValueError('Invalid MIST fingerprint output')
        by_spec = dict(zip(names, vectors))
    mass2_spectra = []
    mass2_map = {}
    mass2_by_spec = {}
    mass2_pred = model_dir/'mass2_preds'/'fp_preds_kaggle_test_mass2.p'
    if strategy == 'mass2':
        mass2_spectra = json.loads((model_dir/'mass2_spectra_map.json').read_text())
        mass2_map = {x['spec']:x for x in mass2_spectra}
        if len(mass2_map) != len(mass2_spectra):
            raise ValueError('Duplicate mass2 input spectrum name')
        if mass2_spectra:
            if not mass2_pred.exists():
                raise FileNotFoundError(mass2_pred)
            with mass2_pred.open('rb') as f:
                raw = pickle.load(f)
            names = list(raw['names'])
            vectors = np.asarray(raw['preds'])
            expected_names = {x['spec'] for x in mass2_spectra}
            if len(names) != len(set(names)) or set(names)-expected_names or not names:
                raise ValueError('Duplicate, unexpected, or empty mass2 MIST spectrum predictions')
            if vectors.shape != (len(names), 4096):
                raise ValueError('Invalid mass2 MIST fingerprint shape')
            mass2_by_spec = dict(zip(names, vectors))
        if {q for q,v in policy.items() if v['status'] == 'mass2_model'} != {x['qid'] for x in mass2_spectra}:
            raise ValueError('Mass2 policy/spectrum mapping mismatch')
    mismatched = set()
    for key, packed in fingerprints.items():
        candidate = next((c for pool in pools.values() for c in pool if c['key'] == key), None)
        if candidate is None:
            raise ValueError('Fingerprint key absent from all final pools')
        mol = Chem.MolFromSmiles(candidate['smiles'])
        if mol is None:
            raise ValueError('Invalid candidate in cross-RDKit check')
        vector = np.zeros((4096,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096), vector)
        actual = np.packbits(vector.astype(np.uint8), bitorder='big').tobytes().hex()
        if actual != packed:
            mismatched.add(key)
    final = output_dir or prepared/'final'
    if final.exists():
        raise FileExistsError(final)
    final.mkdir()
    records = {}
    counts = Counter()
    for qid, query in queries.items():
        chosen = dict(policy[qid])
        if set(chosen['selected_candidate_keys']) & mismatched:
            chosen.update(status='mass_only_fallback', fallback_reason='cross_rdkit_fingerprint_mismatch',
                          formula=None, selected_candidate_keys=[])
        if chosen['status'] == 'mass2_model':
            from mist_phase2 import rank_cosine
            from phase5_mass2 import alternating
            base = dict(chosen, status='mass_only_fallback', formula=None,
                        selected_candidate_keys=[], fallback_reason='formula_ambiguous_explanation')
            result = predict_query(qid, query, pools[qid], formulas, base,
                                   by_spec, fingerprints)
            mass_keys = result['diagnostic']['full_mass_key_order']
            if result['diagnostic']['full_fm_key_order'] != mass_keys:
                raise ValueError('Original EI-ambiguous FM tail is not mass order: '+qid)
            ranked = explanation_audit[qid]['ranked_formulas']
            if len(ranked) < 2 or ranked[0]['mean_e_i'] < .50 or ranked[0]['mean_e_i']-ranked[1]['mean_e_i'] >= .05:
                raise ValueError('Mass2 status does not match the frozen EI gate: '+qid)
            by_key = {c['key']:c for c in pools[qid]}
            selected_set = {k for k in mass_keys if formulas[k]['formula'] in chosen['mass2_formulas']}
            if selected_set != set(chosen['selected_candidate_keys']) or len(selected_set) != len(chosen['selected_candidate_keys']):
                raise ValueError('Incomplete mass2 formula candidate group: '+qid)
            ids = query['hplus_positive_spec_ids']
            groups = []
            failure = None
            if len(chosen['mass2_formulas']) != 2 or ids != chosen['hplus_positive_spec_ids'] or not ids:
                failure = 'incomplete_mass2_formula_or_hplus_input'
            elif chosen['mass2_formulas'] != sorted(
                    chosen['mass2_formulas'], key=lambda form: (
                        abs(next(c['exact_mass'] for c in pools[qid] if formulas[c['key']]['formula']==form)
                            -query['neutral_mass_da']), form)):
                raise ValueError('Mass2 formula order changed: '+qid)
            else:
                for fi, form in enumerate(chosen['mass2_formulas']):
                    group = [k for k in mass_keys if formulas[k]['formula'] == form]
                    names = [f'{spec}__mass2_f{fi}' for spec in ids]
                    if (not group or any(k not in fingerprints for k in group)
                            or any(s not in mass2_by_spec for s in names)
                            or any(mass2_map.get(s, {}).get('formula') != form
                                   or mass2_map.get(s, {}).get('raw_spec') != raw_spec
                                   or mass2_map.get(s, {}).get('qid') != qid
                                   for s, raw_spec in zip(names, ids))):
                        failure = 'incomplete_mass2_model_input_or_fingerprint'
                        break
                    cv = [np.unpackbits(np.frombuffer(bytes.fromhex(fingerprints[k]),
                                      dtype=np.uint8), bitorder='big') for k in group]
                    try:
                        groups.append([k for k,_ in rank_cosine(
                            [mass2_by_spec[s] for s in names], cv, group)])
                    except ValueError as exc:
                        failure = f'invalid_mass2_vector:{exc}'
                        break
            if failure is None:
                order = alternating(groups[0], groups[1], mass_keys)
                smiles = [by_key[k]['smiles'] for k in order[:25]]
            else:
                smiles = result['fm']
                chosen.update(status='mass_only_fallback', fallback_reason=failure)
        else:
            result = predict_query(qid, query, pools[qid], formulas, chosen,
                                   by_spec, fingerprints)
            smiles = result['fm']
        output_reason = chosen['fallback_reason']
        if not smiles:
            mass = query['neutral_mass_da']
            if mass is not None and universe:
                near = sorted(universe, key=lambda c: (abs(c['exact_mass']-mass), c['key']))
                smiles = [c['smiles'] for c in near[:25]]
                output_reason = 'outside_final_mass_fallback'
            else:
                smiles = ['CCO']
                output_reason = 'format_only_no_mass_or_wide_candidate'
        if not 1 <= len(smiles) <= 25:
            raise ValueError(f'Invalid prediction length: {qid}')
        keys = [b.canonical_key(s) for s in smiles]
        if None in keys or len(keys) != len(set(keys)):
            raise ValueError(f'Invalid or duplicate canonical prediction: {qid}')
        counts[chosen['status']] += 1
        if output_reason:
            counts[output_reason] += 1
        records[qid] = {'molecule_id':query['molecule_id'], 'smiles':smiles,
                        'policy_status':chosen['status'], 'reason':output_reason,
                        'final_pool_count':len(pools[qid])}
    with (final/'submission.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['molecule_id','smiles'])
        for row in records.values():
            writer.writerow([row['molecule_id'], ';'.join(row['smiles'])])
    (final/'predictions.json').write_text(json.dumps(records, indent=2)+'\n')
    (final/'run_record.json').write_text(json.dumps({
        'status':'test_only_csv_complete', 'strategy':strategy, 'query_count':len(queries),
        'test_sha256':json.loads((prepared/'run_record.json').read_text())['test_sha256'],
        'source_sha256':sha256(Path(__file__)),
        'candidate_sha256':sha256(prepared/'candidates'/'candidates.json'),
        'policy_sha256':sha256(model_dir/'policy.json'),
        'model_prediction_sha256':sha256(model_pred) if spectra else None,
        'mass2_model_prediction_sha256':sha256(mass2_pred) if mass2_spectra else None,
        'cross_rdkit_candidate_count':len(fingerprints),
        'cross_rdkit_mismatch_count':len(mismatched),
        'predictions_sha256':sha256(final/'predictions.json'),
        'submission_sha256':sha256(final/'submission.csv'),
        'status_counts':dict(counts), 'truth_fields_used':False}, indent=2)+'\n')
    print(json.dumps({'status':'test_only_csv_complete','rows':len(records),
                      'cross_rdkit_mismatches':len(mismatched),
                      'counts':dict(counts)}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='stage', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--test', type=Path, required=True)
    a.add_argument('--archive', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--limit', type=int)
    a.add_argument('--id-list', type=Path)
    a = sub.add_parser('model')
    a.add_argument('--prepared', type=Path, required=True)
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--model-dir', type=Path)
    a.add_argument('--strategy', choices=['fixed61','mass2'], default='fixed61')
    a = sub.add_parser('finalize')
    a.add_argument('--prepared', type=Path, required=True)
    a.add_argument('--model-dir', type=Path, required=True)
    a.add_argument('--final-dir', type=Path)
    a.add_argument('--strategy', choices=['fixed61','mass2'], default='fixed61')
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare_test(a.test, a.archive, a.output, a.limit, a.id_list)
    elif a.stage == 'model':
        model_stage(a.prepared, a.checkpoint, a.model_dir, a.strategy)
    elif a.stage == 'finalize':
        finalize_test(a.prepared, a.model_dir, a.final_dir, a.strategy)