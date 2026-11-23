"""Truth-free, bounded official-MIST subformula explanation audit for frozen phase-four pilots."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs(protocol_path, metadata_path):
    protocol = json.loads(protocol_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if protocol['status'] != 'frozen_before_subformula_explanation' or protocol['formula_metadata_sha256'] != sha(metadata_path):
        raise ValueError('Pilot protocol/metadata identity changed')
    for selection in protocol['selected']:
        row = metadata['queries'][selection['qid']]
        if (row['cost_proxy_w'] != selection['cost_proxy_w'] or
                row['formula_spectrum_pairs'] != selection['formula_spectrum_pairs'] or
                row['hplus_positive_spec_ids'] != selection['hplus_positive_spec_ids'] or
                [f['formula'] for f in row['formulas']] != selection['formulas'] or
                [f['cartesian_upper_bound'] for f in row['formulas']] != selection['formula_cartesian_upper_bounds']):
            raise ValueError('Frozen complete-formula pilot selection changed')
    return protocol, metadata


def prepare(args):
    """Use RDKit2026 only to associate original H+ spectra with frozen qids."""
    import pyarrow.parquet as pq
    import baseline as b
    protocol, metadata = frozen_inputs(args.protocol,args.metadata)
    phase4_members = json.loads(args.members.read_text())
    if protocol['phase4_members_sha256'] != sha(args.members):
        raise ValueError('Ambiguity members changed')
    selected = {row['qid']:row for row in protocol['selected']}
    key_to_qid = {m['canonical_target_key']:m['qid'] for m in phase4_members['members'] if m['qid'] in selected}
    if set(key_to_qid.values()) != set(selected):
        raise ValueError('Pilot qids missing from frozen members')
    rows = defaultdict(list)
    columns = ['ingest_lib','normalized_smiles','ms2_mzs','ms2_normalized_intensities',
               'adduct','ionization_mode','precursor_mz','instrument_type']
    for batch in pq.ParquetFile(args.data/'train.parquet').iter_batches(batch_size=4096,columns=columns):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i,source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            qid = key_to_qid.get(b.canonical_key(row['normalized_smiles']))
            if qid is None or row['adduct'] != '[M+H]+' or row['ionization_mode'] != 'positive':
                continue
            precursor = float(row['precursor_mz'])
            peaks = [(float(m),float(v)) for m,v in zip(row['ms2_mzs'],row['ms2_normalized_intensities'])]
            if not math.isfinite(precursor) or precursor <= 0 or not peaks or any(
                    not math.isfinite(m) or not math.isfinite(v) or m <= 0 or v < 0 for m,v in peaks):
                raise ValueError(f'Invalid raw H+ spectrum: {qid}; do not silently drop a spectrum')
            rows[qid].append({'precursor_mz':precursor,'peaks':peaks,
                              'instrument_type':row['instrument_type'] or 'unknown'})
    if set(rows) != set(selected):
        raise ValueError('Selected pilot spectrum/qid set changed')
    mapping = []
    for qid in selected:
        expected = metadata['queries'][qid]['hplus_positive_spec_ids']
        actual = [f'{qid}_s{i:03}' for i in range(len(rows[qid]))]
        if actual != expected:
            raise ValueError(f'Frozen H+ spectrum order/count changed for {qid}')
        for spec,row in zip(actual,rows[qid]):
            mapping.append({'spec':spec,'qid':qid,'precursor_mz':row['precursor_mz'],
                            'input_peaks':len(row['peaks']),'instrument_type':row['instrument_type']})
    args.output.mkdir(parents=True,exist_ok=False)
    mgf = args.output/'pilot.mgf'
    with mgf.open('w',encoding='utf-8') as f:
        for meta in mapping:
            spec = meta['spec']
            raw = rows[meta['qid']][int(spec.split('_s')[1])]
            f.write(f"BEGIN IONS\nFEATURE_ID={spec}\nPEPMASS={meta['precursor_mz']:.12g}\nCHARGE=1+\nMSLEVEL=2\n")
            for mz,intensity in raw['peaks']:
                f.write(f'{mz:.12g} {intensity:.12g}\n')
            f.write('END IONS\n\n')
    (args.output/'spectra_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (args.output/'run_record.json').write_text(json.dumps({
        'status':'truth_free_pilot_inputs_frozen','source_sha256':sha(__file__),
        'protocol_sha256':sha(args.protocol),'metadata_sha256':sha(args.metadata),
        'members_sha256':sha(args.members),'mgf_sha256':sha(mgf),
        'spectra_map_sha256':sha(args.output/'spectra_map.json'),
        'selected_qids':list(selected),'spectra':len(mapping),'smiles_in_output':False,
        'truth_formula_used':False,'sealed_scored':False},indent=2)+'\n')
    print(json.dumps({'selected_qids':list(selected),'spectra':len(mapping),
                      'by_qid':{q:len(v) for q,v in rows.items()},'mgf_sha256':sha(mgf)}))


def explanation_from_output(processed, precursor, output):
    """Count official observed-mz matches once, excluding the precursor neighborhood."""
    import numpy as np
    peaks = np.asarray(processed,dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[1] != 2 or not np.isfinite(peaks).all():
        raise ValueError('Invalid official processed spectrum')
    mz,weight = peaks[:,0],peaks[:,1]
    if np.any(mz<=0) or np.any(weight<=0) or not np.isfinite(precursor):
        raise ValueError('Invalid official processed peaks or precursor')
    if len(set(map(float,mz))) != len(mz):
        raise ValueError('Official processed spectrum has duplicate observed m/z')
    fragments = {i for i,m in enumerate(mz) if m < precursor-1.5}
    weight2 = np.square(weight)
    total = float(weight2.sum())
    fragment_total = float(weight2[list(fragments)].sum()) if fragments else 0.
    by_mz = {float(m):i for i,m in enumerate(mz)}
    table = output['output_tbl']
    observed = [] if table is None else table['mz']
    ppms = [] if table is None else table['mass_diff']
    if len(observed) != len(ppms):
        raise ValueError('Official observed m/z and ppm fields misaligned')
    matched = set()
    fragment_ppm = []
    for obs,ppm in zip(observed,ppms):
        index = by_mz.get(float(obs))
        if index is None or index in matched:
            raise ValueError('Official assignment does not map uniquely to a processed observed peak')
        matched.add(index)
        if index in fragments:
            fragment_ppm.append(float(ppm))
    matched_fragment = matched & fragments
    if not fragments or fragment_total<=0:
        e_i = e_n = None
    else:
        e_i = float(weight2[list(matched_fragment)].sum()/fragment_total)
        e_n = len(matched_fragment)/len(fragments)
    return {'processed_peak_count':len(peaks),'fragment_peak_count':len(fragments),
            'matched_fragment_peak_count':len(matched_fragment),
            'assigned_subformula_count':len(observed),
            'e_i':e_i,'e_n':e_n,
            'matched_fragment_ppm_median':None if not fragment_ppm else float(np.median(fragment_ppm)),
            'precursor_neighborhood_intensity_fraction':None if total<=0 else float((total-fragment_total)/total)}


def one(args):
    """Run one frozen formula x spectrum pair with unmodified official functions."""
    import numpy as np
    from mist import utils
    import mist.utils.spectra_utils as spectra_utils
    protocol,metadata = frozen_inputs(args.protocol,args.metadata)
    specmap = json.loads(args.spectra_map.read_text())
    input_record = json.loads((args.spectra_map.parent/'run_record.json').read_text())
    if input_record['protocol_sha256'] != sha(args.protocol) or input_record['mgf_sha256'] != sha(args.mgf):
        raise ValueError('Frozen MGF/metadata binding changed')
    selected = {row['qid']:row for row in protocol['selected']}
    if args.qid not in selected:
        raise ValueError('Unselected qid')
    row = metadata['queries'][args.qid]
    if not 0 <= args.formula_index < len(row['formulas']):
        raise ValueError('Formula index outside complete frozen list')
    formula = row['formulas'][args.formula_index]['formula']
    if not row['formulas'][args.formula_index]['upstream_supported']:
        raise ValueError('Unsupported formula cannot be passed to official parser')
    maps = [r for r in specmap if r['spec']==args.spec and r['qid']==args.qid]
    if len(maps)!=1 or args.spec not in row['hplus_positive_spec_ids']:
        raise ValueError('Spectrum not in selected complete H+ set')
    parsed = [(meta,tuples) for meta,tuples in utils.parse_spectra_mgf(args.mgf)
              if meta['FEATURE_ID']==args.spec]
    if len(parsed)!=1:
        raise ValueError('Frozen MGF spectrum missing or duplicated')
    processed = utils.process_spec_file(*parsed[0])
    if processed is None or len(processed)==0:
        raise ValueError('Official preprocessing yielded no peaks')
    processed_for_metric = processed.copy()
    processed_sha256 = hashlib.sha256(processed_for_metric.tobytes()).hexdigest()
    subset_count = []
    original = spectra_utils.get_all_subsets
    def counted(form):
        values,masses = original(form)
        subset_count.append(len(values))
        return values,masses
    spectra_utils.get_all_subsets = counted
    start = time.monotonic()
    try:
        output = utils.get_output_dict(spec_name=args.spec,spec=processed,form=formula,
                                       mass_diff_type='ppm',mass_diff_thresh=20,ion_type='[M+H]+')
    finally:
        spectra_utils.get_all_subsets = original
    official_seconds = time.monotonic()-start
    if output['cand_form']!=formula or output['cand_ion']!='[M+H]+' or len(subset_count)!=1:
        raise ValueError('Official neutral-formula assignment/instrumentation changed')
    report = explanation_from_output(processed_for_metric,maps[0]['precursor_mz'],output)
    result = {'status':'complete','qid':args.qid,'spec':args.spec,
              'formula_index':args.formula_index,'formula':formula,
              'formula_candidate_count':row['formulas'][args.formula_index]['candidate_count'],
              'cartesian_upper_bound':row['formulas'][args.formula_index]['cartesian_upper_bound'],
              'actual_enumerated_subsets_after_official_filter':subset_count[0],
              'official_assignment_seconds':official_seconds,
              'processed_peaks_sha256':processed_sha256,
              'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'source_sha256':sha(__file__),'pilot_protocol_sha256':sha(args.protocol),
              'mgf_sha256':sha(args.mgf),'truth_formula_used':False,
              'sealed_scored':False,**report}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'qid':args.qid,'spec':args.spec,'formula':formula,
                      'e_i':report['e_i'],'e_n':report['e_n'],
                      'subsets':subset_count[0],'seconds':official_seconds}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--members',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('one')
    p.add_argument('--qid',required=True)
    p.add_argument('--spec',required=True)
    p.add_argument('--formula-index',type=int,required=True)
    p.add_argument('--mgf',type=Path,required=True)
    p.add_argument('--spectra-map',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'one':one}[args.action](args)

