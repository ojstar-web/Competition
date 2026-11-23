"""Official COCONUT mass-window candidates and separate mass-only control."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
import baseline as b

SOURCE_URL = 'https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv_lite-10-2026.zip'
SOURCE_SHA256 = 'a602502a4fa0950cc2e59eed690dccd082d1a051bf9f85ccda50a82dd22ca154'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def within_wide_window(raw_mass, query_masses, sorted_input=False):
    if not math.isfinite(raw_mass) or raw_mass <= 0:
        return False
    ordered = query_masses if sorted_input else sorted(query_masses)
    pos = bisect_left(ordered, raw_mass)
    return any(abs(raw_mass-ordered[i]) <= max(.05, 50e-6*ordered[i])
               for i in (pos-1, pos) if 0 <= i < len(ordered))


def mass_only_predictions(mass, candidates):
    tolerance = max(b.CONFIG['mass_floor_da'], mass*b.CONFIG['mass_ppm']*1e-6)
    pool = [c for c in candidates if abs(c['exact_mass']-mass) <= tolerance]
    pool.sort(key=lambda c: (abs(c['exact_mass']-mass), c['key']))
    return [c['smiles'] for c in pool[:25]], len(pool)


def load_query_masses(data, keys, qid_start=0, allow_empty=False):
    selected = {key:f'q{i+qid_start:03}' for i,key in enumerate(keys)}
    groups = {qid:[] for qid in selected.values()}
    audit = {qid:{'source_spectra':0,'legal_mass_spectra':0,
                  'unsupported_adduct':0,'invalid_precursor_or_mass':0}
             for qid in groups}
    columns = ['ingest_lib','normalized_smiles','precursor_mz','adduct']
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096, columns=columns):
        sources = batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices = [i for i,source in enumerate(sources) if source == b.CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            qid = selected.get(b.canonical_key(row['normalized_smiles']))
            if qid is None:
                continue
            audit[qid]['source_spectra'] += 1
            if row['adduct'] not in b.ADDUCTS:
                audit[qid]['unsupported_adduct'] += 1
                continue
            precursor = row['precursor_mz']
            if precursor is None or not math.isfinite(precursor) or precursor <= 0:
                audit[qid]['invalid_precursor_or_mass'] += 1
                continue
            mass = b.neutral_mass(precursor, row['adduct'])
            if not math.isfinite(mass) or mass <= 0:
                audit[qid]['invalid_precursor_or_mass'] += 1
                continue
            groups[qid].append(float(mass))
            audit[qid]['legal_mass_spectra'] += 1
    if any(v['source_spectra']==0 for v in audit.values()):
        raise ValueError('A frozen query structure has no source spectra')
    if not allow_empty and any(not masses for masses in groups.values()):
        raise ValueError('A query structure has no supported mass')
    return (groups,audit) if allow_empty else groups


def frozen_remaining_members(manifest_path, members_path):
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != (manifest_path.parent/'split_manifest.sha256').read_text().split()[0]:
        raise ValueError('Parent split SHA mismatch')
    content = members_path.read_bytes()
    member_hash = hashlib.sha256(content).hexdigest()
    protocol_path = members_path.parent/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['members_sha256'] != member_hash or protocol['parent_split_sha256'] != manifest_hash:
        raise ValueError('Remaining-member protocol binding mismatch')
    addendum_path = members_path.parent/'evaluation_addendum.json'
    addendum = json.loads(addendum_path.read_text())
    protocol_hash = sha256(protocol_path)
    if addendum['parent_protocol_sha256'] != protocol_hash or addendum['parent_members_sha256'] != member_hash:
        raise ValueError('Evaluation addendum binding mismatch')
    manifest = json.loads(manifest_bytes)
    ordered = manifest['partitions']['fresh_dev'][64:]
    members = json.loads(content)['members']
    if len(ordered)!=76 or len(members)!=76 or [m['canonical_target_key'] for m in members]!=ordered:
        raise ValueError('Remaining 76 order changed')
    if any(m['qid']!=f'q{i+64:03}' or m['fresh_dev_index']!=i+64 for i,m in enumerate(members)):
        raise ValueError('Remaining qid mapping changed')
    if set(ordered) & (set(manifest['primary_fresh_dev_keys']) |
                       set(manifest['partitions']['legacy_dev']) |
                       set(manifest['partitions']['sealed'])):
        raise ValueError('Remaining partition overlap')
    return ordered, manifest_hash, member_hash, protocol_hash, sha256(addendum_path)

def frozen_manifest(path):
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != (path.parent/'split_manifest.sha256').read_text().split()[0]:
        raise ValueError('Split manifest SHA256 mismatch')
    manifest = json.loads(content)
    keys = manifest['primary_fresh_dev_keys']
    if len(keys) != 64 or set(keys) & set(manifest['partitions']['sealed']):
        raise ValueError('Only the frozen 64 fresh-dev keys may enter this candidate run')
    return keys, digest


def build(data, archive, manifest_path, output, members_path=None, test_mass_groups=None, test_query_audit=None):
    start = time.monotonic()
    test_mode = test_mass_groups is not None
    remaining = members_path is not None
    if test_mode:
        if manifest_path is not None or remaining or test_query_audit is None:
            raise ValueError('Test candidate build requires only test masses and audit')
        keys = list(test_mass_groups)
        mass_groups = test_mass_groups
        query_audit = test_query_audit
        manifest_hash = member_hash = protocol_hash = addendum_hash = None
    elif remaining:
        keys, manifest_hash, member_hash, protocol_hash, addendum_hash = frozen_remaining_members(manifest_path, members_path)
        mass_groups, query_audit = load_query_masses(data, keys, qid_start=64, allow_empty=True)
    else:
        keys, manifest_hash = frozen_manifest(manifest_path)
        mass_groups = load_query_masses(data, keys)
        member_hash = protocol_hash = addendum_hash = None
    if b.RDKIT_VERSION != '2026.03.3':
        raise RuntimeError('Official RDKit version required')
    if archive.stat().st_size != 199096787 or sha256(archive) != SOURCE_SHA256:
        raise ValueError('COCONUT archive identity mismatch')
    query_masses = sorted(m for values in mass_groups.values() for m in values)
    output.mkdir(parents=True, exist_ok=False)
    record = {'status':'running', 'mode':'test_only' if test_mode else 'development', 'server':platform.node(), 'project':str(Path.cwd()),
              'entry':sys.argv, 'source_url':SOURCE_URL, 'source_zip_sha256':SOURCE_SHA256,
              'manifest_sha256':manifest_hash, 'members_sha256':member_hash,
              'protocol_sha256':protocol_hash, 'evaluation_addendum_sha256':addendum_hash,
              'rdkit':b.RDKIT_VERSION, 'source_sha256':sha256(Path(__file__)),
              'cpu_threads':2, 'virtual_memory_limit_gib':12, 'timeout_seconds':900, 'gpu':'none'}
    if remaining or test_mode:
        b.write_json(output/'query_mass_audit.json', query_audit)
        record['query_structures_without_legal_mass'] = sum(not masses for masses in mass_groups.values())
        record['legal_query_spectra'] = len(query_masses)
    b.write_json(output/'run_record.json', record)
    print(json.dumps({'stage':'query_masses_ready','queries':len(keys),'legal_spectra':len(query_masses),
                      'without_legal_mass':sum(not masses for masses in mass_groups.values())}),flush=True)
    try:
        counts = Counter()
        candidates = {}
        discrepancies = []
        with zipfile.ZipFile(archive) as z:
            members = z.namelist()
            if members != ['coconut_csv_lite-10-2026.csv']:
                raise ValueError('Unexpected ZIP members')
            with z.open(members[0]) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))
                required = {'canonical_smiles','exact_molecular_weight','standard_inchi_key'}
                if not required <= set(reader.fieldnames):
                    raise ValueError('Missing COCONUT columns')
                for row in reader:
                    counts['csv_rows'] += 1
                    if (remaining or test_mode) and counts['csv_rows'] % 100000 == 0:
                        print(json.dumps({'stage':'csv_scan','rows':counts['csv_rows'],
                                          'wide_hits':counts['wide_window_rows'],
                                          'unique_keys':len(candidates),
                                          'elapsed_seconds':time.monotonic()-start}),flush=True)
                    try:
                        raw_mass = float(row['exact_molecular_weight'])
                    except (TypeError, ValueError):
                        counts['missing_or_nonnumeric_raw_mass'] += 1
                        continue
                    if not math.isfinite(raw_mass) or raw_mass <= 0:
                        counts['nonpositive_or_nonfinite_raw_mass'] += 1
                        continue
                    if not within_wide_window(raw_mass, query_masses, sorted_input=True):
                        continue
                    counts['wide_window_rows'] += 1
                    smiles = row['canonical_smiles']
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                    if mol is None:
                        counts['invalid_smiles_in_wide_window'] += 1
                        continue
                    if len(Chem.GetMolFrags(mol)) != 1:
                        counts['multicomponent_in_wide_window'] += 1
                        continue
                    if Chem.GetFormalCharge(mol) != 0:
                        counts['charged_in_wide_window'] += 1
                        continue
                    info = b.structure_info(smiles)
                    if info is None:
                        counts['other_structure_failures_in_wide_window'] += 1
                        continue
                    key, exact_mass = info
                    discrepancies.append(abs(raw_mass-exact_mass))
                    counts['valid_wide_window_rows'] += 1
                    if key in candidates:
                        counts['duplicate_canonical_key_rows'] += 1
                        continue
                    candidates[key] = {'key':key, 'smiles':smiles, 'exact_mass':exact_mass,
                                       'raw_mass':raw_mass, 'coconut_id':row['identifier']}
                    if remaining or test_mode:
                        candidates[key]['csv_row_number'] = counts['csv_rows']
        if not candidates and not (remaining or test_mode):
            raise RuntimeError('No COCONUT candidates in query mass windows')
        frozen = {'source_url':SOURCE_URL, 'source_sha256':SOURCE_SHA256,
                  'manifest_sha256':manifest_hash, 'members_sha256':member_hash,
                  'protocol_sha256':protocol_hash, 'evaluation_addendum_sha256':addendum_hash,
                  'query_mass_da_by_id':
                  {qid:(float(np.median(values)) if values else None)
                   for qid,values in mass_groups.items()},
                  'query_spectrum_masses_da':query_masses,
                  'raw_prefilter':'any query-spectrum neutral mass ±max(0.05 Da, 50 ppm)',
                  'final_filter':'median query neutral mass ±max(0.003 Da, 15 ppm)',
                  'candidates':[candidates[key] for key in sorted(candidates)]}
        path = output/'candidates.json'
        b.write_json(path, frozen)
        candidate_hash = sha256(path)
        (output/'candidates.sha256').write_text(candidate_hash+'  candidates.json\n',encoding='utf-8')
        counts['unique_official_keys_wide_window'] = len(candidates)
        counts['query_structures'] = len(keys)
        counts['query_spectra'] = len(query_masses)
        if discrepancies:
            counts['abs_raw_vs_rdkit_mass_da_quantiles'] = {
                name:float(np.quantile(discrepancies, q)) for name,q in
                [('min',0),('p50',.5),('p90',.9),('p99',.99),('max',1)]}
        b.write_json(output/'candidate_audit.json',dict(counts))
        record.update(status='complete', candidate_sha256=candidate_hash,
                      elapsed_seconds=time.monotonic()-start,
                      max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        b.write_json(output/'run_record.json',record)
    print(json.dumps({'status':record['status'],'audit':dict(counts),
                      'candidate_sha256':record['candidate_sha256'],
                      'elapsed_seconds':record['elapsed_seconds'],
                      'max_rss_mib':record['max_rss_mib']}))


def evaluate(data, manifest_path, candidates_path, output):
    keys, manifest_hash = frozen_manifest(manifest_path)
    frozen_bytes = candidates_path.read_bytes()
    candidate_hash = hashlib.sha256(frozen_bytes).hexdigest()
    if candidate_hash != (candidates_path.parent/'candidates.sha256').read_text().split()[0]:
        raise ValueError('Candidate universe SHA256 mismatch')
    frozen = json.loads(frozen_bytes)
    if frozen['manifest_sha256'] != manifest_hash:
        raise ValueError('Candidate and split manifest differ')
    # Truth is loaded only after the independent candidate list has been frozen.
    from phase1_eval import load_fresh_queries
    groups, truth, _ = load_fresh_queries(data, keys)
    candidates = frozen['candidates']
    candidate_keys = {c['key'] for c in candidates}
    predicted, counts, final_covered = {}, {}, {}
    for qid,rows in sorted(groups.items()):
        masses = [b.neutral_mass(r['precursor_mz'],r['adduct']) for r in rows]
        median = float(np.median(masses))
        if abs(median-frozen['query_mass_da_by_id'][qid]) > 1e-9:
            raise ValueError(f'Query mass changed since candidate freeze: {qid}')
        predicted[qid], counts[qid] = mass_only_predictions(median, candidates)
        tolerance = max(b.CONFIG['mass_floor_da'],median*b.CONFIG['mass_ppm']*1e-6)
        final_covered[qid] = any(c['key']==b.canonical_key(truth[qid]) and
                                  abs(c['exact_mass']-median)<=tolerance for c in candidates)
    raw_covered = {qid:s for qid,s in truth.items() if b.canonical_key(s) in candidate_keys}
    covered = {qid:s for qid,s in truth.items() if final_covered[qid]}
    metrics = {'mass_only_all':b.evaluate(truth,predicted),
               'mass_only_final_filter_covered':b.evaluate(covered,predicted) if covered else None,
               'raw_mass_window_coverage':len(raw_covered)/len(truth),
               'final_filter_coverage':len(covered)/len(truth),
               'candidate_count_quantiles':{name:float(np.quantile(list(counts.values()),q)) for name,q in
                   [('min',0),('p25',.25),('p50',.5),('p75',.75),('max',1)]},
               'note':'Every CSV row was scanned; rows outside raw mass windows were not fully normalized. Source mass errors can exclude structures. No truth was used to add candidates.'}
    output.mkdir(parents=True,exist_ok=False)
    # S3 is an isolation-only negative control; it deliberately removes all query keys.
    s3_remaining = candidate_keys-set(keys)
    s3_audit = {'status':'candidate_isolation_only_no_ranking',
                'source_candidate_sha256':candidate_hash,
                'query_keys_removed':len(candidate_keys & set(keys)),
                'candidate_keys_before':len(candidate_keys),
                'candidate_keys_after':len(s3_remaining),
                'query_key_intersection_after':len(s3_remaining & set(keys)),
                'truth_coverage_after':0.0,
                'note':'Truth-key removal is an artificial S3 negative control. Combined with the S2 reference exclusion audit, it does not test new-structure generation.'}
    if s3_audit['query_key_intersection_after']:
        raise AssertionError('S3 candidate isolation failed')
    b.write_json(output/'metrics.json',metrics)
    b.write_json(output/'s3_isolation_audit.json',s3_audit)
    b.write_json(output/'predictions.json',predicted)
    b.write_json(output/'coverage_by_query.json',
                 {qid:{'raw_mass_window':qid in raw_covered,'final_filter':final_covered[qid],
                       'candidate_count':counts[qid]} for qid in sorted(truth)})
    b.write_json(output/'run_record.json',{'status':'complete','server':platform.node(),
                 'project':str(Path.cwd()),'entry':sys.argv,'manifest_sha256':manifest_hash,
                 'candidate_sha256':candidate_hash,'source_sha256':sha256(Path(__file__)),
                 'cpu_threads':2,'virtual_memory_limit_gib':12,'gpu':'none','metrics':metrics})
    print(json.dumps({'status':'complete','metrics':metrics}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['build','evaluate'])
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--archive',type=Path,default=Path('data/coconut_csv_lite-10-2026.zip'))
    parser.add_argument('--candidates',type=Path)
    parser.add_argument('--members',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='build':
        build(args.data,args.archive,args.manifest,args.output,args.members)
    else:
        if args.candidates is None:
            parser.error('--candidates required for evaluate')
        evaluate(args.data,args.manifest,args.candidates,args.output)


