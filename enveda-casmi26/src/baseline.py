"""Bounded CPU spectral-library baseline; no model training or leaderboard submission."""
import argparse
from array import array
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from rdkit import Chem, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
TAUTOMERS = rdMolStandardize.TautomerEnumerator()
PROTON = 1.007276466621
WATER = 18.01056468403
# (number of neutral molecules, absolute charge, ion mass minus n*neutral mass).
ADDUCTS = {
    '[M+H]+': (1,1,PROTON), '[M-H]-': (1,1,-PROTON),
    '[M+Na]+': (1,1,22.98922070209), '[M+K]+': (1,1,38.96315790649),
    '[M+NH4]+': (1,1,18.03382555335), '[M+Cl]-': (1,1,34.96940126191),
    '[M+CH2O2-H]-': (1,1,46.0054793036-PROTON),
    '[M-H2O+H]+': (1,1,PROTON-WATER),
    '[M-2H2O+H]+': (1,1,PROTON-2*WATER),
    '[M-H2O-H]-': (1,1,-PROTON-WATER),
    '[2M+H]+': (2,1,PROTON), '[2M-H]-': (2,1,-PROTON),
    '[2M+Na]+': (2,1,22.98922070209), '[M+2H]2+': (1,2,2*PROTON),
}
BIN_WIDTH = .02
N_BINS = 100001
N_FEATURES = 2*N_BINS
CONFIG = dict(seed=20261023, validation_molecules=40, mass_ppm=15., mass_floor_da=.003,
              prefilter_ppm=30., bin_width_da=BIN_WIDTH, top_peaks=64,
              intensity_floor=.005, fragment_weight=.7, loss_weight=.3,
              query_precursor_exclusion_da=1.5, top_k=25,
              validation_source='enveda-np-examples', max_reference_spectra=500000,
              max_spectra_per_structure_adduct=8)


def neutral_mass(mz, adduct):
    if adduct not in ADDUCTS:
        raise ValueError(f'Unsupported adduct: {adduct!r}')
    n,z,delta=ADDUCTS[adduct]
    if not np.isfinite(mz) or mz <= 0:
        raise ValueError('Invalid precursor m/z')
    return (mz*z-delta)/n


@lru_cache(maxsize=100000)
def canonical_key(smiles):
    if not isinstance(smiles,str):
        return None
    try:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        key=Chem.MolToInchiKey(TAUTOMERS.Canonicalize(mol))
        return key[:14] if key else None
    except (ValueError,RuntimeError,TypeError):
        return None


@lru_cache(maxsize=100000)
def structure_info(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        return None
    key=canonical_key(smiles)
    return (key,Descriptors.ExactMolWt(mol)) if key else None


def reciprocal_rank(truth, guesses):
    key=canonical_key(truth)
    if not key:
        raise ValueError('Invalid ground-truth SMILES')
    return next((1/r for r,s in enumerate(guesses[:25],1) if canonical_key(s)==key),0.)


def peak_vector(mzs, intensities, precursor):
    mzs=np.asarray(mzs,dtype=float); intensities=np.asarray(intensities,dtype=float)
    if mzs.shape != intensities.shape or mzs.ndim != 1:
        raise ValueError('Peak arrays must be aligned one-dimensional arrays')
    valid=(np.isfinite(mzs)&np.isfinite(intensities)&(mzs>0)&(mzs<=2000)
           &(mzs<=precursor+2)&(intensities>0)&(np.abs(mzs-precursor)>1.5))
    mzs=mzs[valid]; intensities=intensities[valid]
    if not len(mzs):
        return {}
    keep=intensities>=CONFIG['intensity_floor']*intensities.max()
    mzs=mzs[keep]; intensities=intensities[keep]
    order=np.argsort(-intensities,kind='stable')[:CONFIG['top_peaks']]
    mzs=mzs[order]; weights=np.sqrt(intensities[order])
    result={}
    for positions,offset,channel_weight in [(mzs,0,.7),(precursor-mzs,N_BINS,.3)]:
        channel=defaultdict(float)
        for pos,weight in zip(positions,weights):
            if 0<pos<=2000:
                channel[int(np.floor(pos/BIN_WIDTH+.5))]+=float(weight)
        norm=np.sqrt(sum(w*w for w in channel.values()))
        if norm:
            result.update({i+offset:np.sqrt(channel_weight)*w/norm for i,w in channel.items()})
    norm=np.sqrt(sum(w*w for w in result.values()))
    return {i:w/norm for i,w in result.items()} if norm else {}


def sparse_vectors(vectors):
    rows=[]; cols=[]; vals=[]
    for r,vector in enumerate(vectors):
        for c,v in vector.items():
            rows.append(r); cols.append(c); vals.append(v)
    return csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),
                      shape=(len(vectors),N_FEATURES),dtype=np.float32)


def rank_candidates(scores, keys, excluded):
    evidence={}
    for j,key in enumerate(keys):
        if key in excluded:
            continue
        evidence.setdefault(key,np.zeros(scores.shape[0],dtype=float))
        np.maximum(evidence[key],scores[:,j],out=evidence[key])
    return sorted(((key,float(v.mean())) for key,v in evidence.items()),key=lambda x:(-x[1],x[0]))


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load_queries(data, train):
    test=pq.read_table(data/'test.parquet').to_pylist()
    groups=defaultdict(list)
    for row in test:
        neutral_mass(row['precursor_mz'],row['adduct'])  # Unknown test adducts fail loudly.
        peak_vector(row['ms2_mzs'],row['ms2_normalized_intensities'],row['precursor_mz'])
        groups[row['molecule_id']].append(row)
    np_indices=[i for i,x in enumerate(train['ingest_lib']) if x==CONFIG['validation_source']]
    np_smiles=sorted(set(train['normalized_smiles'][i] for i in np_indices))
    by_key={}
    for smiles in np_smiles:
        info=structure_info(smiles)
        if info:
            by_key.setdefault(info[0],smiles)
    ordered=sorted(by_key,key=lambda k:hashlib.sha256(f"{CONFIG['seed']}:{k}".encode()).hexdigest())
    selected=set(ordered[:CONFIG['validation_molecules']])
    truth={f'val_{key}':by_key[key] for key in sorted(selected)}
    validation=defaultdict(list)
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096):
        # Restrict costly conversion of peak arrays to selected source rows.
        sources=batch.column(batch.schema.get_field_index('ingest_lib')).to_pylist()
        indices=[i for i,x in enumerate(sources) if x==CONFIG['validation_source']]
        if not indices:
            continue
        for row in batch.take(indices).to_pylist():
            key=canonical_key(row['normalized_smiles'])
            if key in selected and row['adduct'] in ADDUCTS:
                row['molecule_id']=f'val_{key}'
                validation[row['molecule_id']].append(row)
    if set(validation)!=set(truth):
        raise ValueError('Some validation structures have no supported query spectra')
    return dict(groups),dict(validation),truth,dict(sorted(Counter(r['adduct'] for r in test).items()))


def read_metadata(data):
    cols=['normalized_smiles','inchikey14','ingest_lib','precursor_mz','adduct','ionization_mode']
    table=pq.read_table(data/'train.parquet',columns=cols)
    train=table.to_pydict()
    return train


def candidate_library(data,train,query_masses):
    n=len(train['adduct'])
    masses=np.full(n,np.nan)
    unknown=Counter()
    for i,(mz,adduct) in enumerate(zip(train['precursor_mz'],train['adduct'])):
        if adduct in ADDUCTS and mz is not None and np.isfinite(mz) and mz>0:
            masses[i]=neutral_mass(mz,adduct)
        else:
            unknown[str(adduct)]+=1
    targets=np.sort(np.asarray(query_masses))
    pos=np.searchsorted(targets,masses)
    lo=targets[np.clip(pos-1,0,len(targets)-1)]
    hi=targets[np.clip(pos,0,len(targets)-1)]
    distance=np.minimum(abs(masses-lo),abs(masses-hi))
    tolerance=np.maximum(.006,masses*CONFIG['prefilter_ppm']*1e-6)
    keep=(distance<=tolerance)&np.isfinite(masses)
    keep &= np.asarray(train['ingest_lib']) != CONFIG['validation_source']
    print(f'prefilter_rows={int(keep.sum())}',flush=True)
    metadata=[]; skipped=Counter(); start=0; counts=Counter()
    feature_indices=array('i'); feature_values=array('f'); indptr=array('q',[0])
    for batch in pq.ParquetFile(data/'train.parquet').iter_batches(batch_size=4096):
        local=np.flatnonzero(keep[start:start+batch.num_rows]); start+=batch.num_rows
        if not len(local):
            continue
        for row in batch.take(local).to_pylist():
            info=structure_info(row['normalized_smiles'])
            if not info:
                skipped['invalid_charged_or_multicomponent_structure']+=1
                continue
            # Neutral losses from multicharged/dimer precursors are not comparable to test ions.
            if ADDUCTS[row['adduct']][:2] != (1,1):
                skipped['non_monomer_or_multicharged']+=1
                continue
            vector=peak_vector(row['ms2_mzs'],row['ms2_normalized_intensities'],row['precursor_mz'])
            if not vector:
                skipped['no_usable_peaks']+=1
                continue
            identity_adduct=(info[0],row['adduct'])
            if counts[identity_adduct]>=CONFIG['max_spectra_per_structure_adduct']:
                skipped['spectrum_cap']+=1
                continue
            counts[identity_adduct]+=1
            if len(metadata)>=CONFIG['max_reference_spectra']:
                raise RuntimeError('Accepted reference spectrum budget exceeded')
            metadata.append((info[0],row['normalized_smiles'],info[1],row['ionization_mode']))
            for feature,value in sorted(vector.items()):
                feature_indices.append(feature); feature_values.append(value)
            indptr.append(len(feature_indices))
    if not metadata:
        raise RuntimeError('No candidate library spectra')
    print(f'library_spectra={len(metadata)} structures={len(set(x[0] for x in metadata))}',flush=True)
    matrix=csr_matrix((np.frombuffer(feature_values,dtype=np.float32),
                       np.frombuffer(feature_indices,dtype=np.int32),
                       np.frombuffer(indptr,dtype=np.int64)),shape=(len(metadata),N_FEATURES))
    return metadata,matrix,dict(unknown),dict(skipped)


def retrieve(groups,metadata,matrix,excluded):
    masses=np.asarray([x[2] for x in metadata]); modes=np.asarray([x[3] for x in metadata])
    smiles={x[0]:x[1] for x in metadata}
    predictions={}; diagnostics=[]
    for mol_id,rows in sorted(groups.items()):
        estimates=np.array([neutral_mass(r['precursor_mz'],r['adduct']) for r in rows])
        mass=float(np.median(estimates)); tol=max(CONFIG['mass_floor_da'],mass*CONFIG['mass_ppm']*1e-6)
        ids=np.flatnonzero(abs(masses-mass)<=tol)
        allowed=[i for i in ids if metadata[i][0] not in excluded]
        fallback=False
        if not allowed and not mol_id.startswith('val_'):
            # shortcut: nearest library structure keeps CSV valid but cannot identify absent structures.
            available=np.array([i for i,m in enumerate(metadata) if m[0] not in excluded],dtype=int)
            if not len(available):
                raise RuntimeError('Empty fallback candidate pool')
            nearest=available[np.argmin(abs(masses[available]-mass))]
            ids=np.flatnonzero(np.asarray([m[0] for m in metadata])==metadata[nearest][0])
            fallback=True
        keys=[metadata[i][0] for i in ids]
        if len(ids):
            q=sparse_vectors([peak_vector(r['ms2_mzs'],r['ms2_normalized_intensities'],r['precursor_mz']) for r in rows])
            scores=(q@matrix[ids].T).toarray()
            for j,row in enumerate(rows):
                scores[j,modes[ids]!=row['ionization_mode']]=0
            ranked=rank_candidates(scores,keys,excluded)[:25]
        else:
            ranked=[]
        predictions[mol_id]=[smiles[key] for key,_ in ranked]
        diagnostics.append(dict(molecule_id=mol_id,spectra=len(rows),neutral_mass=mass,
            mass_estimate_spread_da=float(np.ptp(estimates)),candidate_structures=len(set(keys)-excluded),
            returned=len(ranked),fallback=fallback,top_score=ranked[0][1] if ranked else None,
            truth_in_final_mass_pool=(mol_id[4:] in (set(keys)-excluded)) if mol_id.startswith('val_') else None))
    return predictions,diagnostics


def evaluate(truth,predictions):
    ranks=[reciprocal_rank(s,predictions[k]) for k,s in truth.items()]
    return dict(molecules=len(truth),mrr25=float(np.mean(ranks)),
                top1=float(np.mean([x==1 for x in ranks])),
                top5=float(np.mean([x>=.2 for x in ranks])),
                top25=float(np.mean([x>0 for x in ranks])))


def run(args):
    start=time.monotonic(); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    data=Path(args.data)
    record=dict(status='running',config=CONFIG,server=platform.node(),user=__import__('getpass').getuser(),
                project=str(Path.cwd()),python=sys.version,entry=sys.argv,gpu='none',cpu_threads=2,
                started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependencies={p:importlib.metadata.version(p) for p in ['numpy','scipy','pyarrow','rdkit']},
                files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in data.iterdir() if p.name in ['train.parquet','test.parquet','sample_submission.csv']})
    write_json(out/'run_record.json',record)
    try:
        if RDKIT_VERSION!='2026.03.3':
            raise RuntimeError(f'Official scoring version required, got {RDKIT_VERSION}')
        train=read_metadata(data)
        test,val,truth,adducts=load_queries(data,train)
        all_queries={**test,**val}
        qmass=[neutral_mass(r['precursor_mz'],r['adduct']) for rows in all_queries.values() for r in rows]
        metadata,matrix,unknown,skipped=candidate_library(data,train,qmass)
        heldout_keys={canonical_key(s) for s in truth.values()}
        write_json(out/'split.json',dict(seed=CONFIG['seed'],source=CONFIG['validation_source'],
            validation=truth,canonical_structure_keys=sorted(heldout_keys),
            query_spectrum_counts={k:len(v) for k,v in val.items()},
            library_source_excluded=CONFIG['validation_source']))
        known,known_diag=retrieve(val,metadata,matrix,set())
        heldout,heldout_diag=retrieve(val,metadata,matrix,heldout_keys)
        test_predictions,test_diag=retrieve(test,metadata,matrix,set())
        if any(canonical_key(s) in heldout_keys for values in heldout.values() for s in values):
            raise AssertionError('Validation structure leakage')
        metrics=dict(known_structure_library_search={**evaluate(truth,known),
                         'mass_filtered_candidate_coverage':float(np.mean([d['truth_in_final_mass_pool'] for d in known_diag]))},
                     structure_disjoint_library_search=evaluate(truth,heldout),
                     note='No learned model. Held-out structures are absent from candidate library; zero recall is an architectural limit. Downloadable test is training examples, not leaderboard validation.')
        write_json(out/'metrics.json',metrics)
        write_json(out/'validation_predictions.json',dict(known=known,structure_disjoint=heldout))
        write_json(out/'diagnostics.json',dict(test=test_diag,known=known_diag,structure_disjoint=heldout_diag))
        with (data/'sample_submission.csv').open(newline='',encoding='utf-8-sig') as f:
            expected=[r['molecule_id'] for r in csv.DictReader(f)]
        if len(expected)!=len(set(expected)) or set(expected)!=set(test_predictions):
            raise ValueError('Test/sample molecule IDs differ')
        with (out/'submission.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['molecule_id','smiles'])
            for key in expected:
                guesses=test_predictions[key]
                keys=[canonical_key(s) for s in guesses]
                if not 1<=len(guesses)<=25 or None in keys or len(keys)!=len(set(keys)):
                    raise ValueError(f'Invalid candidates for {key}')
                writer.writerow([key,';'.join(guesses)])
        with (out/'submission.csv').open(newline='') as f:
            actual=list(csv.DictReader(f))
        assert [r['molecule_id'] for r in actual]==expected
        p=pq.ParquetFile(data/'train.parquet')
        audit=dict(train_rows=p.metadata.num_rows,train_columns=p.schema_arrow.names,
                   supplied_inchikey14_structures=len(set(train['inchikey14'])),
                   train_sources=dict(Counter(train['ingest_lib'])),
                   test_molecules=len(test),test_spectra=sum(map(len,test.values())),test_adducts=adducts,
                   unsupported_train_adducts=unknown,skipped_reference_rows=skipped,
                   reference_spectra=len(metadata),reference_structures=len(set(m[0] for m in metadata)),
                   submission_rows=len(actual),test_mass_fallbacks=sum(d['fallback'] for d in test_diag),
                   known_truth_in_candidate_library=sum(canonical_key(s) in set(m[0] for m in metadata) for s in truth.values()))
        write_json(out/'data_audit.json',audit)
        record.update(status='complete',metrics=metrics,elapsed_seconds=time.monotonic()-start,
                      max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      submission_sha256=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
        print(json.dumps(dict(status='complete',metrics=metrics,rows=len(actual),elapsed_seconds=record['elapsed_seconds']),ensure_ascii=False),flush=True)
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        write_json(out/'run_record.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data')
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
