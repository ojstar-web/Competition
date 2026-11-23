"""Evaluate frozen fixed61 development predictions without rerunning inference."""
import argparse, getpass, hashlib, json, platform, resource, sys, time
from collections import Counter
from pathlib import Path
import baseline as b

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'members':'outputs/phase5_20261023_01/members.json',
 'predictions':'outputs/phase5_20261023_12/predictions.json',
 'prediction_run':'outputs/phase5_20261023_12/run_record.json',
 'fixed61_protocol':'outputs/phase5_20261023_11/protocol.json',
 'vectors':'outputs/phase5_20261023_11/preds/fp_preds_phase5_all66_fixed61.p',
 'policy':'outputs/phase5_20261023_07/policy.json',
 'pools':'outputs/phase5_20261023_02/pools.json',
 'formulas_primary':'outputs/phase5_20261023_02/formulas_primary.json',
 'formulas_remaining':'outputs/phase3_20261023_03/formula_audit.json',
 'checkpoint':'references/mist_fp_canopus_pretrain.ckpt',
 'predict_source':'src/phase5_predict.py',
 'wrapper_source':'src/mist_phase2.py',
 'evaluate_source':'src/phase5_evaluate.py'}
ARMS=('mass_only','single_formula_hybrid','fm')
QIDS=[f'q{i:03}' for i in range(140)]
def file(name): return ROOT/FILES[name]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()
def read(path): return json.loads(path.read_text())
def save(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def check_predictions():
 pred=read(file('predictions'))
 if tuple(pred)!=ARMS: raise ValueError('Prediction policy fields changed')
 keys={}
 for arm in ARMS:
  if list(pred[arm])!=QIDS: raise ValueError('Prediction qid order changed: '+arm)
  for q in QIDS:
   values=pred[arm][q]
   if not isinstance(values,list) or len(values)>25: raise ValueError('Invalid top25: '+arm+'/'+q)
   canon=[]
   for s in values:
    if not isinstance(s,str): raise ValueError('Non-SMILES candidate')
    if s not in keys: keys[s]=b.canonical_key(s)
    if not keys[s]: raise ValueError('Invalid candidate SMILES: '+arm+'/'+q)
    canon.append(keys[s])
   if len(canon)!=len(set(canon)): raise ValueError('Duplicate canonical candidate: '+arm+'/'+q)
 return pred,keys
def freeze(out):
 if out.exists(): raise FileExistsError(out)
 hashes={name:sha(file(name)) for name in FILES}
 pinned={
 'members':'577fb395cdb8122b783b3f143f7af640d0871d89a01103fb865580fa909de943',
 'predictions':'5ccbb209ee95ee964aadd652e0ff7208297865e661fa82223965e4851b1c4ce4',
 'fixed61_protocol':'4ca9b341361b2b8d85781c7391e12ca58c2e4096433b0c5d209f2fead400e356',
 'vectors':'7aacd2da274b532cdb38f89d227465059ad1dd5abbc8fc5fb0a5d6ef32ce3d0e'}
 if any(hashes[k]!=v for k,v in pinned.items()): raise ValueError('Frozen file identity changed')
 run=read(file('prediction_run'))
 for field,name in [('predictions_sha256','predictions'),('candidate_protocol_sha256','fixed61_protocol'),('model_prediction_sha256','vectors'),('source_sha256','predict_source'),('wrapper_source_sha256','wrapper_source'),('policy_sha256','policy'),('pools_sha256','pools'),('formulas_primary_sha256','formulas_primary'),('formulas_remaining_sha256','formulas_remaining')]:
  if run[field]!=hashes[name]: raise ValueError('Prediction provenance mismatch: '+field)
 if run['status']!='fixed61_candidate_predictions_frozen_before_history_comparison_or_truth': raise ValueError('Unfrozen predictions')
 pred,_=check_predictions()  # Candidate SMILES only; no truth member content.
 policy=read(file('policy'))
 expected={'single_formula_model':15,'multi_gate_model':12,'multi_gate_single_candidate':1,'mass_only_fallback':112}
 if list(policy)!=QIDS or dict(Counter(x['status'] for x in policy.values()))!=expected: raise ValueError('Policy branches changed')
 protocol={'status':'frozen_before_fixed61_truth_evaluation','authorization':'User approved new fixed61 development baseline',
  'scope':'140 exposed fresh_dev q000-q139; sealed70 excluded',
  'historical_phase5_regression':'FAIL preserved (old9 4/9; old remaining76 76/76 literal matches)',
  'arms':list(ARMS),'denominators':{'all140':140,'primary64':64,'remaining76':76},
  'score':'RDKit2026 baseline.canonical_key(prediction SMILES); first top25 exact canonical key match to frozen member canonical_target_key yields 1/rank, else 0',
  'policy_counts':expected,'prediction_lengths':{arm:[min(map(len,pred[arm].values())),max(map(len,pred[arm].values()))] for arm in ARMS},
  'input_sha256':hashes,'sealed_scored':False,'inference_repeated':False,'predictions_modified':False}
 out.mkdir(parents=True)
 save(out/'protocol.json',protocol)
 print(json.dumps({'status':protocol['status'],'protocol_sha256':sha(out/'protocol.json'),'predictions_sha256':hashes['predictions']}),flush=True)
def evaluate(out):
 start=time.monotonic(); protocol_path=out/'protocol.json'
 protocol=read(protocol_path)
 if protocol['status']!='frozen_before_fixed61_truth_evaluation' or protocol['sealed_scored']: raise ValueError('Protocol not frozen')
 if (out/'metrics.json').exists(): raise FileExistsError('Already evaluated')
 for name,value in protocol['input_sha256'].items():
  if sha(file(name))!=value: raise ValueError('Frozen input changed: '+name)
 pred,keys=check_predictions()
 members=read(file('members'))['members']  # First target key access, after freeze.
 if len(members)!=140 or [x['qid'] for x in members]!=QIDS or [x['fresh_dev_index'] for x in members]!=list(range(140)): raise ValueError('Member mapping changed')
 truth={x['qid']:x['canonical_target_key'] for x in members}
 if len(set(truth.values()))!=140 or any(len(k)!=14 for k in truth.values()): raise ValueError('Canonical target keys changed')
 ranks={arm:{q:next((i for i,s in enumerate(pred[arm][q],1) if keys[s]==truth[q]),None) for q in QIDS} for arm in ARMS}
 def rr(arm,q): return 0.0 if ranks[arm][q] is None else 1/ranks[arm][q]
 def metric(arm,qs):
  values=[rr(arm,q) for q in qs]; n=len(qs)
  a=sum(x==1 for x in values); c=sum(x>=.2 for x in values); d=sum(x>0 for x in values)
  return {'molecules':n,'mrr25':sum(values)/n,'top1_count':a,'top5_count':c,'top25_count':d,'top1':a/n,'top5':c/n,'top25':d/n}
 def compare(left,right,qs):
  changes=[rr(right,q)-rr(left,q) for q in qs]
  return {'win':sum(x>0 for x in changes),'tie':sum(x==0 for x in changes),'loss':sum(x<0 for x in changes)}
 result={}
 for name,qs in [('all140',QIDS),('primary64',QIDS[:64]),('remaining76',QIDS[64:])]:
  result[name]={arm:metric(arm,qs) for arm in ARMS}
  result[name]['comparisons']={left+'_to_'+right:compare(left,right,qs) for left,right in [('mass_only','single_formula_hybrid'),('single_formula_hybrid','fm'),('mass_only','fm')]}
 prior3=read(ROOT/'outputs/phase3_20261023_07/metrics.json')
 prior4=read(ROOT/'outputs/phase4_20261023_14/metrics.json')
 for actual,old in [(result['remaining76']['mass_only'],prior3['mass_only_all76']),(result['remaining76']['fm'],prior4['main_all76']['FM'])]:
  if any(abs(actual[k]-old[k])>1e-12 for k in ('mrr25','top1','top5','top25')): raise ValueError('Unchanged historical ranking score mismatch')
 policy=read(file('policy'))
 rows=[{'qid':q,'cohort':'primary64' if i<64 else 'remaining76','policy_status':policy[q]['status'],
        'rank':{arm:ranks[arm][q] for arm in ARMS},'rr':{arm:rr(arm,q) for arm in ARMS},
        'delta_mass_to_single':rr('single_formula_hybrid',q)-rr('mass_only',q),
        'delta_single_to_fm':rr('fm',q)-rr('single_formula_hybrid',q),
        'delta_mass_to_fm':rr('fm',q)-rr('mass_only',q)} for i,q in enumerate(QIDS)]
 save(out/'metrics.json',result); save(out/'per_molecule.json',rows)
 save(out/'run_record.json',{'status':'fixed61_development_truth_evaluated_once','protocol_sha256':sha(protocol_path),
  'input_sha256':protocol['input_sha256'],'metrics_sha256':sha(out/'metrics.json'),
  'per_molecule_sha256':sha(out/'per_molecule.json'),'server':platform.node(),'user':getpass.getuser(),
  'python':sys.version,'rdkit':__import__('rdkit').__version__,'command':sys.argv,
  'cpu_threads':2,'gpu':'none','elapsed_seconds':time.monotonic()-start,
  'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'sealed_scored':False,'inference_repeated':False,'historical_phase5_regression':'FAIL retained'})
 print(json.dumps({'status':'evaluated','metrics_sha256':sha(out/'metrics.json'),'all140':result['all140']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','evaluate']); p.add_argument('--out',type=Path,required=True)
 a=p.parse_args(); {'freeze':freeze,'evaluate':evaluate}[a.action](a.out)
