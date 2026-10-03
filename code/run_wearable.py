"""Descriptive subject-level stability from average window acceleration energy."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from core import geometric_median,adaptive
from settings import RESULTS as OUT, DATA, DERIVED, manifest_settings, require_run_mode
require_run_mode()
SEED=20260918

def preprocess(root):
 rows=[];hashes={}
 for split in ('train','test'):
  sub=np.loadtxt(root/split/f'subject_{split}.txt',dtype=int);labels=np.loadtxt(root/split/f'y_{split}.txt',dtype=int)
  matrices=[]
  for axis in 'xyz':
   path=root/split/'Inertial Signals'/f'body_acc_{axis}_{split}.txt';matrices.append(np.loadtxt(path));hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
  # Average energy per window BEFORE averaging within subject and activity.
  energy=np.sum(np.stack(matrices,axis=-1)**2,axis=-1).mean(axis=1)
  for subject in np.unique(sub):
   for activity in np.unique(labels):
    mask=(sub==subject)&(labels==activity)
    if mask.any():rows.append({'subject':subject,'activity':activity,'n_windows':int(mask.sum()),'energy':float(energy[mask].mean()),'split':split})
 DATA.mkdir(parents=True,exist_ok=True)
 frame=pd.DataFrame(rows);frame.to_csv(DATA/'subject_window_energy.csv',index=False)
 (DATA/'preprocessing_manifest.json').write_text(json.dumps({'formula':'subject mean of window means of body_acc_x^2+body_acc_y^2+body_acc_z^2','units':'squared gravitational acceleration units as distributed','window_samples':128,'overlap':'50 percent; windows are not independent units','window_alignment':'energy is averaged within each window, requiring no phase alignment across subjects','input_sha256':hashes,'source':'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones','historical_readme_license':'original README included in supplied archive says commercial use prohibited','current_portal_license':'CC BY 4.0, official portal checked September 2026'},indent=2))
 return frame

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-path',type=Path);args=parser.parse_args()
 data=preprocess(args.data_path) if args.data_path else pd.read_csv(DERIVED/'subject_window_energy.csv')
 common=sorted(set.intersection(*[set(g.subject) for _,g in data.groupby('activity')]))
 if len(common)!=30:raise ValueError(f'Expected 30 complete subjects, found {len(common)}')
 rng=np.random.default_rng(SEED);order=rng.permutation(common);blocks=order.reshape(5,6)
 affected_whole=blocks[0].tolist();affected_dispersed=blocks[:3,0].tolist()
 methods=['Mean','MOM','Adaptive(1)','Adaptive(2)','Adaptive(4)','Adaptive(16)','Canonical(2)']
 rows=[];partition=[]
 for b,ids in enumerate(blocks):
  for subject in ids:partition.append({'block':b,'subject':int(subject)})
 pd.DataFrame(partition).to_csv(DATA/'subject_partition.csv',index=False)
 for activity,g in data.groupby('activity'):
  x=g.set_index('subject').loc[common,'energy'];clean={}
  for mechanism,amplitude in [('clean',1.),('one_block',3.),('one_block',10.),('dispersed_subjects',3.),('dispersed_subjects',10.)]:
   affected=[] if mechanism=='clean' else (affected_whole if mechanism=='one_block' else affected_dispersed)
   stressed=x.copy();stressed.loc[affected]*=amplitude**2
   tick=time.perf_counter();z=np.array([stressed.loc[ids].mean() for ids in blocks]);block_time=time.perf_counter()-tick
   tick=time.perf_counter();gm=geometric_median(z);gm_time=time.perf_counter()-tick
   for method in methods:
    tick=time.perf_counter();fit=None
    if method=='Mean':estimate=z.mean();tau=np.nan;branch='mean'
    elif method=='MOM':estimate=gm.center[0];tau=np.nan;branch='geometric_median';fit=gm
    else:
     c=float(method.split('(')[1][:-1]);fit=adaptive(z,c,'canonical' if method.startswith('Canonical') else 'pseudo',preliminary=gm);estimate=fit.center[0];tau=fit.threshold;branch=fit.branch
    if mechanism=='clean':clean[method]=estimate
    rows.append({'activity':int(activity),'seed':SEED,'n_subjects':30,'k':5,'m':6,'mechanism':mechanism,'amplitude_multiplier':amplitude,'energy_multiplier':amplitude**2,'affected_subjects':';'.join(map(str,affected)),'n_affected_subjects':len(affected),'realized_block_fraction':0 if mechanism=='clean' else (1/5 if mechanism=='one_block' else 3/5),'method':method,'threshold':tau,'branch':branch,'estimate':estimate,'shift_from_own_clean':abs(estimate-clean[method]),'shift_relative_clean_mean':abs(estimate-clean[method])/x.mean(),'clean_subject_mean':x.mean(),'solver_converged':True if fit is None else fit.converged,'score_norm':0. if fit is None else fit.score_norm,'certificate':0. if fit is None else fit.certificate,'iterations':0 if fit is None else fit.iterations,'block_construction_seconds':block_time,'preliminary_seconds':(0. if method in ('Mean','MOM') else gm_time),'outer_seconds':time.perf_counter()-tick+(gm_time if method=='MOM' else 0.)})
 frame=pd.DataFrame(rows);frame['loss']=frame.method.map(lambda name:'quadratic' if name=='Mean' else ('geometric_median' if name=='MOM' else ('canonical_huber' if name.startswith('Canonical') else 'pseudo_huber')));frame.to_csv(OUT/'wearable_results.csv',index=False)
 (OUT/'wearable_manifest.json').write_text(json.dumps({**manifest_settings(),'seed':SEED,'subjects':common,'blocks':blocks.tolist(),'one_block_affected':affected_whole,'dispersed_affected':affected_dispersed,'activities':{1:'WALKING',2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING',5:'STANDING',6:'LAYING'},'paired':'same 30 participants, partitions, and affected participant IDs across all activities, thresholds, and magnitudes','inference':'none, descriptive stability only; sample provides 30 subject units and 5 block summaries','target':'equal-subject average of each subject average window-level body acceleration energy','data_usage':'Public anonymized IDs, no new recruitment; no independently verified consent or ethics-review details','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 print(frame.groupby(['mechanism','amplitude_multiplier','method']).shift_relative_clean_mean.mean().to_string())
if __name__=='__main__':main()
