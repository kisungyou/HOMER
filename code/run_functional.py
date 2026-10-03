"""Paired finite Fourier experiments. Population means are known exactly."""
import json,time,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm
from core import pseudo_batch,pseudo_components,geometric_median,adaptive
from settings import RESULTS as OUT, CONFIG, manifest_settings, require_run_mode
require_run_mode()
REPS=CONFIG['functional_replications'];DIM=12;SEED=20260917;Z=norm.ppf(.975)
BOOTSTRAP_REPS=CONFIG['bootstrap_replications']

def spectra(name):
 if name=='diffuse':return np.ones(DIM)/DIM
 q=np.arange(1,DIM+1,dtype=float)**-3;return q/q.sum()

def integral(a,b):
 h=[1.]
 for j in range(1,7):
  h.extend([np.sqrt(2)*(np.sin(2*np.pi*j*b)-np.sin(2*np.pi*j*a))/(2*np.pi*j*(b-a)),np.sqrt(2)*(np.cos(2*np.pi*j*a)-np.cos(2*np.pi*j*b))/(2*np.pi*j*(b-a))])
 return np.array(h[:DIM])

def contrasts():
 smooth=[1.]
 for j in range(1,7):
  smooth.extend([np.sqrt(2)*np.cos(2*np.pi*j*.75)*np.exp(-.5*(2*np.pi*j*.05)**2),np.sqrt(2)*np.sin(2*np.pi*j*.75)*np.exp(-.5*(2*np.pi*j*.05)**2)])
 return np.stack([integral(.25,.75),integral(.75,1)-integral(0,.25),np.array(smooth[:DIM])])

def wilson(a):
 p=np.mean(a);n=len(a);den=1+Z*Z/n;mid=(p+Z*Z/(2*n))/den;rad=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/den
 return p,mid-rad,mid+rad

def inference():
 rows=[];times=[];H=contrasts();start=time.perf_counter()
 for si,spectrum in enumerate(['concentrated','diffuse']):
  sd=np.sqrt(spectra(spectrum))
  for di,dist in enumerate(['gaussian','exponential']):
   for a in [4,5,6]:
    k=2**a;m=2**(2*a);n=k*m;seed=SEED+si*100+di*10+a;rng=np.random.default_rng(seed);tick=time.perf_counter()
    if dist=='gaussian':W=rng.normal(size=(REPS,k,DIM))*sd
    else:W=(rng.gamma(m,1,size=(REPS,k,DIM))-m)/np.sqrt(m)*sd
    draw=time.perf_counter()-tick;tick=time.perf_counter();v,diagn=pseudo_batch(W,1.);outer=time.perf_counter()-tick;tick=time.perf_counter()
    score,A,obj,r,w=pseudo_components(W,v[:,None,:],1.)
    q=np.linalg.solve(A,np.broadcast_to(H.T,(REPS,DIM,3)))
    solve_residual=np.linalg.norm(A@q-H.T,axis=(1,2))/(np.linalg.norm(A,axis=(1,2))*np.linalg.norm(q,axis=(1,2))+np.linalg.norm(H))
    projected=np.einsum('bki,bih->bkh',r*w[...,None],q)
    V=np.einsum('bkh,bkl->bhl',projected,projected)/k
    sandwich=time.perf_counter()-tick;se=np.sqrt(np.diagonal(V,axis1=1,axis2=2)/n);est=v@H.T/np.sqrt(m)
    eig=np.linalg.eigvalsh(V);valid=eig[:,0]>1e-12
    for h,name in enumerate(['interval_average','evening_minus_morning','smoothed_evaluation']):
     frame=pd.DataFrame({'spectrum':spectrum,'distribution':dist,'a':a,'n':n,'k':k,'m':m,'lambda':1.,'threshold':1/np.sqrt(m),'seed':seed,'replicate':np.arange(REPS),'contrast':name,'estimate':est[:,h],'target':0.,'target_definition':'population_mean_known_exactly','se':se[:,h],'low':est[:,h]-Z*se[:,h],'high':est[:,h]+Z*se[:,h],'covered':np.abs(est[:,h])<=Z*se[:,h],'length':2*Z*se[:,h],'norm_error':np.linalg.norm(v,axis=1)/np.sqrt(m),'score_norm':diagn['score'],'certificate_standardized':diagn['certificate'],'root_n_certificate':np.sqrt(k)*diagn['certificate'],'iterations':diagn['iterations'],'solver_converged':diagn['converged'],'hessian_condition':diagn['hessian_condition'],'projected_eigen_min':eig[:,0],'relative_solve_residual':solve_residual,'covariance_valid':valid,'certificate_pass':diagn['certificate']*np.linalg.norm(H[h])<.001*se[:,h]*np.sqrt(m)})
     frame['method']='Pseudo-Huber';frame['loss']='pseudo_huber';frame['threshold_rule']='fixed_standardized';frame['branch']='positive_scale';frame['contamination']='none'
     rows.append(frame)
    times.append({'experiment':'inference','spectrum':spectrum,'distribution':dist,'k':k,'m':m,'replications':REPS,'block_law_seconds':draw,'outer_seconds':outer,'sandwich_seconds':sandwich})
    print('inference',spectrum,dist,k,m,'coverage',np.mean(np.abs(est)<=Z*se,axis=0),'fail',sum(~diagn['converged']),flush=True)
 data=pd.concat(rows,ignore_index=True);data.to_csv(OUT/'functional_inference_replicates.csv.gz',index=False)
 summaries=[]
 for keys,g in data.groupby(['spectrum','distribution','a','k','m','n','contrast']):
  coverage,low,high=wilson(g.covered);summaries.append(dict(zip(['spectrum','distribution','a','k','m','n','contrast'],keys),coverage=coverage,mc_low=low,mc_high=high,reps=len(g),mean_length=g.length.mean(),mean_norm_error=g.norm_error.mean(),solver_failures=int((~g.solver_converged).sum()),covariance_failures=int((~g.covariance_valid).sum()),certificate_failures=int((~g.certificate_pass).sum())))
 pd.DataFrame(summaries).to_csv(OUT/'functional_inference_summary.csv',index=False)
 return times

def points_stress():
 rows=[];times=[];K=32;M=64;N=K*M
 for si,spectrum in enumerate(['concentrated','diffuse']):
  sd=np.sqrt(spectra(spectrum))
  for di,dist in enumerate(['gaussian','student_t4']):
   seed=SEED+10000+si*100+di*10;rng=np.random.default_rng(seed);tick=time.perf_counter()
   raw=(rng.normal(size=(REPS,K,M,DIM)) if dist=='gaussian' else rng.standard_t(4,size=(REPS,K,M,DIM))/np.sqrt(2))*sd
   W=raw.mean(2)*np.sqrt(M);draw=time.perf_counter()-tick
   # Fixed sets for both magnitudes and every estimator.
   bad_blocks=np.stack([rng.choice(K,4,replace=False) for _ in range(REPS)])
   bad_raw=np.stack([rng.choice(N,int(.05*N),replace=False) for _ in range(REPS)])
   raw_counts=np.stack([np.bincount(ids//M,minlength=K) for ids in bad_raw])
   np.savez_compressed(OUT/f'functional_pairing_{spectrum}_{dist}.npz',bad_blocks=bad_blocks,bad_raw=bad_raw,clean_summaries=W,seed=seed)
   clean_estimates={}
   for mechanism,magnitude in [('clean',0.),('whole_block',10.),('whole_block',100.),('raw_dispersed',10.),('raw_dispersed',100.)]:
    tick=time.perf_counter();y=W.copy()
    if mechanism=='whole_block':y[np.arange(REPS)[:,None],bad_blocks,0]+=magnitude*np.sqrt(M)
    elif mechanism=='raw_dispersed':y[:,:,0]+=raw_counts*magnitude/np.sqrt(M)
    contamination_time=time.perf_counter()-tick
    actual=np.zeros(REPS) if mechanism=='clean' else (np.full(REPS,4/K) if mechanism=='whole_block' else (raw_counts>0).mean(1))
    methods={};tick=time.perf_counter();gms=[geometric_median(y[i]) for i in range(REPS)];gm_time=time.perf_counter()-tick
    gm=np.stack([g.center for g in gms]);methods['MOM']=(gm,{'score':np.array([g.score_norm for g in gms]),'certificate':np.full(REPS,np.nan),'converged':np.array([g.converged for g in gms]),'iterations':np.array([g.iterations for g in gms])},np.full(REPS,np.nan),'geometric_median',gm_time)
    mean_start=time.perf_counter();empirical=y.mean(1);mean_time=time.perf_counter()-mean_start
    methods['Mean']=(empirical,{'score':np.zeros(REPS),'certificate':np.zeros(REPS),'converged':np.ones(REPS,bool),'iterations':np.zeros(REPS,int)},np.full(REPS,np.inf),'mean',mean_time)
    for lam in [.25,1.,4.,16.]:
     tick=time.perf_counter();v,diagn=pseudo_batch(y,lam,initial=gm);elapsed=time.perf_counter()-tick
     methods[f'PH({lam:g})']=(v,diagn,np.full(REPS,lam),'fixed',elapsed)
    tick=time.perf_counter();fits=[adaptive(y[i],2.,preliminary=gms[i]) for i in range(REPS)];elapsed=time.perf_counter()-tick
    methods['Adaptive(2)']=(np.stack([f.center for f in fits]),{'score':np.array([f.score_norm for f in fits]),'certificate':np.array([f.certificate for f in fits]),'converged':np.array([f.converged for f in fits]),'iterations':np.array([f.iterations for f in fits])},np.array([f.threshold for f in fits]),'all_distance_median',elapsed)
    for method,(estimate,diagn,threshold,rule,elapsed) in methods.items():
     if mechanism=='clean':clean_estimates[method]=estimate.copy()
     frame=pd.DataFrame({'spectrum':spectrum,'distribution':dist,'n':N,'k':K,'m':M,'seed':seed,'replicate':np.arange(REPS),'mechanism':mechanism,'magnitude':magnitude,'nominal_fraction':0 if mechanism=='clean' else (.125 if mechanism=='whole_block' else .05),'realized_block_fraction':actual,'method':method,'threshold_rule':rule,'threshold':threshold/np.sqrt(M),'norm_error':np.linalg.norm(estimate,axis=1)/np.sqrt(M),'squared_error':np.sum(estimate**2,axis=1)/M,'shift_from_own_clean':np.linalg.norm(estimate-clean_estimates[method],axis=1)/np.sqrt(M),'target':'population_clean_mean_zero','score_norm':diagn['score'],'certificate_standardized':diagn['certificate'],'iterations':diagn['iterations'],'solver_converged':diagn['converged'],'preliminary_converged':np.array([g.converged for g in gms]),'preliminary_gap_bound':np.array([g.gap_bound for g in gms]),'preliminary_branch':[g.branch for g in gms]})
     frame['loss']='geometric_median' if method=='MOM' else ('quadratic' if method=='Mean' else 'pseudo_huber')
     frame['branch']='zero_scale' if np.all(threshold==0) else 'positive_scale'
     rows.append(frame);times.append({'experiment':'point_stress','spectrum':spectrum,'distribution':dist,'mechanism':mechanism,'magnitude':magnitude,'method':method,'replications':REPS,'block_construction_seconds':draw,'perturbation_seconds':contamination_time,'preliminary_seconds':(0. if method in ('Mean','MOM') else gm_time),'outer_seconds':elapsed})
    print('point',spectrum,dist,mechanism,magnitude,'MOM',np.linalg.norm(gm,axis=1).mean()/np.sqrt(M),'failures',sum(np.sum(~x[1]['converged']) for x in methods.values()),flush=True)
 data=pd.concat(rows,ignore_index=True);data.to_csv(OUT/'functional_point_replicates.csv.gz',index=False)
 summaries=[];bootstrap=np.random.default_rng(SEED+9999)
 for keys,g in data.groupby(['spectrum','distribution','mechanism','magnitude','method']):
  err=g.norm_error.to_numpy();boot=np.quantile(err[bootstrap.integers(len(err),size=(BOOTSTRAP_REPS,len(err)))],.95,axis=1)
  mean=g.squared_error.mean();sem=g.squared_error.std(ddof=1)/np.sqrt(len(g))
  summaries.append(dict(zip(['spectrum','distribution','mechanism','magnitude','method'],keys),reps=len(g),mean_error=err.mean(),mse=mean,mse_mc_low=(mean-Z*sem if keys[1]=='gaussian' else np.nan),mse_mc_high=(mean+Z*sem if keys[1]=='gaussian' else np.nan),mse_interval_status=('normal_MC_approximation' if keys[1]=='gaussian' else 'not_reported_infinite_fourth_moment'),q95=np.quantile(err,.95),q95_mc_low=np.quantile(boot,.025),q95_mc_high=np.quantile(boot,.975),mean_shift=g.shift_from_own_clean.mean(),mean_realized_block_fraction=g.realized_block_fraction.mean(),solver_failures=int((~g.solver_converged).sum()),preliminary_failures=int((~g.preliminary_converged).sum())))
 pd.DataFrame(summaries).to_csv(OUT/'functional_point_summary.csv',index=False)
 # Paired squared-error differences against MOM, using identical replicate IDs.
 comparisons=[]
 for keys,g in data.groupby(['spectrum','distribution','mechanism','magnitude']):
  pivot=g.pivot(index='replicate',columns='method',values='squared_error')
  for method in pivot:
   diff=pivot[method]-pivot.MOM;se=diff.std(ddof=1)/np.sqrt(len(diff))
   comparisons.append(dict(zip(['spectrum','distribution','mechanism','magnitude'],keys),method=method,mean_difference=diff.mean(),mc_low=(diff.mean()-Z*se if keys[1]=='gaussian' else np.nan),mc_high=(diff.mean()+Z*se if keys[1]=='gaussian' else np.nan),interval_status=('paired_normal_MC_approximation' if keys[1]=='gaussian' else 'not_reported_infinite_fourth_moment')))
 pd.DataFrame(comparisons).to_csv(OUT/'functional_paired_differences.csv',index=False)
 return times

def efficiency_ratios():
 """Paired Gaussian MSE ratios, also available when figures are skipped."""
 raw=pd.read_csv(OUT/'functional_point_replicates.csv.gz')
 summary=pd.read_csv(OUT/'functional_point_summary.csv')
 order=['MOM','PH(0.25)','PH(1)','PH(4)','PH(16)','Mean']
 for spectrum in ['concentrated','diffuse']:
  paired=raw[(raw.spectrum==spectrum)&(raw.distribution=='gaussian')&(raw.mechanism=='clean')].pivot(index='replicate',columns='method',values='squared_error')
  q=summary[(summary.spectrum==spectrum)&(summary.distribution=='gaussian')&(summary.mechanism=='clean')].set_index('method').loc[order]
  rng=np.random.default_rng(20260919);indices=rng.integers(len(paired),size=(BOOTSTRAP_REPS,len(paired)))
  denominator=paired['Mean'].to_numpy()[indices].mean(1);rows=[]
  for method in order:
   ratio=paired[method].to_numpy()[indices].mean(1)/denominator
   lower,upper=np.quantile(ratio,[.025,.975])
   rows.append({'spectrum':spectrum,'method':method,'mse_ratio':q.loc[method,'mse']/q.loc['Mean','mse'],'mc_low':lower,'mc_high':upper,'bootstrap_reps':BOOTSTRAP_REPS,'seed':20260919})
  pd.DataFrame(rows).to_csv(OUT/f'efficiency_ratio_{spectrum}.csv',index=False)

def main():
 start=time.perf_counter();times=inference()+points_stress();efficiency_ratios();pd.DataFrame(times).to_csv(OUT/'functional_timing.csv',index=False)
 (OUT/'functional_manifest.json').write_text(json.dumps({**manifest_settings(),'seed':SEED,'replications':REPS,'dimension':DIM,'basis':'1, sqrt(2)cos(2pi*t),sqrt(2)sin(2pi*t),...,sqrt(2)cos(12pi*t)','spectra':{'concentrated':'j^-3 normalized','diffuse':'1/12'},'mean':'zero Fourier coefficients','coefficient_laws':'independent Gaussian N(0,1), centered Exponential(1), or t4/sqrt(2), scaled by sqrt(eigenvalue)','contamination':'add magnitude e1 in coefficient space; same IDs at both magnitudes','runtime_seconds':time.perf_counter()-start,'bootstrap_seed':SEED+9999,'quantile_bootstrap_replicates':BOOTSTRAP_REPS,'efficiency_bootstrap_seed':20260919,'efficiency_bootstrap_replicates':BOOTSTRAP_REPS,'score_tolerance':1e-12,'geometric_subgradient_tolerance':1e-10,'certificate_se_fraction':.001,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'contrasts':contrasts().tolist()},indent=2))
if __name__=='__main__':main()
