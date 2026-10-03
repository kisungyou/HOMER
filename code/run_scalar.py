"""Exact finite-sum scalar targets and paired interval diagnostics."""
import json,time,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd,scipy
from scipy.stats import norm,t
from scipy.integrate import quad
from core import exact_binomial_target
from settings import RESULTS as OUT, CONFIG, manifest_settings, require_run_mode
require_run_mode()
REPS=CONFIG['scalar_replications'];SEED=20260916;Z=norm.ppf(.975)

def scalar_fit(y,lam):
 fit_start=time.perf_counter()
 lo=y.min(1);hi=y.max(1);v=y.mean(1)
 count=np.zeros(len(y),int)
 for _ in range(45):
  r=y-v[:,None];w=1/np.hypot(1,r/lam);score=(r*w).mean(1);A=(w**3).mean(1)
  active=abs(score)>1e-13*max(1.,lam)
  if not active.any():break
  lo=np.where(score>0,v,lo);hi=np.where(score<0,v,hi)
  candidate=v+score/A
  candidate=np.where((candidate<=lo)|(candidate>=hi),.5*(lo+hi),candidate)
  v=np.where(active,candidate,v);count+=active
 outer_seconds=time.perf_counter()-fit_start;variance_start=time.perf_counter()
 r=y-v[:,None];w=1/np.hypot(1,r/lam);A=(w**3).mean(1);B=((r*w)**2).mean(1)
 score=abs((r*w).mean(1));cert=score*(1+((y.max(1)-y.min(1))/lam)**2)**1.5
 return v,B/A**2,score,cert,count,outer_seconds,time.perf_counter()-variance_start

def wilson(x):
 n=len(x);p=np.mean(x);den=1+Z*Z/n;mid=(p+Z*Z/(2*n))/den;rad=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/den
 return p,mid-rad,mid+rad

def main():
 cells=[]
 for a in (4,5,6):
  for schedule,k,m in [('slow',2**a,2**(2*a)),('boundary',2**(3*a//2),2**((3*a+1)//2)),('many',2**(2*a),2**a)]:
   cells.append(('bernoulli',schedule,a,k,m,.1,1.))
 for k,m in [(16,256),(32,1024),(64,4096)]:cells.append(('gaussian','reference',0,k,m,np.nan,1.))
 for lam in (.5,2.,4.):cells.append(('bernoulli','threshold',5,128,256,.1,lam))
 cells.append(('bernoulli','skewness',5,128,256,.25,1.))
 for k in (8,16):
  for m in (16,256,4096):cells.append(('gaussian','fixed_k',0,k,m,np.nan,1.))
 for k in (8,16,32,64):cells.append(('gaussian','endpoint',0,k,256,np.nan,1e6))
 # Identical Gaussian standardized draws across m at each fixed k.
 rows=[];targets=[];summaries=[];timings=[];start=time.perf_counter()
 for cell,(dist,schedule,a,k,m,p,lam) in enumerate(cells):
  seed=(SEED+4 if schedule=='threshold' else SEED+cell) if schedule!='fixed_k' else SEED+1000+k
  rng=np.random.default_rng(seed);tick=time.perf_counter()
  if dist=='bernoulli':
   target=exact_binomial_target(m,p,lam);y=(rng.binomial(m,p,(REPS,k))-m*p)/np.sqrt(m*p*(1-p))
  else:
   A=quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*(1+x*x/lam**2)**(-1.5),-12,12,epsabs=1e-13)[0]
   B=quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)*x*x/(1+x*x/lam**2),-12,12,epsabs=1e-13)[0]
   target={'m':m,'p':p,'lambda':lam,'u':0.,'theta':0.,'A':A,'B':B,'V':B/A**2,'score_residual':0.,'pmf_mass_error':np.nan,'underflowed_terms':0,'target_type':'Gaussian_quadrature_12sigma_tail_bound'}
   y=rng.normal(size=(REPS,k))
  draw_time=time.perf_counter()-tick;tick=time.perf_counter()
  v,V,score,cert,iters,fit_time,sandwich_time=scalar_fit(y,lam)
  n=k*m;estimate=v/np.sqrt(m);se=np.sqrt(V/n);oracle=np.sqrt(target['V']/n)
  empirical=y.mean(1)/np.sqrt(m);mean_se=np.std(y,axis=1,ddof=1)/np.sqrt(n)
  frame=pd.DataFrame({'cell':cell,'replicate':np.arange(REPS),'seed':seed,'distribution':dist,'schedule':schedule,'a':a,'n':n,'k':k,'m':m,'p':p,'lambda':lam,'threshold':lam/np.sqrt(m),'estimate':estimate,'target':target['theta'],'population_V':target['V'],'estimated_V':V,'score_norm':score,'certificate_standardized':cert,'root_n_certificate':np.sqrt(k)*cert,'iterations':iters,'solver_converged':score<=1e-13*max(1.,lam),'covariance_valid':V>0,'certificate_pass':cert<.001*np.sqrt(V/k),'plugin_low':estimate-Z*se,'plugin_high':estimate+Z*se,'oracle_low':estimate-Z*oracle,'oracle_high':estimate+Z*oracle,'plugin_length':2*Z*se,'oracle_length':2*Z*oracle,'mean_estimate':empirical,'mean_low':empirical-Z*mean_se,'mean_high':empirical+Z*mean_se,'mean_t_low':empirical-t.ppf(.975,k-1)*mean_se,'mean_t_high':empirical+t.ppf(.975,k-1)*mean_se})
  for variance in ('plugin','oracle'):
   for estimand,true in [('mean',0.),('target',target['theta'])]:
    frame[f'{variance}_{estimand}_covered']=(frame[f'{variance}_low']<=true)&(frame[f'{variance}_high']>=true)
  frame['mean_covered']=(frame.mean_low<=0)&(frame.mean_high>=0);frame['mean_t_covered']=(frame.mean_t_low<=0)&(frame.mean_t_high>=0)
  frame['method']='Pseudo-Huber';frame['loss']='pseudo_huber';frame['threshold_rule']='fixed_standardized';frame['branch']='positive_scale';frame['contamination']='none'
  frame['target_approximation']='finite_sum' if dist=='bernoulli' else 'quadrature_tail_less_4e-30'
  rows.append(frame);targets.append({'cell':cell,'distribution':dist,'schedule':schedule,**target})
  for metric in [c for c in frame if c.endswith('_covered')]:
   coverage,low,high=wilson(frame[metric]);summaries.append({'cell':cell,'distribution':dist,'schedule':schedule,'a':a,'k':k,'m':m,'n':n,'p':p,'lambda':lam,'metric':metric,'reps':REPS,'coverage':coverage,'mc_low':low,'mc_high':high,'mean_plugin_length':frame.plugin_length.mean(),'mean_oracle_length':frame.oracle_length.mean(),'solver_failures':int((~frame.solver_converged).sum()),'covariance_failures':int((~frame.covariance_valid).sum()),'certificate_failures':int((~frame.certificate_pass).sum())})
  timings.append({'cell':cell,'block_law_seconds':draw_time,'outer_seconds':fit_time,'sandwich_seconds':sandwich_time})
  print(cell,dist,schedule,k,m,'target',round(target['theta'],7),'cover',round(frame.plugin_mean_covered.mean(),4),round(frame.plugin_target_covered.mean(),4),'certfail',int((~frame.certificate_pass).sum()),flush=True)
 pd.concat(rows,ignore_index=True).to_csv(OUT/'scalar_replicates.csv.gz',index=False)
 pd.DataFrame(summaries).to_csv(OUT/'scalar_summary.csv',index=False);pd.DataFrame(targets).to_csv(OUT/'scalar_targets.csv',index=False);pd.DataFrame(timings).to_csv(OUT/'scalar_timing.csv',index=False)
 config={**manifest_settings(),'seed':SEED,'reps_per_cell':REPS,'cells':len(cells),'runtime_seconds':time.perf_counter()-start,'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),'root_tolerance':1e-13,'certificate_se_fraction':.001,'finite_sum':'All m+1 PMF terms, normalized; floating underflow below representable mass, no intentional truncation','gaussian_quadrature':'[-12,12], tail <4e-33 and bounded score contribution <=lambda^2 times tail','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'scalar_manifest.json').write_text(json.dumps(config,indent=2))
if __name__=='__main__':main()
