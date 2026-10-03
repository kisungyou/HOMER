"""Paired covariance-summary stability illustration, not covariance inference.

The complete protocol was fixed before execution. Each block input is its
unbiased centered sample covariance, with divisor m-1. Full-matrix
vectorization is isometric for the Frobenius norm. Aggregation is performed
in the affine span of the inputs for numerical efficiency, without changing
that geometry. No thresholds or display replications are selected by results.
"""
from pathlib import Path
import hashlib,json,platform,time
import numpy as np
import pandas as pd
from scipy.linalg import block_diag,toeplitz
from core import geometric_median,adaptive
from settings import RESULTS as OUT, CONFIG, manifest_settings, require_run_mode
require_run_mode()
SEED=20260922
REPS=CONFIG['covariance_replications']
D=12
K=32
M=64
DELTAS=(0.,8.,32.)
METHODS=('Mean','GMed','Adaptive PH')
BOOTSTRAP_REPS=CONFIG['bootstrap_replications']
BOOTSTRAP_SEED=20260923
DISPLAY_REPLICATE=0
DISPLAY_DELTA=32.


def sample_covariances(x):
    """Input shape (blocks, observations, coordinates)."""
    centered=x-x.mean(axis=1,keepdims=True)
    return np.einsum('kmi,kmj->kij',centered,centered)/(x.shape[1]-1)


def full_gmedian_diagnostic(flat,v):
    residual=flat-v
    distances=np.linalg.norm(residual,axis=1)
    nz=distances>0
    score=max(0.,np.linalg.norm((residual[nz]/distances[nz,None]).sum(0))-(~nz).sum())/len(flat)
    return score,float(distances.max()*score)


def aggregate(covariances):
    """The affine reduction retains full-matrix Frobenius distances."""
    flat=covariances.reshape(K,-1)
    base=flat.mean(0)
    residual=flat-base
    _,singular,basis=np.linalg.svd(residual,full_matrices=False)
    rank=int(np.sum(singular>np.finfo(float).eps*max(residual.shape)*singular[0]))
    basis=basis[:rank]
    coordinates=residual@basis.T
    reconstruction=np.linalg.norm(coordinates@basis-residual)
    if reconstruction>1e-11*max(1.,np.linalg.norm(residual)):
        raise ArithmeticError('Affine reduction failed its reconstruction check')
    g=geometric_median(coordinates)
    ph=adaptive(coordinates,c=2.,preliminary=g)
    fits={'GMed':g,'Adaptive PH':ph}
    estimates={'Mean':base.reshape(D,D)}
    diagnostics={'Mean':{'iterations':0,'score_norm':0.,'converged':True,'threshold':np.nan,'branch':'arithmetic_mean','certificate':0.,'objective_gap_bound':np.nan}}
    for name,fit in fits.items():
        if name=='GMed' and fit.branch in ('exact_vertex','exact_majority'):
            center=flat[np.argmin(np.linalg.norm(coordinates-fit.center,axis=1))].copy()
        else:
            center=base+fit.center@basis
        matrix=center.reshape(D,D)
        symmetry_error=np.linalg.norm(matrix-matrix.T)
        matrix=(matrix+matrix.T)/2
        center=matrix.ravel()
        estimates[name]=matrix
        if name=='GMed':
            score,gap=full_gmedian_diagnostic(flat,center)
            certificate=np.nan
            converged=fit.converged and score<=1.01e-10
        elif fit.branch=='zero_scale':
            score,gap=full_gmedian_diagnostic(flat,center)
            certificate=np.nan
            converged=fit.converged
        else:
            r=flat-center
            w=1/np.hypot(1.,np.linalg.norm(r,axis=1)/fit.threshold)
            score=float(np.linalg.norm((r*w[:,None]).mean(0)))
            diameter=max(2*np.linalg.norm(residual,axis=1).max(),np.linalg.norm(r,axis=1).max())
            certificate=float(score*(1+(diameter/fit.threshold)**2)**1.5)
            gap=np.nan
            converged=fit.converged and score<=1.01e-12*max(1.,fit.threshold)
        diagnostics[name]={'iterations':fit.iterations,'score_norm':score,'converged':bool(converged),'threshold':fit.threshold,'branch':fit.branch,'certificate':certificate,'objective_gap_bound':gap,'symmetry_error_before_roundoff_symmetrization':symmetry_error}
    for name in METHODS:
        diagnostics[name].update(affine_rank=rank,affine_reconstruction_error=float(reconstruction),preliminary_converged=bool(g.converged),preliminary_gap_bound=diagnostics['GMed']['objective_gap_bound'])
    return estimates,diagnostics


def bootstrap_summaries(data):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    indices=rng.integers(REPS,size=(BOOTSTRAP_REPS,REPS))
    rows=[]
    for (delta,method),g in data.groupby(['delta','method'],sort=True):
        g=g.sort_values('replicate')
        e=g.frobenius_error.to_numpy()
        e2=e**2
        mean_boot=e[indices].mean(1)
        rmse_boot=np.sqrt(e2[indices].mean(1))
        q_boot=np.quantile(e[indices],.95,axis=1)
        rows.append({'delta':delta,'method':method,'replications':len(g),'mean_frobenius_error':e.mean(),'mean_error_mc_low':np.quantile(mean_boot,.025),'mean_error_mc_high':np.quantile(mean_boot,.975),'mse':e2.mean(),'mse_mc_low':np.quantile(rmse_boot**2,.025),'mse_mc_high':np.quantile(rmse_boot**2,.975),'rmse':np.sqrt(e2.mean()),'rmse_mc_low':np.quantile(rmse_boot,.025),'rmse_mc_high':np.quantile(rmse_boot,.975),'q95_error':np.quantile(e,.95),'q95_mc_low':np.quantile(q_boot,.025),'q95_mc_high':np.quantile(q_boot,.975),'mean_shift_from_own_clean':g.shift_from_own_clean.mean(),'min_eigenvalue':g.min_eigenvalue.min(),'max_score_norm':g.score_norm.max(),'max_certificate':g.certificate.max(),'max_preliminary_gap':g.preliminary_gap_bound.max(),'solver_failures':int((~g.converged).sum()),'preliminary_failures':int((~g.preliminary_converged).sum()),'psd_failures':int((~g.psd_valid).sum()),'bootstrap_replications':BOOTSTRAP_REPS})
    pd.DataFrame(rows).to_csv(OUT/'covariance_summary.csv',index=False)
    comparisons=[]
    for delta,g in data.groupby('delta',sort=True):
        pivot=g.pivot(index='replicate',columns='method',values='frobenius_error').sort_index()
        for first,second in [('Adaptive PH','Mean'),('Adaptive PH','GMed'),('GMed','Mean')]:
            for power,metric in [(1,'paired_Frobenius_error_difference'),(2,'paired_squared_Frobenius_error_difference')]:
                diff=(pivot[first]**power-pivot[second]**power).to_numpy()
                boot=diff[indices].mean(1)
                comparisons.append({'delta':delta,'first_method':first,'second_method':second,'metric':metric,'mean_difference':diff.mean(),'mc_low':np.quantile(boot,.025),'mc_high':np.quantile(boot,.975),'replications':REPS,'bootstrap_replications':BOOTSTRAP_REPS})
    pd.DataFrame(comparisons).to_csv(OUT/'covariance_paired_differences.csv',index=False)
    return rows


def main():
    start=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    kernel=toeplitz(.65**np.arange(4))
    truth=.9*block_diag(kernel,kernel,kernel)+.1*np.ones((D,D))
    mean=np.linspace(-1.,1.,D)
    direction=np.r_[np.ones(4),-np.ones(4),np.zeros(4)]/np.sqrt(8.)
    factor=np.linalg.cholesky(truth)
    rows=[]
    all_estimates=np.empty((REPS,len(DELTAS),len(METHODS),D,D))
    clean_inputs=np.empty((REPS,K,D,D))
    corrupted_inputs=np.empty((REPS,2,4,D,D))
    selected_blocks=np.empty((REPS,4),int)
    timings=[]
    checks={}
    for rep in range(REPS):
        tick=time.perf_counter()
        rng=np.random.default_rng(SEED+rep)
        raw=rng.normal(size=(K,M,D))@factor.T+mean
        selected=np.sort(rng.choice(K,4,replace=False))
        shared_factor=rng.normal(size=(4,M))
        selected_blocks[rep]=selected
        original=sample_covariances(raw)
        clean_inputs[rep]=original
        construction_seconds=time.perf_counter()-tick
        if rep==0:
            checks['centered_covariance_numpy_max_error']=float(np.max(np.abs(original[0]-np.cov(raw[0],rowvar=False,ddof=1))))
            checks['translation_invariance_max_error']=float(np.max(np.abs(original-sample_covariances(raw+np.arange(D)))))
            checks['truth_min_eigenvalue']=float(np.linalg.eigvalsh(truth).min())
            if checks['centered_covariance_numpy_max_error']>1e-12 or checks['translation_invariance_max_error']>1e-12:
                raise ArithmeticError('Centered covariance verification failed')
        own_clean={}
        for scenario,delta in enumerate(DELTAS):
            inputs=original.copy()
            if delta:
                perturbed=raw[selected]+np.sqrt(delta)*shared_factor[:,:,None]*direction[None,None,:]
                bad=sample_covariances(perturbed)
                inputs[selected]=bad
                corrupted_inputs[rep,scenario-1]=bad
            tick=time.perf_counter()
            estimates,diag=aggregate(inputs)
            aggregation_seconds=time.perf_counter()-tick
            if rep==0:
                flat=inputs.reshape(K,-1)
                direct_g=geometric_median(flat)
                direct_ph=adaptive(flat,c=2.,preliminary=direct_g)
                for name,direct in [('GMed',direct_g),('Adaptive PH',direct_ph)]:
                    discrepancy=float(np.linalg.norm(estimates[name].ravel()-direct.center))
                    checks[f'full_coordinate_agreement_delta_{delta:g}_{name}']=discrepancy
                    if discrepancy>1e-8 or not direct.converged:
                        raise ArithmeticError('Independent full-coordinate aggregation disagrees')
            for mi,method in enumerate(METHODS):
                estimate=estimates[method]
                if not delta:own_clean[method]=estimate.copy()
                error=float(np.linalg.norm(estimate-truth,'fro'))
                eigmin=float(np.linalg.eigvalsh(estimate).min())
                all_estimates[rep,scenario,mi]=estimate
                rows.append({'seed':SEED+rep,'replicate':rep,'delta':delta,'method':method,'target':'clean_population_covariance','d':D,'k':K,'m':M,'n':K*M,'n_affected_blocks':4 if delta else 0,'realized_block_fraction':4/K if delta else 0.,'c':2. if method=='Adaptive PH' else np.nan,'frobenius_error':error,'squared_frobenius_error':error**2,'relative_frobenius_error':error/np.linalg.norm(truth,'fro'),'shift_from_own_clean':float(np.linalg.norm(estimate-own_clean[method],'fro')),'min_eigenvalue':eigmin,'psd_valid':eigmin>=-1e-10,'symmetry_error':float(np.linalg.norm(estimate-estimate.T)),**diag[method]})
            timings.append({'replicate':rep,'delta':delta,'raw_and_clean_summary_seconds':construction_seconds,'aggregate_three_methods_seconds':aggregation_seconds})
        if (rep+1)%100==0:print(f'Completed {rep+1}/{REPS} paired covariance replications',flush=True)
    data=pd.DataFrame(rows)
    data.to_csv(OUT/'covariance_replicates.csv.gz',index=False)
    summary=bootstrap_summaries(data)
    np.savez_compressed(OUT/'covariance_estimates.npz',estimates=all_estimates,truth=truth,mean=mean,direction=direction,delta=np.array(DELTAS),methods=np.array(METHODS),display_replicate=DISPLAY_REPLICATE,display_delta=DISPLAY_DELTA)
    np.savez_compressed(OUT/'covariance_inputs.npz',clean_block_covariances=clean_inputs,corrupted_block_covariances=corrupted_inputs,bad_block_indices=selected_blocks,delta=np.array(DELTAS[1:]))
    pd.DataFrame(timings).to_csv(OUT/'covariance_timing.csv',index=False)
    manifest={**manifest_settings(),'experiment':'covariance-summary stability illustration','seed':SEED,'replication_seed':'seed + replication index','replications':REPS,'dimension':D,'k':K,'m':M,'n':K*M,'mean':'linearly spaced values from -1 to 1','truth_formula':'0.9*block_diag(Toeplitz(0.65**arange(4)), repeated three times) + 0.1*ones(12,12)','truth_matrix':truth.tolist(),'input_estimator':'within-block sample covariance centered at its own block mean, divided by m-1','geometry':'full matrix vectorization, Euclidean norm equals Frobenius norm','comparator_mean':'unweighted arithmetic mean of the block covariance inputs, not the pooled sample covariance','aggregation':'geometric median and corrected adaptive pseudo-Huber with c=2','computation':'orthonormal affine-span reduction, verified reconstruction; symmetric roundoff correction only, no PSD projection','stress':'add sqrt(delta)*g_i*v to each raw observation of four selected blocks; g_i independent N(0,1)','stress_direction':direction.tolist(),'delta':list(DELTAS),'pairing':'same raw observations, four block indices, and Gaussian factor draws across both positive delta values and all methods','display_replicate':DISPLAY_REPLICATE,'display_delta':DISPLAY_DELTA,'display_selection':'index zero fixed before execution, without checking results','intervals':'pointwise 95 percent percentile bootstrap Monte Carlo intervals, including paired differences; no data-analysis confidence intervals','bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_replications':BOOTSTRAP_REPS,'scientific_scope':'general subset-estimator stability illustration only; no covariance inference or new covariance-aggregation novelty claim','python':platform.python_version(),'numpy':np.__version__,'runtime_seconds':time.perf_counter()-start,'checks':checks,'failures':{'solver':int((~data.converged).sum()),'preliminary':int((~data.preliminary_converged).sum()),'PSD':int((~data.psd_valid).sum())},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'covariance_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(summary)[['delta','method','mean_frobenius_error','mean_error_mc_low','mean_error_mc_high','q95_error','solver_failures']].to_string(index=False),flush=True)
    print('Numerical failures:',manifest['failures'],flush=True)

if __name__=='__main__':main()
