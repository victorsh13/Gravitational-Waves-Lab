"""CPU scalar-only analysis. Never opens X, checkpoints, cal or test."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.m12.g1_npe_contract import LABELS, PINNED, digest, require, write_json


def load(directory):
    directory=Path(directory)
    r=json.loads((directory/'complete.json').read_text())
    require(r['complete'] and r['split']=='val' and r['identity']['version']=='M12.5.2-NPE-v1'
            and r['baseline_sha256']==PINNED['baseline'], 'Bad inference provenance')
    require(digest(directory/'posterior.npz')==r['archive_sha256'],'Changed inference archive')
    with np.load(directory/'posterior.npz',allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    require(a['label_names'].tolist()==LABELS and np.array_equal(a['idx_val'],np.arange(400000,440000))
            and len(np.unique(a['source_id']))==40000, 'Row/source/label contract mismatch')
    for k,v in a.items():
        if v.dtype.kind in 'fc': require(np.isfinite(v).all(),'Nonfinite '+k)
    require(a['mean'].shape==a['variance'].shape==a['y_physical'].shape==(40000,3), 'Bad posterior shape')
    require((a['variance']>=0).all() and (a['draws']==r['draws']).all(),'Invalid variance/draws')
    return a,r


def strata(a):
    y=a['y_physical']; masks={'all':np.ones(len(y),bool)}
    for low,high in ((10,15),(15,20),(20,25.000001)):
        masks['SNR_%g_%g'%(low,min(high,25))]=(a['snr']>=low)&(a['snr']<high)
    for e in np.unique(a['environment']): masks['environment_'+str(e)]=a['environment']==e
    for j,label in enumerate(LABELS):
        masks[label+'_Q10']=y[:,j]<=a['train_label_cuts'][0,j]
        masks[label+'_Q90']=y[:,j]>=a['train_label_cuts'][1,j]
    for mass in LABELS[:2]:
        for mt in ('Q10','Q90'):
            for st in ('Q10','Q90'):
                masks[mass+'_'+mt+'__chi_eff_'+st]=masks[mass+'_'+mt]&masks['chi_eff_'+st]
    z=(y-a['y_mean'])/a['y_std']-a['train_center']
    d=np.einsum('ni,ij,nj->n',z,a['train_precision'],z)
    bins=np.searchsorted(a['train_distance_cuts'],d,side='right')
    for i,n in enumerate(('0_80','80_95','95_99','99_100')): masks['Mahalanobis_'+n]=bins==i
    return masks


def bootstrap(values, env, count, rng, random_environments=False):
    """Paired source resampling; source IDs are unique by the frozen contract.

    Fixed environment counts target this observed mixture. Hierarchical variant
    resamples environment clusters then sources and is a sensitivity analysis
    for a broader environment population, unreliable with few clusters.
    """
    groups=[np.flatnonzero(env==e) for e in np.unique(env)]
    samples=np.empty((count,values.shape[1]))
    for b in range(count):
        selected=rng.integers(len(groups),size=len(groups)) if random_environments else range(len(groups))
        total=np.zeros(values.shape[1]); n=0
        for i in selected:
            g=groups[i]; ix=g[rng.integers(len(g),size=len(g))]
            total+=values[ix].sum(0); n+=len(ix)
        samples[b]=total/n
    return samples


def wilson(k,n,alpha):
    z=NormalDist().inv_cdf(1-alpha/2); p=k/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0,mid-radius),min(1,mid+radius)


def analyze(primary, replicate, bootstrap_count=2000):
    require(bootstrap_count>=2000,'At least 2000 bootstrap replicates')
    a,r=load(primary); a2,r2=load(replicate)
    require(r['training_seed']==123 and r2['training_seed']==124,'Fixed primary 123 and replicate 124 required')
    portable=lambda x:{k:v for k,v in x.items() if k!='dataset_mtime_ns'}
    require(portable(r['identity'])==portable(r2['identity']) and r['checkpoint_sha256']!=r2['checkpoint_sha256'],
            'Different contracts or same checkpoint')
    for k in ('idx_val','source_id','environment','y_physical','m11_pred_val','y_std','y_mean'):
        require(np.array_equal(a[k],a2[k]),'Seed comparison alignment mismatch: '+k)
    y=a['y_physical']; scale=a['y_std'].astype(float)
    m11=a['m11_pred_val'].astype(float)*scale+a['y_mean']
    mean=a['mean']; var=a['variance']; n=a['draws'][:,None]
    bse=(m11-y)**2; rse=(mean-y)**2; delta=bse-rse
    # Eight-batch delete-one jackknife for MC uncertainty of squared mean error.
    bm=a['block_mean']; loo=(8*mean[:,None,:]-bm)/7
    loo_loss=(loo-y[:,None,:])**2
    mcvar=7/8*((loo_loss-loo_loss.mean(1,keepdims=True))**2).sum(1)
    masks=strata(a); tables={}; risk=[]; metric=[]; calibration=[]; ranks=[]; info=[]; budgets=[]; seeds=[]
    rng=np.random.default_rng(125222)
    # All global target contrasts form family of 3; all region contrasts another family.
    region_family=max(1,(len(masks)-1)*3)
    coverage_family=len(masks)*4*3
    rank_family=len(masks)*4
    for group,mask in masks.items():
        size=int(mask.sum())
        if not size: continue
        alpha=.05/3 if group=='all' else .05/region_family
        env=a['environment'][mask]
        boot=bootstrap(np.concatenate((delta[mask],bse[mask],var[mask]-rse[mask]),axis=1),env,bootstrap_count,rng)
        ci=np.quantile(boot[:,:3],[alpha/2,1-alpha/2],axis=0)
        fraction_ci=np.quantile(boot[:,:3]/boot[:,3:6],[alpha/2,1-alpha/2],axis=0)
        consistency_ci=np.quantile(boot[:,6:9],[alpha/2,1-alpha/2],axis=0)
        tail_resolved=bootstrap_count*alpha/2>=10
        # Too few source rows makes inference explicitly descriptive.
        for j,label in enumerate(LABELS):
            source_se=boot[:,j].std(ddof=1)
            mcse=np.sqrt(mcvar[mask,j].sum())/size
            margin=.01*bse[mask,j].mean()
            for units,f in (('physical',1.),('standardized',1/scale[j]**2)):
                risk.append(dict(group=group,label=label,units=units,n=size,environments=len(np.unique(env)),
                    m11_mse=bse[mask,j].mean()*f,npe_mse=rse[mask,j].mean()*f,
                    npe_mse_mc_corrected=(rse[mask,j]-var[mask,j]/n[mask,0]).mean()*f,
                    favorable_delta=delta[mask,j].mean()*f,ci_low=ci[0,j]*f,ci_high=ci[1,j]*f,
                    source_se=source_se*f,mcse=mcse*f,mc_resolved=bool(mcse<=.1*source_se),
                    margin=margin*f,bootstrap_tail_resolved=tail_resolved,practical_improvement=bool(tail_resolved and size>=200 and mcse<=.1*source_se and ci[0,j]>margin),
                    equivalent=bool(tail_resolved and size>=200 and mcse<=.1*source_se and ci[0,j]>-margin and ci[1,j]<margin),
                    rmse_guardrail=bool(rse[mask,j].mean()<=1.02**2*bse[mask,j].mean()),
                    demonstrated_avoidable_fraction=delta[mask,j].mean()/bse[mask,j].mean(),
                    avoidable_fraction_ci_low=fraction_ci[0,j],
                    avoidable_fraction_ci_high=fraction_ci[1,j],
                    irreducible_fraction='UNRESOLVED',interval='validation-selected; fixed environment counts'))
                info.append(dict(group=group,label=label,units=units,n=size,
                    average_posterior_variance=var[mask,j].mean()*f,
                    empirical_mean_mse=rse[mask,j].mean()*f,
                    variance_minus_mse=(var[mask,j]-rse[mask,j]).mean()*f,
                    consistency_ci_low=consistency_ci[0,j]*f,consistency_ci_high=consistency_ci[1,j]*f,
                    prior_variance=a['prior_variance'][j]*f,
                    variance_ratio_to_prior=var[mask,j].mean()/a['prior_variance'][j],
                    mean_mcse_rms=np.sqrt(np.mean(a['mean_mcse'][mask,j]**2))*np.sqrt(f),
                    variance_mcse_of_average=np.sqrt(np.sum(a['variance_mcse'][mask,j]**2))/size*f,
                    interpretation='learned posterior variance, NOT Bayes risk'))
            for name,pred in (('M11',m11),('NPE',mean)):
                truth=y[mask,j]; prediction=pred[mask,j]; err=prediction-truth
                denom=np.sum((truth-truth.mean())**2)
                slope=np.sum((truth-truth.mean())*(prediction-prediction.mean()))/denom if denom>0 else np.nan
                metric.append(dict(group=group,label=label,model=name,n=size,rmse=np.sqrt(np.mean(err**2)),
                    mae=np.mean(abs(err)),r2=1-np.sum(err**2)/denom if denom>0 else np.nan,
                    bias=err.mean(),slope=slope,intercept=prediction.mean()-slope*truth.mean(),
                    q10_signed_bias=np.mean((pred-y)[mask & (y[:,j]<=a['train_label_cuts'][0,j]),j])
                        if np.any(mask & (y[:,j]<=a['train_label_cuts'][0,j])) else np.nan,
                    q90_signed_bias=np.mean((pred-y)[mask & (y[:,j]>=a['train_label_cuts'][1,j]),j])
                        if np.any(mask & (y[:,j]>=a['train_label_cuts'][1,j])) else np.nan))
            for level,lo,hi in ((.5,3,5),(.8,2,6),(.9,1,7),(.95,0,8)):
                covered=(y[mask,j]>=a['quantiles'][mask,lo,j])&(y[mask,j]<=a['quantiles'][mask,hi,j])
                lower,upper=wilson(int(covered.sum()),size,.05/coverage_family)
                calibration.append(dict(group=group,label=label,nominal=level,n=size,coverage=covered.mean(),
                    simultaneous_wilson_low=lower,simultaneous_wilson_high=upper,
                    width=np.mean(a['quantiles'][mask,hi,j]-a['quantiles'][mask,lo,j]),
                    prior_coverage=np.mean((y[mask,j]>=a['prior_quantiles'][lo,j]) &
                                           (y[mask,j]<=a['prior_quantiles'][hi,j])),
                    nominal_is_required=group=='all',
                    note='truth/nuisance strata are stress diagnostics, not guaranteed nominal conditional coverage'))
        # Randomized ranks remove finite-draw lattice; independent jitter fixed in analysis.
        for j,name in enumerate(LABELS+['joint_density']):
            rank=a['ranks'][mask,j] if j<3 else a['joint_density_rank'][mask]
            u=(rank+rng.random(size))/(a['draws'][mask]+1)
            ordered=np.sort(u)
            d=max(np.max(np.arange(1,size+1)/size-ordered),np.max(ordered-np.arange(size)/size))
            bound=np.sqrt(np.log(2*rank_family/.05)/(2*size))
            hist,_=np.histogram(u,bins=np.linspace(0,1,11))
            for bin_id,k in enumerate(hist):
                ranks.append(dict(group=group,label=name,bin=bin_id,count=int(k),n=size,
                    ks_distance=d,simultaneous_dkw_bound=bound,
                    interpretation='held-out rank diagnostic; not fresh iid train-mixture SBC'))
        for j,label in enumerate(LABELS):
            diff=a2['mean'][mask,j]-mean[mask,j]
            seeds.append(dict(group=group,label=label,n=size,
                seed123_mse=rse[mask,j].mean(),seed124_mse=np.mean((a2['mean'][mask,j]-y[mask,j])**2),
                mean_difference=diff.mean(),rms_mean_difference=np.sqrt(np.mean(diff**2)),
                combined_mean_mcse_rms=np.sqrt(np.mean(a['mean_mcse'][mask,j]**2+a2['mean_mcse'][mask,j]**2)),
                variance_difference=np.mean(a2['variance'][mask,j]-var[mask,j]),
                nll_difference=np.mean(a2['nll_physical'][mask]-a['nll_physical'][mask])))
    # Broader-environment uncertainty is a separate estimand, not silently pooled.
    hierarchical=bootstrap(delta,a['environment'],bootstrap_count,rng,True)
    hci=np.quantile(hierarchical,[.05/6,1-.05/6],axis=0)
    tables['environment_sensitivity']=pd.DataFrame([dict(label=l,
        ci_low=hci[0,j],ci_high=hci[1,j],environment_count=len(np.unique(a['environment'])),
        note='environment then source bootstrap; few clusters limit reliability; physical MSE') for j,l in enumerate(LABELS)])
    for prefix,mask in (('half',np.ones(len(y),bool)),('double',np.arange(len(y))<1024)):
        pred=a[prefix+'_mean']; vv=a[prefix+'_variance']
        panel_se=bootstrap(delta[mask],a['environment'][mask],bootstrap_count,rng).std(0,ddof=1)
        for j,label in enumerate(LABELS):
            budgets.append(dict(comparison=prefix,label=label,n=int(mask.sum()),
                reference_draws=int(a['draws'][0]),
                compared_draws=int(a['draws'][0])*(2 if prefix=='double' else .5),
                mean_shift_rms=np.sqrt(np.mean((pred[:,j]-mean[mask,j])**2)),
                risk_shift=np.mean((pred[:,j]-y[mask,j])**2-rse[mask,j]),
                source_se=panel_se[j],
                stable=bool(abs(np.mean((pred[:,j]-y[mask,j])**2-rse[mask,j]))<=.1*panel_se[j]),
                variance_shift=np.mean(vv[:,j]-var[mask,j]),
                note='nested budgets, not independent experiments'))
    paired=pd.DataFrame(dict(row_id=a['idx_val'],source_id=a['source_id'],environment=a['environment'],snr=a['snr']))
    for j,l in enumerate(LABELS):
        paired[l+'_delta']=delta[:,j]; paired[l+'_mc_variance']=mcvar[:,j]
    tables.update(risk=pd.DataFrame(risk),metrics=pd.DataFrame(metric),information=pd.DataFrame(info),
        coverage=pd.DataFrame(calibration),ranks=pd.DataFrame(ranks),sampling_budget=pd.DataFrame(budgets),
        seed_sensitivity=pd.DataFrame(seeds),paired_errors=paired,
        informativeness=pd.DataFrame([dict(group=g,n=int(m.sum()),
            posterior_energy_score=a['energy_score'][m].mean(),prior_energy_score=a['prior_energy_score'][m].mean(),
            nll_physical=a['nll_physical'][m].mean(),boundary_adjustments=int(a['boundary_adjusted'][m].sum()))
            for g,m in masks.items() if m.any()]))
    tables['support']=pd.DataFrame([dict(seed=rr['training_seed'],violations=rr['support_violations'],
        label_boundary_adjustments=int(aa['boundary_adjusted'].sum()),iid_draws_per_row=rr['draws']) for aa,rr in ((a,r),(a2,r2))])
    conclusion=dict(A='Evaluate paired validation risk, MC resolution, seed and budget sensitivity; confirmation remains pending.',
        B='UNRESOLVED: conditional posterior accuracy lacks independent reference/recovery and predictive validation.',
        C='Irreducible-risk fractions UNRESOLVED. Avoidable fractions quantify paired estimator differences only.',
        D='Use M11/NPE OLS slopes and signed train-Q10/Q90 tail biases, including regions; shrinkage can be Bayes optimal.',
        E='Resolved same-X improvement supports estimator-training investigation. No D/U control exists to attribute preprocessing loss. Unvalidated uncertainty cannot justify uncertainty-aware deployment.',
        predictive_diagnostics='NOT AVAILABLE: no X simulator conditional on joint targets alone is implemented; no waveform generation authorized.',
        conditional_calibration='Binned truth/SNR/environment coverage is not calibration conditional on X. Passing marginal ranks/coverage is insufficient.',
        domain=r['domain'],selection='All intervals exploratory due to validation selection; two seeds are sensitivity, not a seed-population CI.',
        cal_test='Untouched. CLI deliberately accepts validation only; confirmatory test requires a reviewed lock and separate authorization.')
    assessment=[]
    for j,label in enumerate(LABELS):
        row=tables['risk'].query("group == 'all' and units == 'physical' and label == @label").iloc[0]
        budget_ok=bool(tables['sampling_budget'].query('label == @label')['stable'].all())
        seedrow=tables['seed_sensitivity'].query("group == 'all' and label == @label").iloc[0]
        replicate_favors=bool(seedrow.seed124_mse < row.m11_mse)
        status=('exploratory same-X improvement supported' if row.practical_improvement and budget_ok and replicate_favors
                else 'risk improvement unresolved under predeclared checks')
        assessment.append(dict(label=label,A=status,posterior_accuracy='UNRESOLVED',
            primary_practical_improvement=bool(row.practical_improvement),budget_stable=budget_ok,
            replicate_favors_npe=replicate_favors,irreducible_fraction='UNRESOLVED'))
    tables['assessment']=pd.DataFrame(assessment)
    conclusion['A']=assessment
    conclusion['D']=tables['metrics'].query("group == 'all'")[
        ['label','model','slope','intercept','q10_signed_bias','q90_signed_bias']].to_dict(orient='records')
    conclusion['E']=('Prioritize a controlled estimator-training investigation for supported targets; do not attribute the cause to architecture.'
        if any(x['primary_practical_improvement'] and x['budget_stable'] and x['replicate_favors_npe'] for x in assessment)
        else 'No intervention class is established: first resolve reference approximation or numerical precision.') + ' No preprocessing-loss attribution or validated uncertainty deployment follows from this X-only experiment.'
    return tables,conclusion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--replicate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();out=a.output.resolve()
    require(out!=ROOT and ROOT not in out.parents and not out.exists(),'Use a new external output')
    tables,conclusion=analyze(a.primary,a.replicate,a.bootstrap)
    out.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items(): table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'analysis.json',dict(complete=True,answers=conclusion,bootstrap=a.bootstrap,
        code_sha256=digest(__file__),receipts={str(d):digest(d/'complete.json') for d in (a.primary,a.replicate)},
        tables={n:digest(out/(n+'.csv')) for n in tables}))
    print(json.dumps(conclusion,indent=2))

if __name__=='__main__': main()
