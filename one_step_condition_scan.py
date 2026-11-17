#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Large-sample, condition-controlled one-step consistency scan.

This is a stricter version of one_step_consistency.py:
  * larger N grid and more independent seeds;
  * records identification covariance/Schur condition numbers;
  * reports slopes both on all batches and on a well-conditioned subset;
  * uses independent train seeds for every N and score-only test data.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import data_driven as dd
import drc_core as core
import models

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'output'); TBL=os.path.join(OUT,'tables'); LOG=os.path.join(OUT,'logs')
for d in (OUT,TBL,LOG): os.makedirs(d,exist_ok=True)

def _write(name,headers,rows,notes=None):
    with open(os.path.join(TBL,name+'.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)
    with open(os.path.join(TBL,name+'.md'),'w',encoding='utf-8') as f:
        f.write('| '+' | '.join(headers)+' |\n'); f.write('|'+'|'.join(['---']*len(headers))+'|\n')
        for r in rows: f.write('| '+' | '.join(str(x) for x in r)+' |\n')
        if notes: f.write('\n'+notes+'\n')

def _one(spec,N,seed):
    A,B,E=spec['A'],spec['B'],spec['E']; Q,R,alpha,lam=spec['Q'],spec['R'],spec['alpha'],spec['lam']
    opt=core.solve_drc_pi(A,B,E,Q,R,alpha,lam,spec['d_bar']); V=spec['d_bar'].reshape(1,-1)
    data=dd.collect_closed_loop(spec,opt['K'],opt['p'],opt['N'],opt['z'],N,
        seed=seed,expl_scale=2.0,state_obs_noise_std=1e-4,input_obs_noise_std=1e-5,
        dist_obs_noise_std=1e-5,target_obs_noise_std=1e-4,dist_probe_std=1.0*spec['d_std'])
    est=dd.identify_full_iv(data,n_lags=1,instrument_mode='probe')
    pe=core.policy_evaluate(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,
        opt['K'],opt['p'],opt['N'],opt['z'])
    imp=core.policy_improve(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,
        pe['P'],pe['chi'])
    # conditioning of the structural design and value map
    cond_phi=float(est.get('cond_Phi',np.inf))
    cond_schur=float('nan')
    try:
        H_uu=R+alpha*est['B_hat'].T@pe['P']@est['B_hat']; H_um=alpha*est['B_hat'].T@pe['P']@est['E_hat']
        H_mm=-lam*np.eye(E.shape[1])+alpha*est['E_hat'].T@pe['P']@est['E_hat']
        S=H_uu-H_um@np.linalg.solve(H_mm,H_um.T); cond_schur=float(np.linalg.cond(S))
    except Exception: pass
    return dict(eA=np.linalg.norm(est['A_hat']-A)/np.linalg.norm(A),
                eB=np.linalg.norm(est['B_hat']-B)/np.linalg.norm(B),
                eE=np.linalg.norm(est['E_hat']-E)/np.linalg.norm(E),
                eP=np.linalg.norm(pe['P']-opt['P'])/np.linalg.norm(opt['P']),
                eK=np.linalg.norm(imp['K']-opt['K'],'fro'),
                ep=np.linalg.norm(imp['p']-opt['p']),
                cond_phi=cond_phi,cond_schur=cond_schur)

def run(spec,Ns=(8000,16000,32000,64000,128000),seeds=range(5)):
    per={}; per_good={}; all_rows=[]
    for N in Ns:
        rows=[_one(spec,N,s) for s in seeds]
        for r in rows: r['N']=int(N)
        all_rows.extend(rows)
        per[int(N)]={k:float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != 'N'}
        per[int(N)]['cond_phi_max']=float(np.nanmax([r['cond_phi'] for r in rows]))
        per[int(N)]['cond_schur_max']=float(np.nanmax([r['cond_schur'] for r in rows]))
        per[int(N)]['n']=len(rows)
    keys=('eA','eB','eE','eP','eK','ep')
    slopes={k:dd.loglog_slope(list(per),[per[N][k] for N in per]) for k in keys}
    # Strict conditioning subset: below the 75th percentile of both diagnostics.
    cp=np.array([r['cond_phi'] for r in all_rows]); cs=np.array([r['cond_schur'] for r in all_rows])
    qp=float(np.nanquantile(cp,0.75)); qs=float(np.nanquantile(cs,0.75))
    good=[r for r in all_rows if r['cond_phi']<=qp and r['cond_schur']<=qs]
    for N in Ns:
        rr=[r for r in good if r['N']==int(N)]
        if rr:
            per_good[int(N)]={k:float(np.nanmean([r[k] for r in rr])) for k in keys}
            per_good[int(N)]['n']=len(rr)
    slopes_good={k:float('nan') for k in keys}
    if len(per_good)>=2:
        slopes_good={k:dd.loglog_slope(list(per_good),[per_good[N][k] for N in per_good]) for k in keys}
    return dict(per_n=per,per_n_good=per_good,slopes=slopes,slopes_good=slopes_good,
                all_points=all_rows,
                well_conditioned_points=int(len(good)),
                total_points=len(all_rows),
                cond_phi_quantile=qp,cond_schur_quantile=qs)

def main():
    rows=[]; report={}; detail=[]
    for name,spec in (('humanoid',models.humanoid_spec()),('microgrid',models.microgrid_spec())):
        r=run(spec); report[name]=r; s=r['slopes']
        rows.append([name,f'{s["eA"]:.4f}',f'{s["eB"]:.4f}',f'{s["eE"]:.4f}',f'{s["eP"]:.4f}',f'{s["eK"]:.4f}',f'{s["ep"]:.4f}',f'{r["slopes_good"]["eP"]:.4f}',f'{r["slopes_good"]["eK"]:.4f}',r['well_conditioned_points'],r['total_points'],f'{r["cond_phi_quantile"]:.4e}',f'{r["cond_schur_quantile"]:.4e}'])
        for N,v in r['per_n'].items(): detail.append([name,N,v['n'],f'{v["eA"]:.4e}',f'{v["eB"]:.4e}',f'{v["eE"]:.4e}',f'{v["eP"]:.4e}',f'{v["eK"]:.4e}',f'{v["ep"]:.4e}',f'{v["cond_phi_max"]:.4e}',f'{v["cond_schur_max"]:.4e}'])
        print(name,s,'well',r['well_conditioned_points'],'/',r['total_points'])
    _write('od_e5e_one_step_condition_scan',['Scenario','slope eA','slope eB','slope eE','slope eP','slope eK','slope ep','filtered slope eP','filtered slope eK','well-conditioned points','total points','Q75 cond Phi','Q75 cond Schur'],rows,notes='Larger-N scan with exploration scale 2.0, probe scale 1.0, five seeds per N. The main slopes retain all batches; condition diagnostics are reported rather than silently filtering difficult batches.')
    _write('od_e5e_one_step_condition_points',['Scenario','N','n','mean eA','mean eB','mean eE','mean eP','mean eK','mean ep','max cond Phi','max cond Schur'],detail,notes='Per-N points for the condition-controlled one-step scan.')
    with open(os.path.join(LOG,'od_e5e_one_step_condition_scan.json'),'w',encoding='utf-8') as f: json.dump({k:{'per_n':v['per_n'],'per_n_good':v['per_n_good'],'slopes':v['slopes'],'slopes_good':v['slopes_good'],'conditioning':{x:v[x] for x in ('well_conditioned_points','total_points','cond_phi_quantile','cond_schur_quantile')}} for k,v in report.items()},f,indent=2,default=float)
    return report

if __name__=='__main__': main()
