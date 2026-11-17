#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical validation for the local parametric-rate theorem.

Protocol:
  * independent Gaussian persistent-excitation state/input/disturbance data;
  * one model identification, one fixed-policy evaluation, one improvement;
  * fixed policy known, no SDPI iteration;
  * independent seeds and two target-noise levels.

This validates the theorem's stated local PE/well-conditioned setting.  The
closed-loop and high-noise diagnostics remain separate and are not substituted
for this test.
"""
from __future__ import annotations
import csv,json,os
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

def run(spec,Ns=(8000,16000,32000,64000,128000,256000),seeds=(0,1,2,3,4),target_noise=1e-5):
    A,B,E=spec['A'],spec['B'],spec['E']; Q,R,alpha,lam=spec['Q'],spec['R'],spec['alpha'],spec['lam']
    opt=core.solve_drc_pi(A,B,E,Q,R,alpha,lam,spec['d_bar']); V=spec['d_bar'].reshape(1,-1)
    per={}; points=[]
    for N in Ns:
        vals=[]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            X=rng.standard_normal((A.shape[0],N)); U=1.5*rng.standard_normal((B.shape[1],N))
            D=spec['d_bar'][:,None]+spec['d_std'][:,None]*rng.standard_normal((E.shape[1],N))
            Xp=A@X+B@U+E@D+target_noise*rng.standard_normal((A.shape[0],N))
            data=dict(X=X,U=U,D=D,Xp=Xp,nx=A.shape[0],nu=B.shape[1],nd=E.shape[1])
            est=dd.identify_full_ols(data)
            pe=core.policy_evaluate(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,opt['K'],opt['p'],opt['N'],opt['z'])
            imp=core.policy_improve(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,pe['P'],pe['chi'])
            # Schur-complement margin of the current value map.
            H_uu=R+alpha*est['B_hat'].T@pe['P']@est['B_hat']; H_um=alpha*est['B_hat'].T@pe['P']@est['E_hat']; H_mm=-lam*np.eye(E.shape[1])+alpha*est['E_hat'].T@pe['P']@est['E_hat']
            S_u=H_uu-H_um@np.linalg.solve(H_mm,H_um.T)
            vals.append(dict(eA=np.linalg.norm(est['A_hat']-A)/np.linalg.norm(A),eB=np.linalg.norm(est['B_hat']-B)/np.linalg.norm(B),eE=np.linalg.norm(est['E_hat']-E)/np.linalg.norm(E),eP=np.linalg.norm(pe['P']-opt['P'])/np.linalg.norm(opt['P']),eK=np.linalg.norm(imp['K']-opt['K'],'fro'),ep=np.linalg.norm(imp['p']-opt['p']),sigma_u=float(np.min(np.linalg.eigvalsh(S_u)))))
        per[int(N)]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
        points.append([N,per[int(N)]['sigma_u'],per[int(N)]['eA'],per[int(N)]['eB'],per[int(N)]['eE'],per[int(N)]['eP'],per[int(N)]['eK'],per[int(N)]['ep']])
    slopes={k:dd.loglog_slope(list(per),[per[N][k] for N in per]) for k in ('eA','eB','eE','eP','eK','ep')}
    return dict(per_n=per,slopes=slopes,points=points,target_noise=target_noise)

def main():
    rows=[]; detail=[]; rep={}
    for name,spec in (('humanoid',models.humanoid_spec()),('microgrid',models.microgrid_spec())):
        r=run(spec); rep[name]=r; s=r['slopes']
        rows.append([name,'low-noise PE',f'{s["eA"]:.4f}',f'{s["eB"]:.4f}',f'{s["eE"]:.4f}',f'{s["eP"]:.4f}',f'{s["eK"]:.4f}',f'{s["ep"]:.4f}'])
        for x in r['points']: detail.append([name,'low-noise PE']+x)
        print(name,s)
    _write('od_e5g_rigorous_rate_validation',['Scenario','Protocol','slope eA','slope eB','slope eE','slope eP','slope eK','slope ep'],rows,notes='Independent Gaussian PE design, fixed policy, one evaluation and one improvement, five seeds per N, target noise 1e-5. This is numerical validation of the local well-conditioned theorem, not a proof of a universal constant.')
    _write('od_e5g_rigorous_rate_points',['Scenario','Protocol','N','min eig S_u','mean eA','mean eB','mean eE','mean eP','mean eK','mean ep'],detail,notes='Per-N points for the low-noise PE validation.')
    with open(os.path.join(LOG,'od_e5g_rigorous_rate_validation.json'),'w',encoding='utf-8') as f: json.dump(rep,f,indent=2,default=float)
    return rep

if __name__=='__main__': main()
