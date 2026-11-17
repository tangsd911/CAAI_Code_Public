#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Well-conditioned PE diagnostic for the one-step consistency theorem.

This experiment is deliberately separate from the closed-loop Algorithm 4
simulation.  It generates independent Gaussian state/input/disturbance
regressors, identifies A,B,E by ordinary least squares, and performs exactly
one fixed-policy evaluation and one improvement.  The purpose is to separate
identification conditioning from the additional correlations and stopping
effects in the closed-loop SDPI loop.
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

def run(spec,Ns=(8000,16000,32000,64000,128000,256000),seeds=(0,1,2)):
    A,B,E=spec['A'],spec['B'],spec['E']; Q,R,alpha,lam=spec['Q'],spec['R'],spec['alpha'],spec['lam']
    opt=core.solve_drc_pi(A,B,E,Q,R,alpha,lam,spec['d_bar']); V=spec['d_bar'].reshape(1,-1)
    per={}; details=[]
    for N in Ns:
        vals=[]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            X=rng.standard_normal((A.shape[0],N)); U=1.5*rng.standard_normal((B.shape[1],N))
            D=spec['d_bar'][:,None]+spec['d_std'][:,None]*rng.standard_normal((E.shape[1],N))
            Xp=A@X+B@U+E@D+1e-4*rng.standard_normal((A.shape[0],N))
            data=dict(X=X,U=U,D=D,Xp=Xp,nx=A.shape[0],nu=B.shape[1],nd=E.shape[1])
            est=dd.identify_full_ols(data)
            pe=core.policy_evaluate(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,opt['K'],opt['p'],opt['N'],opt['z'])
            imp=core.policy_improve(est['A_hat'],est['B_hat'],est['E_hat'],Q,R,alpha,lam,V,pe['P'],pe['chi'])
            W=np.vstack([X,U,D,np.ones((1,N))]); c=np.linalg.cond(W@W.T/N)
            vals.append(dict(eA=np.linalg.norm(est['A_hat']-A)/np.linalg.norm(A),eB=np.linalg.norm(est['B_hat']-B)/np.linalg.norm(B),eE=np.linalg.norm(est['E_hat']-E)/np.linalg.norm(E),eP=np.linalg.norm(pe['P']-opt['P'])/np.linalg.norm(opt['P']),eK=np.linalg.norm(imp['K']-opt['K'],'fro'),ep=np.linalg.norm(imp['p']-opt['p']),cond=c))
        per[int(N)]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
        details.append([N,per[int(N)]['cond'],per[int(N)]['eA'],per[int(N)]['eB'],per[int(N)]['eE'],per[int(N)]['eP'],per[int(N)]['eK'],per[int(N)]['ep']])
    slopes={k:dd.loglog_slope(list(per),[per[N][k] for N in per]) for k in ('eA','eB','eE','eP','eK','ep')}
    return dict(per_n=per,slopes=slopes,details=details)

def main():
    rows=[]; detail=[]; rep={}
    for name,spec in (('humanoid',models.humanoid_spec()),('microgrid',models.microgrid_spec())):
        r=run(spec); rep[name]=r; s=r['slopes']
        rows.append([name,f'{s["eA"]:.4f}',f'{s["eB"]:.4f}',f'{s["eE"]:.4f}',f'{s["eP"]:.4f}',f'{s["eK"]:.4f}',f'{s["ep"]:.4f}'])
        for x in r['details']: detail.append([name]+x)
        print(name,s)
    _write('od_e5f_one_step_orthogonal_consistency',['Scenario','slope eA','slope eB','slope eE','slope eP','slope eK','slope ep'],rows,notes='Independent Gaussian PE design, not a closed-loop Algorithm 4 rollout. It isolates identification conditioning for one fixed-policy evaluation and one improvement.')
    _write('od_e5f_one_step_orthogonal_points',['Scenario','N','cond(W^T W)','mean eA','mean eB','mean eE','mean eP','mean eK','mean ep'],detail,notes='Per-N points for the orthogonal/PE diagnostic.')
    with open(os.path.join(LOG,'od_e5f_one_step_orthogonal_consistency.json'),'w',encoding='utf-8') as f: json.dump(rep,f,indent=2,default=float)
    return rep

if __name__=='__main__': main()
