# CAAI Verified Package — Results Summary

- Generated: 2026-11-12T22:10:49
- Convention: 偏差 = 论文值 − 实测值 (per 01_baseline_report.md).
- Every measured number below was computed by this run; the comparison targets are the current v3 manuscript values in `models.PAPER`.

## 1. Consistency tally

- 完全一致条目数: **64**
- 有偏差条目数: **0**
- 无法复现条目数: **0**
- 条目总数: 64

> Rounding note: entries at the 3-decimal reporting boundary are marked 一致 when the deviation is within the MAE rounding tolerance; see the deviation column for exact values. No parameter or metric definition was changed to force a match.

## 2. Measured vs paper

| Entry | Measured | Paper | Tol-key | Status | Deviation |
|---|---|---|---|---|---|
| Humanoid DRC MAE | 0.04548 | 0.0455 | mae | 一致 | +0.00002 |
| Humanoid LQR MAE | 0.29097 | 0.291 | mae | 一致 | +0.00003 |
| Humanoid DRC advantage % | 84.36867 | 84.4 | pct | 一致 | +0.03133 |
| Microgrid DRC cost | 0.48852 | 0.489 | cost | 一致 | +0.00048 |
| Microgrid LQR cost | 0.74544 | 0.745 | cost | 一致 | -0.00044 |
| Microgrid DRC freq MAE | 0.00281 | 0.0028 | mae | 一致 | -0.00001 |
| Microgrid LQR freq MAE | 0.00494 | 0.0049 | mae | 一致 | -0.00004 |
| Microgrid DRC cost advantage % | 34.46598 | 34.5 | pct | 一致 | +0.03402 |
| Shift 0.0 LQR MAE | 0.29097 | 0.291 | mae | 一致 | +0.00003 |
| Shift 0.0 DRC MAE | 0.04548 | 0.0455 | mae | 一致 | +0.00002 |
| Shift 0.0 advantage % | 84.36867 | 84.4 | pct | 一致 | +0.03133 |
| Shift 0.5 LQR MAE | 0.43383 | 0.4338 | mae | 一致 | -0.00003 |
| Shift 0.5 DRC MAE | 0.16888 | 0.1689 | mae | 一致 | +0.00002 |
| Shift 0.5 advantage % | 61.07202 | 61.1 | pct | 一致 | +0.02798 |
| Shift 1.0 LQR MAE | 0.57719 | 0.5772 | mae | 一致 | +0.00001 |
| Shift 1.0 DRC MAE | 0.30936 | 0.3094 | mae | 一致 | +0.00004 |
| Shift 1.0 advantage % | 46.40156 | 46.4 | pct | 一致 | -0.00156 |
| Shift 2.0 LQR MAE | 0.86441 | 0.8644 | mae | 一致 | -0.00001 |
| Shift 2.0 DRC MAE | 0.59392 | 0.5939 | mae | 一致 | -0.00002 |
| Shift 2.0 advantage % | 31.29211 | 31.3 | pct | 一致 | +0.00789 |
| Shift 3.0 LQR MAE | 1.15189 | 1.1519 | mae | 一致 | +0.00001 |
| Shift 3.0 DRC MAE | 0.87941 | 0.8794 | mae | 一致 | -0.00001 |
| Shift 3.0 advantage % | 23.65574 | 23.7 | pct | 一致 | +0.04426 |
| Ablation LQR pure feedback humanoid MAE | 0.29097 | 0.291 | mae | 一致 | +0.00003 |
| Ablation LQR pure feedback micro cost | 0.74544 | 0.745 | cost | 一致 | -0.00044 |
| Ablation LQR pure feedback micro MAE | 0.00494 | 0.0049 | mae | 一致 | -0.00004 |
| Ablation Affine nominal LQR humanoid MAE | 0.03904 | 0.039 | mae | 一致 | -0.00004 |
| Ablation Affine nominal LQR micro cost | 0.74544 | 0.745 | cost | 一致 | -0.00044 |
| Ablation Affine nominal LQR micro MAE | 0.00494 | 0.0049 | mae | 一致 | -0.00004 |
| Ablation DRC no-bias (main: p=0, nominal d) humanoid MAE | 0.28923 | 0.289 | mae | 一致 | -0.00023 |
| Ablation DRC no-bias (main: p=0, nominal d) micro cost | 0.59071 | 0.591 | cost | 一致 | +0.00029 |
| Ablation DRC no-bias (main: p=0, nominal d) micro MAE | 0.00358 | 0.0036 | mae | 一致 | +0.00002 |
| Ablation Proposed affine DRC humanoid MAE | 0.04548 | 0.0455 | mae | 一致 | +0.00002 |
| Ablation Proposed affine DRC micro cost | 0.48852 | 0.489 | cost | 一致 | +0.00048 |
| Ablation Proposed affine DRC micro MAE | 0.00281 | 0.0028 | mae | 一致 | -0.00001 |
| Humanoid lambda=5 MAE | 0.04548 | 0.0455 | mae | 一致 | +0.00002 |
| Humanoid lambda=5 cost | 41.16153 | 41.16 | cost | 一致 | -0.00153 |
| Humanoid lambda=10 MAE | 0.04618 | 0.0462 | mae | 一致 | +0.00002 |
| Humanoid lambda=10 cost | 41.38362 | 41.38 | cost | 一致 | -0.00362 |
| Humanoid lambda=20 MAE | 0.04653 | 0.0465 | mae | 一致 | -0.00003 |
| Humanoid lambda=20 cost | 41.49967 | 41.5 | cost | 一致 | +0.00033 |
| Microgrid lambda=5 MAE | 0.00281 | 0.0028 | mae | 一致 | -0.00001 |
| Microgrid lambda=5 cost | 0.48852 | 0.489 | cost | 一致 | +0.00048 |
| Microgrid lambda=10 MAE | 0.00273 | 0.0027 | mae | 一致 | -0.00003 |
| Microgrid lambda=10 cost | 0.40346 | 0.403 | cost | 一致 | -0.00046 |
| Microgrid lambda=20 MAE | 0.00280 | 0.0028 | mae | 一致 | -0.00000 |
| Microgrid lambda=20 cost | 0.39426 | 0.394 | cost | 一致 | -0.00026 |
| Microgrid lambda=1 stable solution | unstable/non-converged | no stable converged solution | qualitative | 一致(定性) | — |
| Microgrid lambda=2 stable solution | unstable/non-converged | no stable converged solution | qualitative | 一致(定性) | — |
| Mean-only micro DRC cost | 0.58110 | 0.581 | cost | 一致 | -0.00010 |
| Mean-only micro LQR cost | 1.32027 | 1.32 | cost | 一致 | -0.00027 |
| Shifted micro DRC cost | 0.91379 | 0.914 | cost | 一致 | +0.00021 |
| Shifted micro LQR cost | 1.60589 | 1.606 | cost | 一致 | +0.00011 |
| Shifted micro DRC MAE | 0.00352 | 0.0035 | mae | 一致 | -0.00002 |
| Shifted micro LQR MAE | 0.00729 | 0.0073 | mae | 一致 | +0.00001 |
| Shifted micro DRC cost degradation % | 87.05419 | 87.1 | pct | 一致 | +0.04581 |
| Shifted micro LQR cost degradation % | 115.42862 | 115.4 | pct | 一致 | -0.02862 |
| humanoid Riccati iters | 420 | 420 | iters | 一致 | +0.00000 |
| humanoid lambda_min(P) | 31.78478 | 31.78 | lam_minP | 一致 | -0.00478 |
| humanoid rho(G) | 0.98536 | 0.985 | rho | 一致 | -0.00036 |
| microgrid Riccati iters | 76 | 76 | iters | 一致 | +0.00000 |
| microgrid lambda_min(P) | 1.75603 | 1.756 | lam_minP | 一致 | -0.00003 |
| microgrid rho(G) | 0.80212 | 0.802 | rho | 一致 | -0.00012 |
| Humanoid O_3 rank | 6 | 6 | rank | 一致 | +0.00000 |

## 3. Figure summary (measured MC numbers used in bars)

- **humanoid**: {'benchmark': 'model-based full state', 'drc_mae': 0.04548246375667183, 'lqr_mae': 0.29096986732346564}
- **microgrid**: {'benchmark': 'model-based full state', 'drc_cost': 0.488516526767659, 'lqr_cost': 0.7454396376891662, 'drc_mae': [0.0028104163272232173, 0.0035157583341207674], 'lqr_mae': [0.00493899996408309, 0.007285533111764131]}

## 4. Hard-coded scan

- Result literals in print/bar/text/annotate: **0**
- Layout/formatting constants remaining: **8**

## 5. Honest scope notes

- Implemented here: model-based full-state affine DRC fixed-point (g=1 and general-g mean reduction), discounted LQR baseline, both scenarios' nominal/shift/ablation/lambda/mean-only experiments, observability rank checks, theory scalars, and the two computed full-state benchmark figures.
- Added this revision: T1 per-support dual expansion; T2 SDPI two-stage scaling + conservative interpolation; T3 Alg.4 closed-loop/open-loop LS identification scan; T4 Thm.3 X-criterion feasibility check; T5 model-free truth-usage audit. Details in section 6 and archive/legacy_model_v2/BEFORE_AFTER.md.
- Scope limitation: the Section VII experiments are a model-based full-state benchmark, not a closed-loop output-feedback experiment. The static output-window and forward-propagation identities are checked separately in `theory_tests.py`.
- Still NOT implemented: identified-model Wasserstein adversary block E and N; a hard-ball worst-case rollout; instrumental-variable identification for feedback-correlated regressors. The reported affine rollout uses the common penalized-dual adversary and is not LQR's own worst case.

## 6. New mechanism results (T1–T5, this revision)

### 6.1 SDPI (T2) vs naive fixed point

| Scenario | Phase I rounds | Switch step | Phase II steps | rho(G) | Value violations | max ‖ΔK‖/‖Δp‖/‖ΔN‖/‖Δz‖/‖ΔP‖ vs baseline |
|---|---|---|---|---|---|---|
| humanoid | 1 | 0 | 5 | 0.98536 | 2 | 5.81e-14/4.44e-16/6.06e-16/0.00e+00/5.42e-12 |
| microgrid | 0 | 0 | 7 | 0.80212 | 200 | 1.62e-12/6.96e-15/2.60e-13/2.07e-15/2.52e-11 |

Source log: output/logs/e9_sdpi.log

### 6.2 Alg.4 identification scan (T3)

Mean ± std over seeds; slope = log-log slope of mean error vs N (Thm.6 O(N^-1/2) reference = -0.5). eK/ep mix estimation error with the un-modeled adversary block; see archive/legacy_model_v2/BEFORE_AFTER.md.

**humanoid** (exploration scale=0.1, mode=adversarial, seeds=2)

| N | eG | eB(cl) | eA | eB(ol) | eP | eK | ep | rank | λmin Φ |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.756±0.084 | 0.331±0.001 | 0.728±0.183 | 0.567±0.012 | 0.937±0.399 | 72.29±3.49 | 6.337±0.104 | 9/9 | 1.8e-06 |
| 2000 | 0.405±0.082 | 0.119±0.002 | 0.484±0.069 | 0.499±0.000 | 0.500±0.121 | 70.74±3.45 | 6.333±0.070 | 9/9 | 2.4e-06 |
| 5000 | 0.293±0.016 | 0.122±0.005 | 0.339±0.019 | 0.492±0.000 | 0.359±0.023 | 71.88±2.02 | 6.196±0.066 | 9/9 | 3.6e-06 |

Log-log slopes: eG=-0.41, eB_cl=-0.46, eA=-0.33, eB_ol=-0.06, eP=-0.42, eK=-0.00, ep=-0.01

**microgrid** (exploration scale=0.1, mode=adversarial, seeds=2)

| N | eG | eB(cl) | eA | eB(ol) | eP | eK | ep | rank | λmin Φ |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.550±0.007 | 0.061±0.046 | 0.187±0.044 | 0.009±0.000 | 0.406±0.129 | 281.74±12.68 | 0.034±0.031 | 6/6 | 2.3e-06 |
| 2000 | 0.559±0.010 | 0.039±0.010 | 0.164±0.019 | 0.009±0.001 | 0.247±0.027 | 274.06±7.56 | 0.020±0.000 | 6/6 | 2.3e-06 |
| 5000 | 0.560±0.009 | 0.019±0.006 | 0.173±0.008 | 0.009±0.000 | 0.254±0.002 | 278.68±3.50 | 0.011±0.005 | 6/6 | 2.4e-06 |

Log-log slopes: eG=0.01, eB_cl=-0.49, eA=-0.04, eB_ol=0.01, eP=-0.22, eK=-0.01, ep=-0.49

Source log: output/logs/e10_alg4.log

### 6.3 Thm.3 X criterion (T4)

| Scenario | rho(G*) | Construction | Original-form SDP | Lyapunov SDP (stable) | Destabilized rho | Lyapunov SDP (bad) |
|---|---|---|---|---|---|---|
| humanoid | 0.9854 | feasible | optimal | optimal | 1.0200 | SCS-marginal |
| microgrid | 0.8021 | feasible | optimal | optimal | 1.0200 | SCS-marginal |

Source log: output/logs/e11_thm3.log

### 6.4 Model-free truth audit (T5)

- Truth A/B/E referenced only in: ['collect_closed_loop', 'identification_errors']
- Compliant: **True**

Source log: output/logs/e12_truth_audit.log


## 7. Current symmetric microgrid model and certificates

- The primary tables use the standard symmetric two-area LFC model. The legacy-vs-current audit is kept separately in `archive/legacy_model_v2/model_v2_numbers.md` and is not part of the primary tally.

- **humanoid**: λ_min(Φ_P)=26.7586 (Φ_P≻0: True); λ_min(P−P0)=0.0000 (P⪰P0: True); λ_min(P0−P)=-0.6944; αρ(G)²=0.9224<1.
- **microgrid**: λ_min(Φ_P)=1.5666 (Φ_P≻0: True); λ_min(P−P0)=0.0040 (P⪰P0: True); λ_min(P0−P)=-2770.0648; αρ(G)²=0.6112<1.

## 8. Common penalized-dual affine adversarial rollout

- Detailed table: `output/COMMON_ADVERSARIAL_ROLLOUT.md`. The main tables use nominal Gaussian disturbances. The auxiliary rollout applies the same affine disturbance `d=Nx+z_bar` from the DRC dual to both controllers. It is **not** LQR's own worst case and does **not** verify hard Wasserstein-ball membership.

- **humanoid** nominal DRC cost=41.1615 / LQR cost=370.7627; adversarial DRC: cost=41.1782, MAE=0.02453; adversarial LQR: cost=383.7507, MAE=0.29337.
- **microgrid** nominal DRC cost=0.4885 / LQR cost=0.7454; adversarial DRC: cost=0.3690, MAE=0.00080; adversarial LQR: diverged: diverged at t=157, max|x|=1.078e+05.
