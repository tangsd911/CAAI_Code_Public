# Algorithm 4 / IV / E,N Identification / SDPI Verification Report

Working directory: `CAAI_Code_Final_v21`

## 1. Implemented pipeline

The executable Algorithm 4 is organized into the following sequence.

1. Collect an independent pilot sample and estimate the disturbance support `V`.
2. Collect a fresh closed-loop batch under the current `(K,p,N,z)` with input exploration.
3. Estimate the closed-loop map `G`, input matrix `B`, and affine intercept `F`; with the independent probe, separate `E,N,z` by IV/2SLS.
4. Estimate the affine adversary block in `d-q=Nx+z+xi`, obtaining `N_hat,z_hat`.
5. Run model-based SDPI on the identified model.
6. Compute `rho(A+BK+EN)` on the true model only after the update as a post-hoc diagnostic.
7. Stop only after recording all four policy deltas and the exact stopping reason.

The true `A,B,E` are not used by the identified-model SDPI. They are used only for
data generation and post-hoc scoring. The true spectral radius does not gate or reject
an Algorithm 4 update.

## 2. IV design and identifiability

The structural equation is

```text
x_{k+1} = A x_k + B u_k + E d_k + c + error.
```

Under an affine adversary and closed-loop feedback,

```text
d_k = N x_k + z + q_k + xi_k.
```

Without an exogenous disturbance-channel experiment (`q_k=0`), `d_k` is
feedback-correlated with `x_k`, while `xi_k` also enters the transition error.
Therefore `E` is confounded with `A+EN`; a full-rank regressor matrix does not
remove this endogeneity. This case is retained as the explicit negative result
`od_e10_identifiability_negative`.

For the positive case, an independent disturbance probe `q_k` is generated and
used as an excluded instrument. The 2SLS design is

```text
W = [x_k, u_k, d_k, 1]
Z = [x_k, u_k, x_{k-1}, u_{k-1}, q_k, q_{k-1}, 1].
```

The first stage is `W_hat = P_Z W`, followed by structural least squares on
`W_hat`. All least-squares stages are row-scaled and transformed back to the
original coefficient coordinates. The following diagnostics are stored:

- rank of `W` and `Z`;
- excluded-instrument partial first-stage `F`;
- first-stage minimum eigenvalue;
- over-identification Sargan/Hansen-type statistic and minimum p-value;
- relative errors `eA,eB,eE,eN,ez`;
- explicit status `identifiable`, `rank_deficient`, `weak_instrument`, or
  `not_identifiable`.

## 3. Scaling back-map and SDPI verification

The scaled model uses

```text
A_beta = beta A,  B_beta = beta B,  E_beta = beta E,  V_beta = beta V.
```

If `(K_s,p_s,N_s,z_s)` is the policy returned on this scaled model, the
coordinate map back to the original system is

```text
K_orig = K_s,  N_orig = N_s,  p_orig = p_s,  z_orig = z_s.
```

This is implemented in `sdpi.scaled_policy_to_original`. Acceptance is never
based on the scaled radius alone. The code explicitly checks

```text
G_orig = A + B K_orig + E N_orig,
rho(G_orig) < 1,
rho(beta G_orig) < 1.
```

If the copied candidate is not stable, the code tries another scaling round and
finally uses a directly verified discounted-LQR fallback. Every attempt is
recorded.

The improvement step checks `H_uu > 0`, `H_mm < 0`, both Schur complements,
their condition numbers, and their definiteness before solving. A singular or
boundary Schur complement stops the iteration with
`schur_complement_boundary`/`schur_complement_singular`; no pseudo-inverse is
used. The stop condition tests all four blocks `K,p,N,z`.

## 4. Reproduction commands

```bash
cd CAAI_Code_Final
PYTHONDONTWRITEBYTECODE=1 python3 final_alg4_experiments.py
PYTHONDONTWRITEBYTECODE=1 python3 theory_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 algorithm4_tests.py
PYTHONPYCACHEPREFIX=/tmp/caai_pycache python3 -m py_compile \
  data_driven.py algorithm4.py sdpi.py final_alg4_experiments.py
```

The full experiment driver took 52.4 s on the current machine. It writes CSV and
Markdown tables under `output/tables/`; no result value is read back as an
algorithm input.

## 5. Main end-to-end results

`od_e4_algorithm4_end_to_end.md`: `N=12000`, five seeds, two fresh-data rounds,
input exploration, disturbance probe, IV/2SLS.

| Scenario | Method | eA | eB | eE | eN policy | true rho min | post-hoc stable |
|---|---|---:|---:|---:|---:|---:|---:|
| Humanoid | IV/2SLS | `6.315e-4` | `1.079e-4` | `3.851e-4` | `1.323e-2` | `0.985659` | 5/5 |
| Microgrid | IV/2SLS | `1.342e-3` | `1.854e-5` | `6.821e-3` | `1.361e-1` | `0.802267` | 5/5 |

The E-equation first-stage partial `F` was `485.62 +/- 11.70` for humanoid and
`361.87 +/- 11.42` for microgrid. The N-equation first-stage partial `F` was
`323.93 +/- 11.69` and `2361.37 +/- 36.08`, respectively. The direct relative
`N` errors were large for humanoid (`9.478`, because the reference `N` has small
norm and its regression signal is weak relative to disturbance innovation), and
`2.154e-2` for microgrid. The policy-space `N` error after SDPI is the more
relevant end-to-end quantity and is reported separately.

All 10 end-to-end runs (five seeds in each scenario) were updated without a true-model
gate; their post-hoc true-model radii were below one in all runs. Both fresh-data rounds completed in every run with SDPI stop reason
`stationarity`; the outer status is `round_limit` because the configured two
rounds ended before the outer stopping tolerance. Full histories are retained in
`output/logs/final_alg4_experiments.json`.

The focused test suite also verifies exact recovery in the noiseless designed
experiment: maximum matrix errors were `1.2e-12` for humanoid and `2.6e-13` for
microgrid, including `N,z`. The scaling-map tests reached verified original
radii `0.999840` and `0.982146` from deliberately non-stabilizing initial gains.

## 6. One-step `O(N^-1/2)` rates

`od_e5_algorithm4_rates.md`: `N=4000,8000,16000,32000`, five seeds, one
fixed-policy evaluation/improvement round. This is the local statistical
statement proved in the manuscript, not a global outer-loop rate.

| Scenario | Method | eA | eB | eE | eP | eK | ep | eN policy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Humanoid | IV/2SLS | `-0.644` | `-0.495` | `-0.511` | `-0.544` | `-0.538` | `-0.397` | `-0.501` |
| Microgrid | IV/2SLS | `-0.507` | `-0.535` | `-0.490` | `-0.481` | `-0.498` | `-0.217` | `-0.522` |

Most identification and policy slopes are close to `-1/2`, but the affine-bias
slope is weaker (`-0.397` humanoid and `-0.217` microgrid). This table is therefore
a conditioning diagnostic, not a general empirical validation of Theorem 11. The
low-noise PE study in `od_e5g_rigorous_rate_validation.md` is the rate validation
reported in the paper.

`od_e5b_adversary_block_rates.md` isolates `N,z` under a fixed model-based DRC
policy. The IV slopes were `-0.407` for humanoid `eN` and `-0.345` for microgrid
`eN`. They are not yet at `-1/2`; the microgrid block has weak directions and a
finite-sample plateau. This is a real limitation, not a reason to relabel the
slope.

## 7. SDPI numerical checks

`od_e9_sdpi_diagnostics.md` records the identified-model SDPI run.

| Scenario | stop | max delta | cond S_u | cond S_m | residual P | residual chi | residual psi |
|---|---|---:|---:|---:|---:|---:|---:|
| Humanoid | stationarity | `6.883e-13` | `1.770` | `1.003` | `4.307e-13` | `4.965e-16` | `0` |
| Microgrid | stationarity | `3.754e-12` | `1.424` | `1.367` | `2.575e-12` | `8.909e-16` | `1.110e-16` |

The deliberate `lambda=0.01` case has indefinite `H_mm`; both scenarios stopped
at `schur_complement_boundary` instead of using a pseudo-inverse. This verifies
the boundary behavior, not a controller result at that `lambda`.

## 8. Negative identifiability result

`od_e10_identifiability_negative.md` uses `q=0`. In all five seeds and both
scenarios, `E` was reported as non-identifiable even though the raw regressor
rank was full. This is the correct result: rank is not exogeneity. A positive
full `(E,N)` experiment requires either exogenous disturbance-channel excitation
or valid prior knowledge.

## 9. Remaining limitations

1. The `N` block rate is not yet uniformly `O(N^-1/2)` in the reported finite
   samples; microgrid is especially limited by weak directions.
2. The end-to-end outer loop has post-hoc stability diagnostics and empirical errors
   but no uniform global-rate proof.
3. The positive IV experiment requires an independent disturbance probe. This is
   an experimental-design requirement, not available for free in every field
   deployment.
4. The true-model spectral radius is a post-hoc diagnostic only. It is neither used
   to synthesize the policy nor used to accept or reject an update.
5. The SDPI conclusion remains conditional on nonsingular/well-conditioned Schur
   complements and bounded stabilizing iterates; no global convergence or
   per-step cost monotonicity is claimed.
