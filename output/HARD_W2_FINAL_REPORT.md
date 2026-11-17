# Final hard-Wasserstein and g>1 simulation report

## Scope

- Hard-ball feasibility is checked on the explicit atomwise adversary distribution: `W2_max <= zeta` at every sampled trajectory point.
- `selected lambda` minimizes the sampled finite-horizon Lagrangian among ball-feasible candidates on the outer lambda grid.
- `boundary lambda` is the refined infeasible-to-feasible transition; its complementarity product is reported separately.
- These are finite-horizon sampled-trajectory certificates. They do not prove global Bellman optimality or continuous-lambda global optimality.

## Hard-ball zeta scan

- Rows: 24; feasible: 24; infeasible: 0.
- humanoid / DRC: 6 radii (0.00500 to 0.20000).
- humanoid / LQR: 6 radii (0.00500 to 0.20000).
- microgrid / DRC: 6 radii (0.00500 to 0.20000).
- microgrid / LQR: 6 radii (0.00500 to 0.20000).
- humanoid / DRC: candidate W2_max/zeta ranges from 0.000077 to 136.986980.
- humanoid / LQR: candidate W2_max/zeta ranges from 0.000131 to 257.219800.
- microgrid / DRC: candidate W2_max/zeta ranges from 0.000039 to 111.947960.
- microgrid / LQR: candidate W2_max/zeta ranges from 0.000057 to inf.

Unified table: `output/tables/od_e6_hard_wasserstein_own_adversary.csv`.
Candidate audit: `output/tables/od_e6_lambda_candidate_audit.csv`.

## General-g membership

- Summary rows: 20; feasible summary rows: 20.
- Individual support-draw rows: 200; infeasible draws: 0.
- humanoid / g=1 / DRC: feasible rate=1.00, mean selected lambda=15.000000, worst W2 max=2.061677e-02, boundary active rate=1.00.
- humanoid / g=1 / LQR: feasible rate=1.00, mean selected lambda=35.000000, worst W2 max=1.499151e-02, boundary active rate=1.00.
- humanoid / g=2 / DRC: feasible rate=1.00, mean selected lambda=13.500000, worst W2 max=2.651266e-02, boundary active rate=1.00.
- humanoid / g=2 / LQR: feasible rate=1.00, mean selected lambda=35.000000, worst W2 max=1.842057e-02, boundary active rate=1.00.
- humanoid / g=5 / DRC: feasible rate=1.00, mean selected lambda=14.500000, worst W2 max=2.723738e-02, boundary active rate=1.00.
- humanoid / g=5 / LQR: feasible rate=1.00, mean selected lambda=35.000000, worst W2 max=1.585696e-02, boundary active rate=1.00.
- humanoid / g=10 / DRC: feasible rate=1.00, mean selected lambda=15.000000, worst W2 max=2.234469e-02, boundary active rate=1.00.
- humanoid / g=10 / LQR: feasible rate=1.00, mean selected lambda=35.000000, worst W2 max=1.593665e-02, boundary active rate=1.00.
- humanoid / g=20 / DRC: feasible rate=1.00, mean selected lambda=15.000000, worst W2 max=2.109179e-02, boundary active rate=1.00.
- humanoid / g=20 / LQR: feasible rate=1.00, mean selected lambda=35.000000, worst W2 max=1.518745e-02, boundary active rate=1.00.
- microgrid / g=1 / DRC: feasible rate=1.00, mean selected lambda=7.500000, worst W2 max=3.301010e-02, boundary active rate=0.00.
- microgrid / g=1 / LQR: feasible rate=1.00, mean selected lambda=20.000000, worst W2 max=1.475352e-02, boundary active rate=1.00.
- microgrid / g=2 / DRC: feasible rate=1.00, mean selected lambda=12.750000, worst W2 max=4.640535e-02, boundary active rate=1.00.
- microgrid / g=2 / LQR: feasible rate=1.00, mean selected lambda=31.000000, worst W2 max=1.924484e-02, boundary active rate=1.00.
- microgrid / g=5 / DRC: feasible rate=1.00, mean selected lambda=8.750000, worst W2 max=3.733711e-02, boundary active rate=0.70.
- microgrid / g=5 / LQR: feasible rate=1.00, mean selected lambda=25.500000, worst W2 max=1.885085e-02, boundary active rate=1.00.
- microgrid / g=10 / DRC: feasible rate=1.00, mean selected lambda=9.750000, worst W2 max=3.249984e-02, boundary active rate=0.60.
- microgrid / g=10 / LQR: feasible rate=1.00, mean selected lambda=23.000000, worst W2 max=2.180937e-02, boundary active rate=1.00.
- microgrid / g=20 / DRC: feasible rate=1.00, mean selected lambda=8.500000, worst W2 max=4.586845e-02, boundary active rate=0.90.
- microgrid / g=20 / LQR: feasible rate=1.00, mean selected lambda=23.000000, worst W2 max=1.967932e-02, boundary active rate=1.00.

Unified table: `output/tables/od_e7_g_w2_membership.csv`.
Per-draw audit: `output/tables/od_e7_g_w2_support_draws.csv`.

## Failure counts

- W2_exceeds_zeta: 144.
- closed_loop_unstable: 18.
- fixed_point_not_converged: 6.

## Commands

```bash
cd /tmp/caai_final_w2
PYTHONDONTWRITEBYTECODE=1 python3 theory_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 -c 'import agent_od_experiments as A; A.exp_hard_wasserstein()'
PYTHONDONTWRITEBYTECODE=1 python3 -c 'import agent_od_experiments as A; A.exp_g_w2_membership()'
PYTHONDONTWRITEBYTECODE=1 python3 final_w2_report.py
```
