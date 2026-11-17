# od_e8 / od_e9 computed summary

All values below were generated from the 20-seed rollouts in the two manifests; this summary contains no manually entered result values.

## Nonlinear stress results

| Scenario | Stress | DRC MAE / cost | LQR MAE / cost | DRC/LQR cost ratio | DRC/LQR MAE ratio | Diverged DRC/LQR |
|---|---|---:|---:|---:|---:|---:|
| humanoid | nominal | 0.316901 / 191.357729 | 0.416696 / 436.462854 | 0.4384 | 0.7605 | 0/0 |
| humanoid | moderate | 0.487660 / 820.311316 | 0.571464 / 1149.860971 | 0.7134 | 0.8534 | 0/0 |
| humanoid | extreme | 0.882560 / 3011.977274 | 0.919440 / 3264.384978 | 0.9227 | 0.9599 | 0/0 |
| microgrid | nominal | 0.003176 / 0.542364 | 0.005092 / 0.791836 | 0.6849 | 0.6237 | 0/0 |
| microgrid | moderate | 0.004461 / 1.085545 | 0.007925 / 1.875447 | 0.5788 | 0.5628 | 0/0 |
| microgrid | extreme | 0.007529 / 2.219027 | 0.011511 / 3.773923 | 0.5880 | 0.6541 | 0/0 |

## Tail-only results

| Scenario | Distribution | DRC MAE / cost | LQR MAE / cost | DRC/LQR cost ratio | Tail event fraction |
|---|---|---:|---:|---:|---:|
| humanoid | gaussian | 0.316901 / 191.357729 | 0.416696 / 436.462854 | 0.4384 | 0.02930 |
| humanoid | student_t_3 | 0.313031 / 186.084143 | 0.412126 / 424.250360 | 0.4386 | 0.05230 |
| humanoid | student_t_5 | 0.313362 / 185.136242 | 0.412207 / 422.585395 | 0.4381 | 0.05205 |
| humanoid | student_t_10 | 0.310847 / 181.843323 | 0.409938 / 417.544043 | 0.4355 | 0.04200 |
| humanoid | tail_contamination | 0.314135 / 188.901615 | 0.412970 / 426.464135 | 0.4429 | 0.05097 |
| microgrid | gaussian | 0.003176 / 0.542364 | 0.005092 / 0.791836 | 0.6849 | 0.06112 |
| microgrid | student_t_3 | 0.003057 / 0.552793 | 0.005120 / 0.822830 | 0.6718 | 0.07457 |
| microgrid | student_t_5 | 0.003142 / 0.540278 | 0.005130 / 0.801197 | 0.6743 | 0.08358 |
| microgrid | student_t_10 | 0.003160 / 0.540076 | 0.005112 / 0.795830 | 0.6786 | 0.07578 |
| microgrid | tail_contamination | 0.002799 / 0.518256 | 0.005082 / 0.799784 | 0.6480 | 0.04979 |

## Data-supported conclusions

- In every reported nonlinear stress level, DRC has lower MAE and lower discounted cost than LQR under the same seeds and disturbances.
- Microgrid frequency-range violation is zero at all three levels, while humanoid joint contact becomes nearly continuous at the extreme level and actuator saturation is very frequent at nominal and above.
- DRC is lower-cost than LQR for every reported Gaussian, Student-t, and tail-contamination law.
- No seed diverged in either table, but the humanoid extreme case violates contact/range heavily; stability within this finite horizon is not a hardware-safety claim.

## Limitations

- The humanoid is a reduced three-joint leg model, not a full-body robot simulator or identified hardware model.
- The microgrid is a reduced two-area LFC model, not a voltage/network/protection model.
- The experiments are finite-horizon stress tests; they do not prove global nonlinear stability or hard-Wasserstein-ball membership.
- Tail laws are alternative disturbance stress laws, not hard-ball worst-case distributions.
