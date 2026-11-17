| Scenario | empirical C1 | residual coefficient | max test ratio | violation fraction | max test overrun | calibration points | held-out points |
|---|---|---|---|---|---|---|---|
| humanoid | 1.268738e+02 | 1.000000e+00 | 0.3948 | 0.0000 | -1.260441e+00 | 48 | 48 |
| microgrid | 1.116026e+02 | 1.000000e+00 | 0.4360 | 0.0000 | -1.023492e+00 | 48 | 48 |

The audited inequality is err <= C1/sqrt(N) + ||T_d||*||d_window-vbar_window||. The residual coefficient is one by the triangle inequality and is not tuned. C1 is calibrated on seeds 100-102 and tested on seeds 200-202; this is finite-sample evidence, not a proof of a universal constant.
