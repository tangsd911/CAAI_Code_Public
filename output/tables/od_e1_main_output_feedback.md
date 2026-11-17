| Scenario | Controller | Feedback | MAE | MAE std | Discounted cost | Cost std | Reconstruction RMSE | Observer rho | Max state | Diverged seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| humanoid | DRC | full state | 0.045482 | 0.000894 | 41.161535 | 0.477294 | — | — | — | 0 |
| humanoid | DRC | unknown-history OFB (KF) | 0.121189 | 0.006555 | 58.581330 | 2.233366 | 2.122e-01 | 0.991236 | 2.518e+01 | 0 |
| humanoid | LQR | full state | 0.290970 | 0.000998 | 370.762661 | 2.565953 | — | — | — | 0 |
| humanoid | LQR | unknown-history OFB (KF) | 0.314970 | 0.003509 | 388.859961 | 4.963427 | 2.122e-01 | 0.991236 | 2.493e+01 | 0 |
| microgrid | DRC | full state | 0.002810 | 0.000026 | 0.488517 | 0.005141 | — | — | — | 0 |
| microgrid | DRC | unknown-history OFB (KF) | 0.003176 | 0.000036 | 0.543027 | 0.005514 | 1.806e-03 | 0.797523 | 3.012e+00 | 0 |
| microgrid | LQR | full state | 0.004939 | 0.000030 | 0.745440 | 0.008581 | — | — | — | 0 |
| microgrid | LQR | unknown-history OFB (KF) | 0.005093 | 0.000088 | 0.791925 | 0.023684 | 1.806e-03 | 0.797523 | 1.604e+00 | 0 |

OFB uses y=Cx and a steady-state Kalman observer with no realized disturbance-history input. The full-state row uses the same 20 seeds and x0. OFB is therefore the primary unknown-history closed-loop result; the exact static window requires d history.
