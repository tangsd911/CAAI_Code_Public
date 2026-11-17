| Controller | Humanoid MAE (meas) | Micro cost (meas) | Micro MAE (meas) | Hum MAE (paper) | Mic cost (paper) | Mic MAE (paper) | dev hum | dev cost | dev MAE |
|---|---|---|---|---|---|---|---|---|---|
| LQR (pure feedback) | 0.2910 | 0.745 | 0.0049 | 0.291 | 0.745 | 0.0049 | +0.0000 | -0.000 | -0.0000 |
| Affine nominal LQR | 0.0390 | 0.745 | 0.0049 | 0.039 | 0.745 | 0.0049 | -0.0000 | -0.000 | -0.0000 |
| DRC without bias (main: DRC gain, p=0, nominal d) | 0.2892 | 0.591 | 0.0036 | 0.289 | 0.591 | 0.0036 | -0.0002 | +0.000 | +0.0000 |
| 附注 (a): p=0, adversarial d=Nx+z | 0.2916 | 10.509 | 0.0177 | — | — | — | — | — | — |
| 附注 (b): p=0, z=0, adversarial d=Nx | 0.0000 | 0.000 | 0.0000 | — | — | — | — | — | — |
| Proposed affine DRC | 0.0455 | 0.489 | 0.0028 | 0.0455 | 0.489 | 0.0028 | +0.0000 | +0.000 | -0.0000 |

Affine nominal LQR bias: p=pinv(B)((I-A)x_ref-E dbar). Main no-bias口径: DRC gain retained, p=0, nominal Gaussian rollout; this reproduces the paper row 0.289 / 0.591 / 0.0036. 附注 (a) keeps the adversarial offset z in the rollout disturbance; 附注 (b) sets z=0 and reduces to a homogeneous closed loop (degenerate zero forcing).
