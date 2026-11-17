| scenario | reference | controller | seeds | deterministic steady error norm | MC MAE | MC cost | bias residual | rho_G |
|---|---|---|---|---|---|---|---|---|
| humanoid | [0.2, 0.0, 0.3, 0.0, 0.2, 0.0] | pure LQR | 20 | 0.28774012202481764 | 0.290969867329221 | 370.7626605066165 | 0.0 | 0.9853624421086357 |
| humanoid | [0.2, 0.0, 0.3, 0.0, 0.2, 0.0] | affine LQR bias | 20 | 1.070060299010908e-11 | 0.039043687823951886 | 39.216010073680195 | 0.0 | 0.9853624421086357 |

The affine bias is computed from the nominal model and d_bar. The deterministic error uses zero measurement disturbance and the MC error uses the nominal Gaussian disturbance law.
