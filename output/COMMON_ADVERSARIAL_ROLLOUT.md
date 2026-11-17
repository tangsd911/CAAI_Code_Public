# Common penalized-dual affine adversarial rollout vs nominal Gaussian

- **Nominal** columns: `d = d_bar + d_std * eps`, 20 seeds (seeds 1000..1019), steady window after step `ss`.

- **Common-adversary** columns: the same deterministic affine disturbance `d = N x + z_bar` from the penalized DRC dual is applied to both controllers (single run from x_ref = 0).

- This is **not** LQR's own worst-case disturbance and does **not** verify membership in a hard Wasserstein ball.

- Main-table comparisons use the nominal Gaussian rollout. A divergent auxiliary run is reported verbatim.


## humanoid

| Controller | 口径 | steady MAE | discounted cost | mean ||u||^2 | note |
|---|---|---|---|---|---|
| DRC | nominal Gaussian | 0.04548 | 41.16153 | — | 20-seed mean |
| DRC | common dual d=Nx+z | 0.02453 | 41.17824 | 38.77155 | Δcost=+0.01671, ratio=1.000 |
| LQR | nominal Gaussian | 0.29097 | 370.76266 | — | 20-seed mean |
| LQR | common dual d=Nx+z | 0.29337 | 383.75071 | 39.49681 | Δcost=+12.98805, ratio=1.035 |

## microgrid

| Controller | 口径 | steady MAE | discounted cost | mean ||u||^2 | note |
|---|---|---|---|---|---|
| DRC | nominal Gaussian | 0.00281 | 0.48852 | — | 20-seed mean |
| DRC | common dual d=Nx+z | 0.00080 | 0.36905 | 0.01602 | Δcost=-0.11947, ratio=0.755 |
| LQR | nominal Gaussian | 0.00494 | 0.74544 | — | 20-seed mean |
| LQR | common dual d=Nx+z | — | — | — | **diverged at t=157, max|x|=1.078e+05** |
