# Unified nonlinear validation model card

This card documents the model used by `nonlinear_validation.py` for `od_e8`
and `od_e9`. It is a **reduced nonlinear validation model**, not a high-fidelity
humanoid or utility-grid simulator. The controller and observer are fixed at
the nominal linear-model designs; they are not retuned at the stress levels.

## Humanoid reduced three-joint leg

For joint `i=1,2,3`, state `x_i=(theta_i, omega_i)`, torque `tau_i`, external
torque disturbance `d_i`, and `dt=0.01 s`:

```text
dot theta_i = omega_i
dot omega_i = ( tau_i + d_i - b_i omega_i - g_i sin(theta_i)
                - f_c,i tanh(omega_i/epsilon_f,i) - tau_contact,i ) / J_i

tau_contact,i =
  -k_c,i (theta_i-theta_max,i) - c_c,i omega_i,   theta_i > theta_max,i
  -k_c,i (theta_i+theta_max,i) - c_c,i omega_i,   theta_i < -theta_max,i
   0,                                             otherwise

tau_i = sat( u_i, [-tau_bound,i, tau_bound,i] )
x_{k+1} = x_k + dt * f(x_k,u_k,d_k)
```

| Quantity | Value | Unit | Source/scope |
|---|---:|---|---|
| `J` | `[0.5, 0.3, 0.2]` | kg m^2 | reduced manuscript three-joint example |
| `b` | `[0.8, 0.5, 0.3]` | N m s/rad | reduced manuscript example |
| gravity torque coefficient `g` | `[10, 8, 6]` | N m | reduced manuscript example |
| joint limits `theta_max` | `[0.45, 0.65, 0.75]` | rad | stress-model joint ranges |
| contact stiffness `k_c` | `[200, 180, 160]` | N m/rad | compliant-limit stress parameters |
| contact damping `c_c` | `[5, 4, 3]` | N m s/rad | compliant-limit stress parameters |
| Coulomb friction `f_c` | `[0.3, 0.25, 0.2]` | N m | smooth-friction stress parameters |
| friction velocity scale `epsilon_f` | `0.02` | rad/s | smooth approximation parameter |
| torque bound `tau_bound` | `5.973608` per joint | N m | reduced-model symmetric actuator bound |
| initial state | `[0.30,0,0.30,0,0.30,0]` times initial-state factor | rad, rad/s | manuscript initial condition |

The torque bound is applied to the commanded torque; the external disturbance
is added after the actuator saturation. `Contact fraction` counts steps with at
least one joint outside its range; `Contact joint-step fraction` divides by all
three joint-steps; `Friction active fraction` counts joint-steps with
`|omega|>0.02 rad/s`.

## Microgrid nonlinear two-area LFC

The linear matrices are those in `models.microgrid_continuous()`:
`x=[df1,dPm1,df2,dPm2]`, `u=[dPc1,dPc2]`,
`d=[dPN1,dPN2,dPW1,dPW2]`, `Ts=0.1 s`.

```text
dot f_i = [Ac x + Bc u + Ec d]_frequency,i
          - (D_nl,i/(2 H_i)) f_i |f_i|
dot Pm_i = [Ac x + Bc u + Ec d]_Pm,i

u_i = sat( command_i, [-u_bound,i, u_bound,i] )
x_{k+1} = x_k + Ts * [Ac x_k + Bc u_k + Ec d_k + nonlinear(x_k)]
```

| Quantity | Value | Unit | Source/scope |
|---|---:|---|---|
| `H` | `[5.0, 4.0]` | s | standard two-area LFC model |
| `D` | `[0.8, 0.6]` | pu/Hz | standard two-area LFC model |
| `Tg` | `[0.5, 0.4]` | s | standard two-area LFC model |
| `T12` | `0.2` | pu/Hz | standard two-area LFC model |
| `D_nl` | `[0.20, 0.20]` | pu/Hz^2 | quadratic-damping stress parameter |
| command bound | `0.420593` per area | pu | reduced-model governor saturation |
| frequency range | `[-0.5, 0.5]` | Hz | operating-range diagnostic |
| initial state | `[0.1,0,0.1,0]` times initial-state factor | Hz, pu | manuscript initial condition |

`Frequency violation fraction` counts steps with either frequency outside the
range; `Max frequency excess` is the maximum positive excursion beyond 0.5 Hz,
and is zero when the range is respected.

## Observer, disturbances, and stress levels

The observer is a nominal-model steady-state Kalman filter:

```text
xhat_{k+1} = A xhat_k + B u_k + E d_bar + L (y_k - C xhat_k)
y_k = C x_k + measurement noise
```

The realized disturbance path is not passed to the observer. Process covariance
is `E diag(d_std^2) E' + 1e-10 I`; measurement-noise standard deviation is
`1e-8`. The same seed and disturbance law are used for DRC and LQR.

Stress tuples are recorded in `output/logs/od_e8_nonlinear_manifest.json`:

- humanoid: `(disturbance scale, initial-state scale, torque-bound factor,
  friction scale, contact-stiffness scale)`:
  nominal `(1.0,1.0,1.0,1.0,1.0)`, moderate `(1.5,1.15,0.75,1.25,1.25)`,
  extreme `(2.5,1.35,0.50,1.75,1.50)`.
- microgrid: `(disturbance scale, initial-state scale, command-bound factor,
  nonlinear-damping scale)`:
  nominal `(1.0,1.0,1.0,1.0)`, moderate `(1.5,1.25,0.75,1.5)`,
  extreme `(2.0,1.50,0.50,2.0)`.

The disturbance scales multiply both the mean and standard deviation. The
default horizons and seeds are the manuscript horizons and `1000...1019`.

## Reproduce

```bash
cd CAAI_Code_Final
PYTHONDONTWRITEBYTECODE=1 python3 nonlinear_validation.py
```

Generated artifacts:

- `output/tables/od_e8_nonlinear_contact_saturation.{md,csv}`
- `output/tables/od_e9_tail_only.{md,csv}`
- `output/od_e8_nonlinear_contact_saturation.{md,csv}`
- `output/od_e9_tail_only.{md,csv}`
- `output/logs/od_e8_nonlinear_manifest.json`
- `output/logs/od_e9_tail_only_manifest.json`

## Limitations

- The humanoid model is a three-joint reduced leg surrogate; it does not model
  full-body contact geometry, actuator rate dynamics, friction identification,
  or hardware-level uncertainty.
- The microgrid model is a reduced two-area LFC model; it does not model
  voltage dynamics, converter limits, network topology, or protection.
- The results are stress-test evidence under the declared model and seeds, not
  a proof of global nonlinear stability and not a hard-Wasserstein-ball
  certificate.
