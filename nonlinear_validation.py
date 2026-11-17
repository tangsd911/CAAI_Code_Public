#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified nonlinear closed-loop validation and tail-stress experiments.

This module is deliberately a *reduced* nonlinear validation model, not a
high-fidelity whole-body humanoid or power-system simulator.  It replaces the
older fragmented saturation patch with one explicit closed-loop model:

* the controller is synthesized on the nominal linear model used in the paper;
* the plant is nonlinear and is driven by the same random disturbance path for
  both DRC and LQR;
* the feedback uses a steady-state Kalman output observer driven by ``y=Cx``;
* the applied control is explicitly saturated and the stage cost uses that
  saturated signal;
* every reported number is computed from the rollouts in this file.

Humanoid reduced model (three planar joints)
--------------------------------------------
For joint i, with state (theta_i, omega_i), input torque tau_i, external
disturbance torque d_i, and time step dt=0.01 s:

    dot theta_i = omega_i
    dot omega_i = ( tau_i + d_i - b_i omega_i
                    - g_i sin(theta_i)
                    - f_c,i tanh(omega_i / epsilon_f,i)
                    - tau_contact,i ) / J_i

    tau_contact,i = -k_c,i (theta_i-theta_max,i) - c_c,i omega_i
                    if theta_i > theta_max,i
                  = -k_c,i (theta_i+theta_max,i) - c_c,i omega_i
                    if theta_i < -theta_max,i
                  = 0 otherwise

The commanded torque is clipped to a symmetric torque bound.  The contact is
a compliant spring-damper joint limit, not a rigid unilateral constraint.
The parameters are reduced-model values, not identified hardware parameters:
J=[0.5,0.3,0.2] kg m^2, b=[0.8,0.5,0.3] N m s/rad, g=[10,8,6] N m,
limits=[0.45,0.65,0.75] rad, k_c=[200,180,160] N m/rad,
c_c=[5,4,3] N m s/rad, and f_c=[0.3,0.25,0.2] N m.

Microgrid reduced model (standard two-area LFC)
-----------------------------------------------
The linear part is exactly ``models.microgrid_continuous()`` with H, D, Tg,
T12 from the manuscript's reduced two-area model.  State x=[df1,dPm1,df2,dPm2],
input u=[dPc1,dPc2], disturbance d=[dPN1,dPN2,dPW1,dPW2], and Ts=0.1 s.  The
validated nonlinear plant is

    dot f_i = [A_c x + B_c u + E_c d]_frequency,i
              - (D_nl,i/(2 H_i)) f_i |f_i|
    dot P_m,i = [A_c x + B_c u + E_c d]_Pm,i

with nominal D_nl=[0.20,0.20] pu/Hz^2.  The governor command is clipped to a
symmetric per-area bound; the frequency operating range is |f_i| <= 0.5 Hz.

Observer and policy
-------------------
The observer is the same nominal steady-state Kalman filter used elsewhere in
the package:

    xhat_{k+1} = A xhat_k + B u_k + E d_bar
                 + L (y_k - C xhat_k),   y_k = C x_k + noise.

It is not given the realized disturbance history.  DRC uses the affine policy
``u=K(xhat-x_ref)+p`` from the fixed-lambda solver.  LQR uses
``u=-K_lqr(xhat-x_ref)``.  The observer covariance and policy are not retuned
for the nonlinear stress levels.

Stress design
-------------
The three levels are design parameters, recorded in the generated manifest:

humanoid: nominal (1.0,1.0,1.0,1.0,1.0), moderate
(1.5,1.15,0.75,1.25,1.25), extreme (2.5,1.35,0.50,1.75,1.50), where the tuple
is (disturbance scale, initial-state scale, torque-bound factor, friction
scale, contact-stiffness scale).

microgrid: nominal (1.0,1.0,1.0,1.0), moderate
(1.5,1.25,0.75,1.5), extreme (2.0,1.50,0.50,2.0), where the tuple is
(disturbance scale, initial-state scale, command-bound factor, nonlinear
damping scale).

The default horizon is the paper horizon (humanoid T=3000, ss=1500;
microgrid T=40000, ss=30000) and the default seed set is
``range(1000,1020)``.  The tail experiment uses the nominal stress level.

Outputs
-------
``output/tables/od_e8_nonlinear_contact_saturation.md`` and ``.csv``
``output/tables/od_e9_tail_only.md`` and ``.csv``
plus copies in ``output/`` and JSON logs/manifests in ``output/logs/``.

The ``od_e9`` table compares the nominal Gaussian law, covariance-matched
Student-t laws, and a covariance-matched tail-contamination mixture.  The
mixture is not a claim that the central body is unchanged: with probability
1-eps a Gaussian with a reduced scale is used, and with probability eps a
Gaussian with a larger scale is used; scales are chosen so that the marginal
covariance is exactly the nominal one.  This isolates tail contamination while
recording the central variance compromise.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

import drc_core as core
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")
LOG = os.path.join(OUT, "logs")
for _d in (OUT, TBL, LOG):
    os.makedirs(_d, exist_ok=True)

# Model-design constants.  These are parameters, not result values.
HUMANOID_DT = 0.01
HUMANOID_J = np.array([0.5, 0.3, 0.2], dtype=float)
HUMANOID_B = np.array([0.8, 0.5, 0.3], dtype=float)
HUMANOID_GRAVITY = np.array([10.0, 8.0, 6.0], dtype=float)
HUMANOID_LIMITS = np.array([0.45, 0.65, 0.75], dtype=float)
HUMANOID_CONTACT_K = np.array([200.0, 180.0, 160.0], dtype=float)
HUMANOID_CONTACT_C = np.array([5.0, 4.0, 3.0], dtype=float)
HUMANOID_FRICTION_C = np.array([0.30, 0.25, 0.20], dtype=float)
HUMANOID_FRICTION_EPS = 0.02
# Symmetric reduced-model torque bound used by the paper's stress example.
HUMANOID_TAU_BOUND = np.array([5.973608, 5.973608, 5.973608], dtype=float)

MICROGRID_DT = 0.1
MICROGRID_NL_DAMPING = np.array([0.20, 0.20], dtype=float)
MICROGRID_FREQ_LIMIT = 0.5
# Symmetric governor-command bound used by the paper's stress example.
MICROGRID_U_BOUND = np.array([0.420593, 0.420593], dtype=float)

DEFAULT_SEEDS = tuple(range(1000, 1020))
DEFAULT_MEASUREMENT_NOISE = 1e-8
PROCESS_FLOOR = 1e-10
DIVERGENCE_LIMIT = 1.0e6

STRESS_CONFIG: Dict[str, Dict[str, Dict[str, float]]] = {
    "humanoid": {
        "nominal": dict(disturbance_scale=1.0, initial_state_scale=1.0,
                        torque_bound_factor=1.0, friction_scale=1.0,
                        contact_stiffness_scale=1.0),
        "moderate": dict(disturbance_scale=1.5, initial_state_scale=1.15,
                         torque_bound_factor=0.75, friction_scale=1.25,
                         contact_stiffness_scale=1.25),
        "extreme": dict(disturbance_scale=2.5, initial_state_scale=1.35,
                        torque_bound_factor=0.50, friction_scale=1.75,
                        contact_stiffness_scale=1.50),
    },
    "microgrid": {
        "nominal": dict(disturbance_scale=1.0, initial_state_scale=1.0,
                        command_bound_factor=1.0, nonlinear_damping_scale=1.0),
        "moderate": dict(disturbance_scale=1.5, initial_state_scale=1.25,
                         command_bound_factor=0.75, nonlinear_damping_scale=1.5),
        "extreme": dict(disturbance_scale=2.0, initial_state_scale=1.50,
                        command_bound_factor=0.50, nonlinear_damping_scale=2.0),
    },
}

TAIL_CONFIG = {
    "gaussian": dict(kind="gaussian"),
    "student_t_3": dict(kind="student_t", df=3),
    "student_t_5": dict(kind="student_t", df=5),
    "student_t_10": dict(kind="student_t", df=10),
    "tail_contamination": dict(kind="tail_contamination", epsilon=0.05,
                                tail_scale=4.0),
}


# --------------------------------------------------------------------------- #
#  Small I/O helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json(name: str, payload: Mapping) -> str:
    path = os.path.join(LOG, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
    return path


def _f(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_formal_table(name: str, headers: Sequence[str], rows: Sequence[Sequence],
                        notes: str) -> Tuple[str, str]:
    """Write a table to output/tables and a convenience copy to output/."""
    csv_path = os.path.join(TBL, name + ".csv")
    md_path = os.path.join(TBL, name + ".md")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(["" if x is None else str(x) for x in row])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("" if x is None else str(x)
                                      for x in row) + " |\n")
        f.write("\n" + notes.strip() + "\n")
    # Convenience copies requested by the experiment interface.  They are
    # generated only from the computed table above.
    shutil.copyfile(csv_path, os.path.join(OUT, name + ".csv"))
    shutil.copyfile(md_path, os.path.join(OUT, name + ".md"))
    return csv_path, md_path


# --------------------------------------------------------------------------- #
#  Observer
# --------------------------------------------------------------------------- #
def _observer_gain(spec: Mapping, measurement_noise_std: float = DEFAULT_MEASUREMENT_NOISE):
    nx, ny = spec["A"].shape[0], spec["C"].shape[0]
    qw = spec["E"] @ np.diag(np.square(spec["d_std"])) @ spec["E"].T
    qw = qw + PROCESS_FLOOR * np.eye(nx)
    rv = max(float(measurement_noise_std), 1e-12) ** 2 * np.eye(ny)
    gain, _, _ = core.kalman_observer_gain(spec["A"], spec["C"], qw, rv)
    return gain


# --------------------------------------------------------------------------- #
#  Humanoid model
# --------------------------------------------------------------------------- #
def _humanoid_step(x: np.ndarray, u: np.ndarray, d: np.ndarray,
                   stress: Mapping[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """One explicit-Euler humanoid step with contact, friction and saturation.

    ``u`` must already be the applied (saturated) torque.  The returned metadata
    describes the current state before the update, which is the state used by
    the stage cost and by the range/contact diagnostics.
    """
    theta = x[[0, 2, 4]]
    omega = x[[1, 3, 5]]
    contact_torque = np.zeros(3)
    contact_any = 0
    contact_joints = 0
    friction_active = 0
    max_excess = 0.0

    for i in range(3):
        if theta[i] > HUMANOID_LIMITS[i]:
            contact_torque[i] = (
                -HUMANOID_CONTACT_K[i] * stress["contact_stiffness_scale"]
                * (theta[i] - HUMANOID_LIMITS[i])
                - HUMANOID_CONTACT_C[i] * omega[i])
            contact_any = 1
            contact_joints += 1
            max_excess = max(max_excess, float(theta[i] - HUMANOID_LIMITS[i]))
        elif theta[i] < -HUMANOID_LIMITS[i]:
            contact_torque[i] = (
                -HUMANOID_CONTACT_K[i] * stress["contact_stiffness_scale"]
                * (theta[i] + HUMANOID_LIMITS[i])
                - HUMANOID_CONTACT_C[i] * omega[i])
            contact_any = 1
            contact_joints += 1
            max_excess = max(max_excess, float(-theta[i] - HUMANOID_LIMITS[i]))
        if abs(omega[i]) > HUMANOID_FRICTION_EPS:
            friction_active += 1

    friction = (HUMANOID_FRICTION_C * stress["friction_scale"]
                * np.tanh(omega / HUMANOID_FRICTION_EPS))
    tau = (u + d - HUMANOID_B * omega - HUMANOID_GRAVITY * np.sin(theta)
           - friction + contact_torque)
    rhs = np.zeros_like(x)
    rhs[[0, 2, 4]] = omega
    rhs[[1, 3, 5]] = tau / HUMANOID_J
    x_next = x + HUMANOID_DT * rhs
    return x_next, dict(contact_any=bool(contact_any),
                        contact_joints=contact_joints,
                        friction_active=friction_active,
                        max_range_excess=max_excess)


# --------------------------------------------------------------------------- #
#  Microgrid model
# --------------------------------------------------------------------------- #
def _microgrid_step(x: np.ndarray, u: np.ndarray, d: np.ndarray,
                    spec: Mapping, stress: Mapping[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """One discrete step of the nonlinear two-area LFC model.

    The nominal matrices are the forward-Euler matrices from ``models.py``.
    The nonlinear correction is integrated with the same forward-Euler step:
        x_{k+1}=A x_k+B u_k+E d_k+Ts*[-D_nl f|f|/(2H),0,...].
    """
    ac, _, _, _ = models.microgrid_continuous()
    h = np.array([5.0, 4.0], dtype=float)
    f = x[[0, 2]]
    damping_scale = stress["nonlinear_damping_scale"]
    nonlinear = np.array([-MICROGRID_NL_DAMPING[0] * damping_scale * f[0] * abs(f[0]) / (2.0 * h[0]),
                          -MICROGRID_NL_DAMPING[1] * damping_scale * f[1] * abs(f[1]) / (2.0 * h[1])])
    x_next = (spec["A"] @ x + spec["B"] @ u + spec["E"] @ d
              + MICROGRID_DT * np.array([nonlinear[0], 0.0, nonlinear[1], 0.0]))
    freq_excess = np.maximum(np.abs(f) - MICROGRID_FREQ_LIMIT, 0.0)
    return x_next, dict(freq_violation=bool(np.any(freq_excess > 0.0)),
                        max_freq_excess=float(np.max(freq_excess)))


# --------------------------------------------------------------------------- #
#  Disturbance laws for the tail experiment
# --------------------------------------------------------------------------- #
def _draw_standard_disturbance(rng, law: Mapping) -> Tuple[np.ndarray, float]:
    """Return a standardized disturbance z and a tail-event indicator.

    d = d_bar + diag(d_std) z, so z has zero mean and unit marginal variance
    for all laws below when the covariance is finite.
    """
    kind = law["kind"]
    if kind == "gaussian":
        z = rng.standard_normal(len(law["_std"]))
        return z, float(np.linalg.norm(z) > 3.0)
    if kind == "student_t":
        df = int(law["df"])
        if df <= 2:
            raise ValueError("Student-t validation requires df>2 for finite variance")
        # t_df / sqrt(df/(df-2)) has unit variance.
        z = rng.standard_t(df, size=len(law["_std"])) / np.sqrt(df / (df - 2.0))
        return z, float(np.linalg.norm(z) > 3.0)
    if kind == "tail_contamination":
        eps = float(law["epsilon"])
        a = float(law["tail_scale"])
        if not (0.0 < eps < 1.0 / (a * a)):
            raise ValueError("tail contamination requires 0<eps<1/a^2")
        low_scale = np.sqrt((1.0 - eps * a * a) / (1.0 - eps))
        high = rng.random() < eps
        scale = a if high else low_scale
        z = scale * rng.standard_normal(len(law["_std"]))
        return z, float(high)
    raise ValueError(kind)


def _disturbance_sampler(law: Mapping) -> Callable:
    """Build a sampler without mutating the shared law dictionary."""
    law = dict(law)

    def sampler(rng, mean, std):
        law["_std"] = std
        z, tail = _draw_standard_disturbance(rng, law)
        return mean + std * z, tail

    return sampler


# --------------------------------------------------------------------------- #
#  One closed-loop rollout
# --------------------------------------------------------------------------- #
def _run_one(spec: Mapping, controller: str, K: np.ndarray, p: np.ndarray,
             stress: Mapping[str, float], seed: int,
             law: Optional[Mapping] = None,
             T: Optional[int] = None, ss: Optional[int] = None,
             measurement_noise_std: float = DEFAULT_MEASUREMENT_NOISE) -> Dict:
    """Run one nonlinear output-feedback closed-loop seed.

    The same seed and the same disturbance law are used for DRC and LQR in the
    comparison.  Results for a diverged run are retained and explicitly marked;
    they are never replaced by a nominal value.
    """
    scenario = spec["name"]
    T = int(spec["T"] if T is None else T)
    ss = int(spec["ss"] if ss is None else ss)
    rng = np.random.default_rng(seed)
    L = _observer_gain(spec, measurement_noise_std)

    x = np.asarray(spec["x0"], dtype=float).copy()
    if scenario == "humanoid":
        x[[0, 2, 4]] *= stress["initial_state_scale"]
    else:
        x *= stress["initial_state_scale"]
    x_ref = np.asarray(spec["x_ref"], dtype=float)
    xhat = x_ref.copy()

    dist_scale = float(stress["disturbance_scale"])
    d_mean = np.asarray(spec["d_bar"], dtype=float) * dist_scale
    d_std = np.asarray(spec["d_std"], dtype=float) * dist_scale
    if law is None:
        law = {"kind": "gaussian"}
    sampler = _disturbance_sampler(law)

    if scenario == "humanoid":
        u_bound = HUMANOID_TAU_BOUND * stress["torque_bound_factor"]
    elif scenario == "microgrid":
        u_bound = MICROGRID_U_BOUND * stress["command_bound_factor"]
    else:
        raise ValueError(scenario)

    mae_values, cost_values, recon_values = [], [], []
    saturated_steps = 0
    contact_steps = 0
    contact_joint_steps = 0
    friction_joint_steps = 0
    freq_violation_steps = 0
    tail_events = 0
    max_state = 0.0
    max_range_excess = 0.0
    max_freq_excess = 0.0
    diverged_step = None

    for t in range(T):
        e = x - x_ref
        u_cmd = K @ (xhat - x_ref) + p
        saturated = bool(np.any(np.abs(u_cmd) > u_bound + 1e-12))
        saturated_steps += int(saturated)
        u = np.clip(u_cmd, -u_bound, u_bound)

        x_abs_max = float(np.max(np.abs(x)))
        max_state = max(max_state, x_abs_max)
        if not np.all(np.isfinite(x)) or max_state > DIVERGENCE_LIMIT:
            diverged_step = t
            break

        d_noise, tail_event = sampler(rng, d_mean, d_std)
        tail_events += int(tail_event)
        y = spec["C"] @ x + measurement_noise_std * rng.standard_normal(spec["C"].shape[0])

        if t >= ss:
            if spec["mae_mode"] == "freq":
                mae_values.append(float(np.linalg.norm(e[[0, 2]])))
            elif spec["mae_mode"] == "angle":
                mae_values.append(float(np.linalg.norm(e[[0, 2, 4]])))
            else:
                mae_values.append(float(np.linalg.norm(e)))
            cost_values.append(float(e @ spec["Q"] @ e + u @ spec["R"] @ u))
            recon_values.append(float(np.linalg.norm(xhat - x)))

        if scenario == "humanoid":
            x_next, info = _humanoid_step(x, u, d_noise, stress)
            contact_steps += int(info["contact_any"])
            contact_joint_steps += int(info["contact_joints"])
            friction_joint_steps += int(info["friction_active"])
            max_range_excess = max(max_range_excess, info["max_range_excess"])
        else:
            x_next, info = _microgrid_step(x, u, d_noise, spec, stress)
            freq_violation_steps += int(info["freq_violation"])
            max_freq_excess = max(max_freq_excess, info["max_freq_excess"])

        # Nominal-model observer, no realized disturbance history is passed in.
        xhat = (spec["A"] @ xhat + spec["B"] @ u + spec["E"] @ spec["d_bar"]
                + L @ (y - spec["C"] @ xhat))
        x = x_next

    valid = len(mae_values) > 0
    if not valid:
        return dict(seed=seed, valid=False, diverged_step=diverged_step,
                    mae=None, cost=None, recon_rmse=None, max_state=max_state,
                    saturation_fraction=None, contact_fraction=None,
                    contact_joint_fraction=None, friction_active_fraction=None,
                    frequency_violation_fraction=None,
                    max_range_excess=None, max_freq_excess=None,
                    tail_fraction=None, n_steps=t + 1 if 't' in locals() else 0)

    return dict(
        seed=int(seed), valid=True, diverged_step=diverged_step,
        mae=float(np.mean(mae_values)),
        cost=float(np.mean(cost_values) / (1.0 - spec["alpha"])),
        recon_rmse=float(np.sqrt(np.mean(np.square(recon_values)))),
        max_state=float(max_state),
        saturation_fraction=float(saturated_steps / (t + 1)),
        contact_fraction=float(contact_steps / (t + 1)) if scenario == "humanoid" else None,
        contact_joint_fraction=(float(contact_joint_steps / (3 * (t + 1)))
                                if scenario == "humanoid" else None),
        friction_active_fraction=(float(friction_joint_steps / (3 * (t + 1)))
                                  if scenario == "humanoid" else None),
        frequency_violation_fraction=(float(freq_violation_steps / (t + 1))
                                      if scenario == "microgrid" else None),
        max_range_excess=(float(max_range_excess) if scenario == "humanoid" else None),
        max_freq_excess=(float(max_freq_excess) if scenario == "microgrid" else None),
        tail_fraction=float(tail_events / (t + 1)),
        n_steps=int(t + 1),
    )


# --------------------------------------------------------------------------- #
#  Aggregation and output
# --------------------------------------------------------------------------- #
def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return None, None
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return mean, std


def _aggregate(runs: Sequence[Mapping]) -> Dict:
    valid = [r for r in runs if r.get("valid")]
    mae, mae_std = _mean_std(r.get("mae") for r in valid)
    cost, cost_std = _mean_std(r.get("cost") for r in valid)
    recon, recon_std = _mean_std(r.get("recon_rmse") for r in valid)
    sat, sat_std = _mean_std(r.get("saturation_fraction") for r in valid)
    contact, contact_std = _mean_std(r.get("contact_fraction") for r in valid)
    contact_joint, _ = _mean_std(r.get("contact_joint_fraction") for r in valid)
    friction, _ = _mean_std(r.get("friction_active_fraction") for r in valid)
    fviol, fviol_std = _mean_std(r.get("frequency_violation_fraction") for r in valid)
    tail, _ = _mean_std(r.get("tail_fraction") for r in valid)
    return dict(
        n_seeds=len(runs), valid_seeds=len(valid), diverged_seeds=len(runs) - len(valid),
        mae=mae, mae_std=mae_std, cost=cost, cost_std=cost_std,
        recon_rmse=recon, recon_rmse_std=recon_std,
        saturation_fraction=sat, saturation_std=sat_std,
        contact_fraction=contact, contact_std=contact_std,
        contact_joint_fraction=contact_joint,
        friction_active_fraction=friction,
        frequency_violation_fraction=fviol,
        frequency_violation_std=fviol_std,
        tail_fraction=tail,
        max_state=float(max((r.get("max_state", 0.0) for r in valid), default=np.nan)),
        max_range_excess=float(max((r.get("max_range_excess") or 0.0 for r in valid),
                                 default=0.0)),
        max_freq_excess=float(max((r.get("max_freq_excess") or 0.0 for r in valid),
                                default=0.0)),
    )


def _solve_policies(spec: Mapping):
    drc = core.solve_drc_pi(spec["A"], spec["B"], spec["E"],
                            spec["Q"], spec["R"], spec["alpha"],
                            spec["lam"], spec["d_bar"])
    klqr, lqr_rho, lqr_iters, lqr_converged = core.solve_discounted_lqr(
        spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"])
    policies = {
        "DRC": dict(K=drc["K"], p=drc["p"], rho=drc["rho_G"],
                    iters=drc["iters"], converged=drc["converged"]),
        "LQR": dict(K=-klqr, p=np.zeros(spec["B"].shape[1]),
                    rho=lqr_rho, iters=lqr_iters, converged=lqr_converged),
    }
    return policies


def _scenario_specs():
    return {"humanoid": models.humanoid_spec(),
            "microgrid": models.microgrid_spec()}


def run_nonlinear_validation(seeds: Optional[Sequence[int]] = None,
                             write_outputs: bool = True) -> Dict:
    """Run nominal/moderate/extreme nonlinear closed-loop experiments.

    The returned dictionary is keyed by ``(scenario, stress_level, controller)``.
    If ``write_outputs`` is true, ``od_e8_*`` Markdown/CSV and manifest/log
    files are written atomically from the computed results.
    """
    seeds = tuple(DEFAULT_SEEDS if seeds is None else seeds)
    specs = _scenario_specs()
    results = {}
    all_runs = {}
    rows = []

    for scenario, spec in specs.items():
        policies = _solve_policies(spec)
        for level, stress in STRESS_CONFIG[scenario].items():
            for controller, policy in policies.items():
                runs = []
                for seed in seeds:
                    run = _run_one(spec, controller, policy["K"], policy["p"],
                                   stress, seed)
                    runs.append(run)
                agg = _aggregate(runs)
                agg["policy_rho"] = policy["rho"]
                agg["policy_iters"] = policy["iters"]
                agg["policy_converged"] = policy["converged"]
                results[(scenario, level, controller)] = agg
                all_runs[f"{scenario}::{level}::{controller}"] = runs
                rows.append([
                    scenario, level, controller, len(seeds),
                    agg["valid_seeds"], agg["diverged_seeds"],
                    _f(agg["mae"]), _f(agg["mae_std"]),
                    _f(agg["cost"]), _f(agg["cost_std"]),
                    _f(agg["saturation_fraction"], 4),
                    _f(agg["contact_fraction"], 4),
                    _f(agg["contact_joint_fraction"], 4),
                    _f(agg["friction_active_fraction"], 4),
                    _f(agg["frequency_violation_fraction"], 4),
                    _f(agg["max_state"], 3),
                    _f(agg["max_range_excess"], 6),
                    _f(agg["max_freq_excess"], 6),
                    _f(agg["recon_rmse"], 6),
                ])

    headers = [
        "Scenario", "Stress level", "Controller", "Seeds", "Valid seeds",
        "Diverged seeds", "MAE", "MAE std", "Discounted cost", "Cost std",
        "Saturation fraction", "Contact fraction (any joint)",
        "Contact joint-step fraction", "Friction active fraction",
        "Frequency violation fraction", "Max abs state",
        "Max joint-range excess (rad)", "Max frequency excess (Hz)",
        "Observer RMSE",
    ]
    notes = """**Scope and definitions.** The humanoid rows are a reduced three-joint
planar model with gravity, viscous and smooth-Coulomb friction, compliant
joint-limit contact, torque saturation, and operating-range diagnostics; they
are not a claim of full humanoid hardware fidelity. The microgrid rows use the
standard two-area LFC linear model plus quadratic frequency damping, governor
command saturation, and a ±0.5 Hz operating-range diagnostic. The observer is
a nominal-model steady-state Kalman filter driven by `y=Cx`; it receives only
the nominal disturbance mean, never the realized disturbance history. The stage
cost uses the saturated applied input. Stress tuples and all fixed parameters
are recorded in `output/logs/od_e8_nonlinear_manifest.json`.

Humanoid contact fraction is the fraction of control steps with at least one
joint outside its limit; contact joint-step fraction divides by the three
joint-steps. Friction active fraction is the fraction of joint-steps with
`|omega| > 0.02 rad/s`. Microgrid frequency violation is the fraction of steps
with either frequency outside ±0.5 Hz. `Max frequency excess` is the maximum
positive excursion beyond 0.5 Hz; it is zero when the range is respected.
The joint-range excess column is the maximum positive excursion beyond the
corresponding joint limit (rad); it is zero when all joints remain in range.
"""
    if write_outputs:
        _write_formal_table("od_e8_nonlinear_contact_saturation", headers, rows, notes)
        manifest = dict(
            experiment="od_e8_nonlinear_contact_saturation",
            scenario_specs={k: dict(name=v["name"], T=int(v["T"]), ss=int(v["ss"]),
                                    x0=v["x0"], d_bar=v["d_bar"], d_std=v["d_std"],
                                    C=v["C"], dt=(HUMANOID_DT if k == "humanoid" else MICROGRID_DT))
                             for k, v in specs.items()},
            stress_config=STRESS_CONFIG,
            humanoid_parameters=dict(J=HUMANOID_J, b=HUMANOID_B,
                                     gravity=HUMANOID_GRAVITY, limits=HUMANOID_LIMITS,
                                     contact_k=HUMANOID_CONTACT_K, contact_c=HUMANOID_CONTACT_C,
                                     friction_c=HUMANOID_FRICTION_C,
                                     friction_eps=HUMANOID_FRICTION_EPS,
                                     dt=HUMANOID_DT, torque_bound=HUMANOID_TAU_BOUND),
            microgrid_parameters=dict(nonlinear_damping=MICROGRID_NL_DAMPING,
                                      frequency_limit=MICROGRID_FREQ_LIMIT,
                                      command_bound=MICROGRID_U_BOUND,
                                      dt=MICROGRID_DT),
            seeds=list(seeds), measurement_noise_std=DEFAULT_MEASUREMENT_NOISE,
            process_floor=PROCESS_FLOOR, divergence_limit=DIVERGENCE_LIMIT,
            results={f"{k[0]}::{k[1]}::{k[2]}": v for k, v in results.items()},
            raw_runs=all_runs,
        )
        _write_json("od_e8_nonlinear_manifest", manifest)
        with open(os.path.join(LOG, "od_e8_nonlinear_validation.log"), "w",
                  encoding="utf-8") as f:
            f.write("od_e8 nonlinear validation completed\n")
            for (scenario, level, controller), value in results.items():
                f.write(f"{scenario}/{level}/{controller}: "
                        f"valid={value['valid_seeds']}/{value['n_seeds']}, "
                        f"mae={value['mae']}, cost={value['cost']}\n")
    return results


def run_tail_only_validation(seeds: Optional[Sequence[int]] = None,
                             write_outputs: bool = True) -> Dict:
    """Run Gaussian, Student-t and covariance-matched tail-contamination laws."""
    seeds = tuple(DEFAULT_SEEDS if seeds is None else seeds)
    results = {}
    all_runs = {}
    rows = []
    specs = _scenario_specs()

    for scenario, spec in specs.items():
        policies = _solve_policies(spec)
        stress = STRESS_CONFIG[scenario]["nominal"]
        for law_name, law_base in TAIL_CONFIG.items():
            # Keep the live sampler state out of the shared configuration.
            for controller, policy in policies.items():
                runs = []
                for seed in seeds:
                    run = _run_one(spec, controller, policy["K"], policy["p"],
                                   stress, seed, law=law_base)
                    runs.append(run)
                agg = _aggregate(runs)
                results[(scenario, law_name, controller)] = agg
                all_runs[f"{scenario}::{law_name}::{controller}"] = runs
                rows.append([
                    scenario, law_name, controller, len(seeds),
                    agg["valid_seeds"], agg["diverged_seeds"],
                    _f(agg["mae"]), _f(agg["mae_std"]),
                    _f(agg["cost"]), _f(agg["cost_std"]),
                    _f(agg["tail_fraction"], 5),
                    _f(agg["saturation_fraction"], 4),
                    _f(agg["contact_fraction"], 4)
                    if scenario == "humanoid" else "—",
                    _f(agg["frequency_violation_fraction"], 4)
                    if scenario == "microgrid" else "—",
                    _f(agg["max_state"], 3),
                ])

    headers = [
        "Scenario", "Distribution", "Controller", "Seeds", "Valid seeds",
        "Diverged seeds", "MAE", "MAE std", "Discounted cost", "Cost std",
        "Tail event fraction", "Saturation fraction", "Contact fraction",
        "Frequency violation fraction", "Max abs state",
    ]
    notes = """**Tail-only comparison.** `gaussian` is the nominal Gaussian law.
`student_t_3/5/10` use covariance-matched Student-t laws. `tail_contamination`
is a covariance-matched two-scale Gaussian mixture with epsilon=0.05 and tail
scale=4: with probability 0.95 a Gaussian with reduced scale is used, and with
probability 0.05 a Gaussian with scale 4 is used; the marginal covariance is
exactly the nominal one. It therefore isolates increased tail mass, but it does
not preserve the original central variance. `Tail event fraction` is the
empirical fraction of the mixture's tail component for the contamination law,
and the fraction with standardized Euclidean norm greater than 3 for Gaussian
and Student-t laws. All rows use the nominal nonlinear stress parameters, 20
seeds where available, and the same unknown-history Kalman observer as the
main nonlinear experiment. This is a stress comparison, not a proof that a
fixed Wasserstein radius contains these alternative laws.
"""
    if write_outputs:
        _write_formal_table("od_e9_tail_only", headers, rows, notes)
        manifest = dict(experiment="od_e9_tail_only", seeds=list(seeds),
                        nominal_stress=STRESS_CONFIG,
                        distributions=TAIL_CONFIG,
                        results={f"{k[0]}::{k[1]}::{k[2]}": v
                                 for k, v in results.items()},
                        raw_runs=all_runs)
        _write_json("od_e9_tail_only_manifest", manifest)
        with open(os.path.join(LOG, "od_e9_tail_only.log"), "w",
                  encoding="utf-8") as f:
            f.write("od_e9 tail-only validation completed\n")
            for (scenario, law, controller), value in results.items():
                f.write(f"{scenario}/{law}/{controller}: "
                        f"valid={value['valid_seeds']}/{value['n_seeds']}, "
                        f"mae={value['mae']}, cost={value['cost']}, "
                        f"tail={value['tail_fraction']}\n")
    return results


def main() -> None:
    print("Running od_e8 nonlinear/contact/saturation validation ...")
    nonlinear = run_nonlinear_validation(write_outputs=True)
    print("od_e8 complete")
    print("Running od_e9 tail-only/heavy-tail comparison ...")
    tail = run_tail_only_validation(write_outputs=True)
    print("od_e9 complete")
    print(json.dumps(dict(nonlinear_rows=len(nonlinear), tail_rows=len(tail)),
                     indent=2))


if __name__ == "__main__":
    main()
