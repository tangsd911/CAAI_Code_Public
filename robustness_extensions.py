#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness experiments added for the CAAI revision.

Scope
-----
This module implements the code-side experiments requested in the revision:

1. a g=1 Wasserstein-radius / lambda outer grid scan.  For each fixed penalty
   lambda it simulates the induced affine penalized-dual adversary, computes
   the empirical 2-Wasserstein radius to the one-point reference support, and
   evaluates a finite-horizon Lagrangian objective for target radii zeta;
2. an LQR-specific penalized-dual affine adversary (not a hard-ball worst-case
   distribution), reported next to the common-adversary stress test;
3. covariance-only Gaussian shifts, covariance-matched Student-t heavy tails,
   g>1 empirical-support sample-size sensitivity, and an input-saturation
   nonlinearity stress test.

Every number in the generated tables is computed by the functions below.  The
module deliberately does not claim that a grid search certifies a hard
Wasserstein constraint.  The only exact W2-membership check below is for the
one-point reference support used by the g=1 scan; for an empirical adversary
sequence d_0,...,d_{T-1} it is

    W2_hat = sqrt( (1/T) sum_k ||d_k - d_bar||^2 ).

The finite-horizon objective includes the Lagrangian term lambda*zeta^2 and is
labelled a grid-scan dual/Lagrangian surrogate, not a certificate for the
infinite-horizon hard-ball problem.
"""

import contextlib
import csv
import io
import os
import sys
import warnings

import numpy as np

import drc_core as core
import models
import wasserstein_experiments as wx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")
LOG = os.path.join(OUT, "logs")
for _d in (OUT, TBL, LOG):
    os.makedirs(_d, exist_ok=True)

# Experimental design grids (not result values).
ZETA_GRID = np.array([0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10])
LAMBDA_GRID = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
COV_SCALES = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
TAIL_DFS = (3, 5, 10)
G_GRID = (1, 2, 5, 10, 20, 50)
SAT_FACTORS = np.array([0.25, 0.5, 1.0, 2.0])
DIVERGENCE_LIMIT = 1.0e8


# --------------------------------------------------------------------------- #
#  Generic table / table-rendering helpers
# --------------------------------------------------------------------------- #
def _write_table(name, headers, rows, notes=None):
    csv_path = os.path.join(TBL, name + ".csv")
    md_path = os.path.join(TBL, name + ".md")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(["" if x is None else str(x) for x in row])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("" if x is None else str(x)
                                      for x in row) + " |\n")
        if notes:
            f.write("\n" + notes + "\n")
    return csv_path, md_path


def _f(x, n=6):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{n}f}"


# --------------------------------------------------------------------------- #
#  Custom disturbance Monte-Carlo rollout
# --------------------------------------------------------------------------- #
def _simulate_custom(spec, K, p, sampler, seed, sat_bound=None,
                     T=None, ss=None):
    """Simulate one policy under a user-supplied disturbance sampler.

    ``sampler(rng, x)`` returns one disturbance vector.  If ``sat_bound`` is
    not None, every component of u is clipped to [-sat_bound, sat_bound].
    Returns None metrics on divergence and records the divergence step.
    """
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha = spec["Q"], spec["R"], spec["alpha"]
    x_ref = spec["x_ref"]
    T = spec["T"] if T is None else T
    ss = spec["ss"] if ss is None else ss
    rng = np.random.default_rng(seed)
    x = x_ref.copy()
    maes, stages = [], []
    sat_hits = 0
    max_x = 0.0
    diverged = None

    for t in range(T):
        e = x - x_ref
        u = K @ e + p
        if sat_bound is not None:
            u_raw = u
            u = np.clip(u, -sat_bound, sat_bound)
            sat_hits += int(np.any(np.abs(u_raw) > sat_bound + 1e-15))
        d = np.asarray(sampler(rng, x), dtype=float)
        if t >= ss:
            if spec["mae_mode"] == "freq":
                maes.append(float(np.linalg.norm(e[[0, 2]])))
            elif spec["mae_mode"] == "angle":
                maes.append(float(np.linalg.norm(e[[0, 2, 4]])))
            else:
                maes.append(float(np.linalg.norm(e)))
            stages.append(float(e @ Q @ e + u @ R @ u))
        x = A @ x + B @ u + E @ d
        max_x = max(max_x, float(np.max(np.abs(x))))
        if not np.all(np.isfinite(x)) or max_x > DIVERGENCE_LIMIT:
            diverged = t
            break

    if diverged is not None or not maes:
        return dict(diverged=diverged, mae=None, cost=None,
                    sat_fraction=None, max_state=max_x)
    return dict(
        diverged=None,
        mae=float(np.mean(maes)),
        cost=float(np.mean(stages) / (1.0 - alpha)),
        sat_fraction=float(sat_hits / T),
        max_state=max_x,
    )


def _mc_custom(spec, K, p, sampler, seeds, sat_bound=None):
    runs = [_simulate_custom(spec, K, p, sampler, seed, sat_bound=sat_bound)
            for seed in seeds]
    good = [r for r in runs if r["mae"] is not None]
    if not good:
        return dict(mae=None, mae_std=None, cost=None, cost_std=None,
                    sat_fraction=None, n_diverged=len(runs), n_valid=0)
    return dict(
        mae=float(np.mean([r["mae"] for r in good])),
        mae_std=float(np.std([r["mae"] for r in good])),
        cost=float(np.mean([r["cost"] for r in good])),
        cost_std=float(np.std([r["cost"] for r in good])),
        sat_fraction=float(np.mean([r["sat_fraction"] for r in good])),
        n_diverged=len(runs) - len(good),
        n_valid=len(good),
    )


def _gaussian_sampler(spec, cov_scale=1.0):
    def sampler(rng, x):
        return spec["d_bar"] + np.sqrt(cov_scale) * spec["d_std"] * (
            rng.standard_normal(spec["d_bar"].size))
    return sampler


def _student_t_sampler(spec, df):
    # Normalize the scale so the covariance equals d_std^2 when df>2.
    scale = np.sqrt(df / (df - 2.0))
    def sampler(rng, x):
        return spec["d_bar"] + spec["d_std"] * (
            rng.standard_t(df, size=spec["d_bar"].size) / scale)
    return sampler


def _det_affine_trajectory(spec, K, p, N, z, relative_to_error=False):
    """Run d=Nx+z (or d=Ne+z) and return the realized trajectories."""
    A, B, E = spec["A"], spec["B"], spec["E"]
    x_ref = spec["x_ref"]
    x = x_ref.copy()
    xs, us, ds = [], [], []
    diverged = None
    for t in range(spec["T"]):
        e = x - x_ref
        u = K @ e + p
        d = N @ (e if relative_to_error else x) + z
        xs.append(x.copy())
        us.append(u.copy())
        ds.append(d.copy())
        x = A @ x + B @ u + E @ d
        if not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) > DIVERGENCE_LIMIT:
            diverged = t
            break
    return dict(x=np.asarray(xs), u=np.asarray(us), d=np.asarray(ds),
                diverged=diverged)


def _point_w2_radius(d, center):
    """Exact empirical W2 radius to a one-point reference for squared cost."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return None
    return float(np.sqrt(np.mean(np.sum((d - center[None, :]) ** 2, axis=1))))


def _finite_horizon_lagrangian(spec, dvals, lam, zeta_radius):
    """Finite-horizon Lagrangian rollout at a target W2 radius.

    The rollout uses the affine maximizer from the fixed-lambda penalized
    problem.  The reported quantity is

      sum_k alpha^k [stage_k - lam ||d_k-d_bar||^2 + lam zeta^2]
      + alpha^T V(x_T).

    It is a grid-scan dual/Lagrangian surrogate for the hard-ball problem, not
    a proof that the constraint is active or that the grid point is optimal.
    """
    K, p = dvals["K"], dvals["p"]
    N, z = dvals["N"], dvals["z"]
    traj = _det_affine_trajectory(spec, K, p, N, z,
                                  relative_to_error=False)
    if traj["diverged"] is not None:
        return dict(diverged=traj["diverged"], objective=None,
                    w2_radius=None, mean_sq_penalty=None)
    xs, us, ds = traj["x"], traj["u"], traj["d"]
    T = len(xs)
    alpha = spec["alpha"]
    weights = alpha ** np.arange(T)
    e = xs - spec["x_ref"]
    stage = np.einsum("ki,ij,kj->k", e, spec["Q"], e)
    stage += np.einsum("ki,ij,kj->k", us, spec["R"], us)
    sq = np.sum((ds - spec["d_bar"][None, :]) ** 2, axis=1)
    lag = float(np.sum(weights * (stage - lam * sq + lam * zeta_radius ** 2)))
    xT = xs[-1]
    VT = float(xT @ dvals["P"] @ xT + 2.0 * dvals["chi"] @ xT + dvals["psi"])
    objective = lag + float(alpha ** T) * VT
    return dict(diverged=None, objective=objective,
                w2_radius=_point_w2_radius(ds, spec["d_bar"]),
                mean_sq_penalty=float(np.mean(sq)))


# --------------------------------------------------------------------------- #
#  1. Wasserstein radius / lambda outer grid scan (g=1)
# --------------------------------------------------------------------------- #
def exp_wasserstein_radius_scan():
    surface_rows = []
    calibration_rows = []
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        lam_data = {}
        calibration = []
        for lam in LAMBDA_GRID:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with np.errstate(over="ignore", invalid="ignore",
                                     divide="ignore"):
                        d = core.solve_drc_pi(
                            spec["A"], spec["B"], spec["E"], spec["Q"],
                            spec["R"], spec["alpha"], float(lam),
                            spec["d_bar"])
                valid = bool(d["converged"] and d["rho_G"] < 1.0
                             and d["H_uu_min_eig"] > 0.0
                             and d["H_ww_min_eig"] < 0.0)
            except Exception as exc:  # numerical failure is a real result
                surface_rows.append([name, _f(lam, 3), "False", "—", "—", "—",
                                     "—", "—", "—", "False",
                                     type(exc).__name__])
                lam_data[float(lam)] = None
                continue
            if not valid:
                surface_rows.append([name, _f(lam, 3), "True", _f(d["rho_G"]),
                                     _f(d["H_uu_min_eig"], 3),
                                     _f(d["H_ww_min_eig"], 3), "—", "—", "—",
                                     "False", "diagnostic not valid"])
                lam_data[float(lam)] = None
                continue
            scan = _finite_horizon_lagrangian(spec, d, float(lam), 0.0)
            if scan["objective"] is None:
                surface_rows.append([name, _f(lam, 3), "True", _f(d["rho_G"]),
                                     _f(d["H_uu_min_eig"], 3),
                                     _f(d["H_ww_min_eig"], 3), "—", "—", "—",
                                     "—", "False", "affine rollout diverged"])
                lam_data[float(lam)] = None
                continue
            lam_data[float(lam)] = dict(d=d, scan=scan)
            for zeta in ZETA_GRID:
                # Re-evaluate the already simulated trajectory at this radius;
                # only the explicit lambda*zeta^2 term changes with zeta.
                scan_z = _finite_horizon_lagrangian(
                    spec, d, float(lam), float(zeta))
                member = bool(scan_z["w2_radius"] <= float(zeta) + 1e-14)
                surface_rows.append([
                    name, _f(lam, 3), "True", _f(d["rho_G"]),
                    _f(d["H_uu_min_eig"], 3), _f(d["H_ww_min_eig"], 3),
                    _f(scan_z["w2_radius"]), _f(zeta, 5),
                    _f(scan_z["objective"]), str(member),
                    "finite-horizon grid scan",
                ])

        for zeta in ZETA_GRID:
            candidates = []
            members = []
            for lam, item in lam_data.items():
                if item is None:
                    continue
                sc = _finite_horizon_lagrangian(spec, item["d"], lam, float(zeta))
                if sc["objective"] is None:
                    continue
                candidates.append((lam, sc))
                if sc["w2_radius"] <= float(zeta) + 1e-14:
                    members.append((lam, sc))
            if candidates:
                best = min(candidates, key=lambda x: x[1]["objective"])
                best_member = (min(members, key=lambda x: x[1]["objective"])
                               if members else None)
                calibration_rows.append([
                    name, _f(zeta, 5), _f(best[0], 3),
                    _f(best[1]["objective"]), _f(best[1]["w2_radius"]),
                    str(best[1]["w2_radius"] <= float(zeta) + 1e-14),
                    "—" if best_member is None else _f(best_member[0], 3),
                    "—" if best_member is None else _f(best_member[1]["objective"]),
                    "—" if best_member is None else _f(best_member[1]["w2_radius"]),
                    "—" if best_member is None else
                    _f(float(zeta) - best_member[1]["w2_radius"]),
                    "grid approximation; not a hard-ball certificate",
                ])
                calibration.append(dict(
                    zeta=float(zeta), best_lambda=float(best[0]),
                    best_objective=float(best[1]["objective"]),
                    best_w2=float(best[1]["w2_radius"]),
                    best_is_member=bool(best[1]["w2_radius"] <= float(zeta) + 1e-14),
                    member_lambda=(None if best_member is None else float(best_member[0])),
                    member_objective=(None if best_member is None else
                                      float(best_member[1]["objective"])),
                    member_w2=(None if best_member is None else
                               float(best_member[1]["w2_radius"])),
                ))
            else:
                calibration_rows.append([name, _f(zeta, 5), "—", "—", "—",
                                         "False", "—", "—", "—", "—",
                                         "no valid grid point"])
                calibration.append(dict(zeta=float(zeta), best_lambda=None,
                                        best_objective=None, best_w2=None,
                                        best_is_member=False,
                                        member_lambda=None,
                                        member_objective=None,
                                        member_w2=None))
        report[name] = dict(valid_lambdas=[k for k, v in lam_data.items() if v],
                            failed_lambdas=[k for k, v in lam_data.items() if not v],
                            calibration=calibration)

    _write_table(
        "e16a_wasserstein_lambda_scan",
        ["Scenario", "lambda", "solver_converged", "rho_G", "H_uu_min",
         "H_ww_min", "adversary_W2_radius", "target_zeta",
         "dual_objective", "W2_member", "scope"],
        surface_rows,
        notes=("g=1 reference support at d_bar. W2_member is exact for the "
               "finite adversary sequence and one-point reference. The dual "
               "objective is finite-horizon and lambda-grid scanned."),
    )
    _write_table(
        "e16b_wasserstein_radius_calibration",
        ["Scenario", "target_zeta", "best_grid_lambda", "best_dual_objective",
         "W2_at_best_dual", "best_dual_is_member", "best_member_lambda",
         "best_member_objective", "W2_at_best_member", "radius_slack", "scope"],
        calibration_rows,
        notes=("Grid calibration only. The hard-ball problem would require an "
               "outer minimization over lambda with a membership/complementarity "
               "certificate; this table reports the discrete scan honestly."),
    )

    lines = ["# Wasserstein radius / lambda grid scan (g=1)\n",
             "The reference support is the one-point distribution at `d_bar`. "
             "For the induced affine adversary sequence, "
             "`W2_hat=sqrt(mean_k ||d_k-d_bar||^2)` is exact for that finite "
             "sequence. The objective is a finite-horizon Lagrangian surrogate "
             "with the term `lambda*zeta^2`; this is a grid scan, not a proof "
             "of hard-constraint optimality.\n"]
    for name in ("humanoid", "microgrid"):
        lines.append(f"- **{name}**: valid lambda grid points "
                     f"{report[name]['valid_lambdas']}; failed points "
                     f"{report[name]['failed_lambdas']}.")
    with open(os.path.join(OUT, "WASSERSTEIN_RADIUS_SCAN.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  2. LQR-specific penalized-dual affine adversary
# --------------------------------------------------------------------------- #
def exp_lqr_own_adversary():
    rows = []
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        drc = core.solve_drc_pi(
            spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
            spec["alpha"], spec["lam"], spec["d_bar"])
        lqr = wx.lqr_own_adversary(spec, spec["lam"],
                                   V=spec["d_bar"].reshape(1, -1))

        nom_d = _mc_custom(spec, drc["K"], drc["p"], _gaussian_sampler(spec),
                           spec["seeds"])
        nom_l = _mc_custom(spec, lqr["K"], lqr["p"], _gaussian_sampler(spec),
                           spec["seeds"])

        # DRC's own dual adversary is d=Nx+z (absolute state in the solver).
        traj_d = _det_affine_trajectory(
            spec, drc["K"], drc["p"], drc["N"], drc["z"],
            relative_to_error=False)
        # LQR's own dual adversary is returned in absolute-state coordinates:
        # d = N_e x + z_e.
        traj_l = _det_affine_trajectory(
            spec, lqr["K"], lqr["p"], lqr["N"], lqr["z"],
            relative_to_error=False)

        def summarize_traj(traj):
            if traj["diverged"] is not None:
                return dict(mae=None, cost=None, w2=None,
                            diverged=traj["diverged"])
            xs, us, ds = traj["x"], traj["u"], traj["d"]
            e = xs - spec["x_ref"]
            stage = np.einsum("ki,ij,kj->k", e, spec["Q"], e)
            stage += np.einsum("ki,ij,kj->k", us, spec["R"], us)
            if spec["mae_mode"] == "freq":
                mae = float(np.mean(np.linalg.norm(e[:, [0, 2]], axis=1)))
            elif spec["mae_mode"] == "angle":
                mae = float(np.mean(np.linalg.norm(e[:, [0, 2, 4]], axis=1)))
            else:
                mae = float(np.mean(np.linalg.norm(e, axis=1)))
            return dict(
                mae=mae,
                cost=float(np.mean(stage) / (1.0 - spec["alpha"])),
                w2=_point_w2_radius(ds, spec["d_bar"]),
                diverged=None)

        adv_d = summarize_traj(traj_d)
        adv_l = summarize_traj(traj_l)
        report[name] = dict(
            nominal=dict(drc=nom_d, lqr=nom_l),
            own_adversary=dict(drc=adv_d, lqr=adv_l),
            lqr_lambda_crit=lqr["lam_crit"],
            lqr_inner_well_posed=lqr["inner_well_posed"],
            lam=spec["lam"],
        )
        for ctrl, nom, adv in (("DRC", nom_d, adv_d),
                              ("LQR", nom_l, adv_l)):
            rows.append([
                name, ctrl, _f(nom["mae"]), _f(nom["cost"]),
                _f(adv["mae"]), _f(adv["cost"]), _f(adv["w2"]),
                "—" if adv["diverged"] is None else str(adv["diverged"]),
                _f(lqr["lam_crit"]),
                str(lqr["inner_well_posed"]),
                "own penalized-dual affine adversary; not hard-ball worst case",
            ])
    _write_table(
        "e17_lqr_own_adversary",
        ["Scenario", "Controller", "nominal MAE", "nominal discounted cost",
         "own-adversary MAE", "own-adversary discounted cost",
         "own-adversary W2 radius", "diverged step",
         "LQR lambda_crit", "LQR inner well posed", "scope"],
        rows,
        notes=("The LQR adversary is its own fixed-lambda penalized-dual affine "
               "maximizer, not a hard Wasserstein-ball worst-case distribution."),
    )
    lines = ["# LQR-specific penalized-dual affine adversary\n",
             "`LQR` uses its own affine maximizer of "
             "`alpha V_LQR(e_next)-lambda||d-d_bar||^2`; the DRC adversary is "
             "never reused. "
             "This is not a hard-ball worst-case distribution.\n"]
    for name, r in report.items():
        lines.append(
            f"- **{name}**: LQR lambda_crit={r['lqr_lambda_crit']:.6f}, "
            f"lambda={r['lam']:.1f}, inner_well_posed={r['lqr_inner_well_posed']}.")
    with open(os.path.join(OUT, "LQR_OWN_ADVERSARY.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  3. Covariance-only Gaussian shifts
# --------------------------------------------------------------------------- #
def exp_covariance_only():
    rows = []
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        drc = core.solve_drc_pi(
            spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
            spec["alpha"], spec["lam"], spec["d_bar"])
        Klqr, _, _, _ = core.solve_discounted_lqr(
            spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"])
        seeds = spec["seeds"][:10]
        report[name] = {}
        for scale in COV_SCALES:
            sampler = _gaussian_sampler(spec, float(scale))
            rd = _mc_custom(spec, drc["K"], drc["p"], sampler, seeds)
            rl = _mc_custom(spec, -Klqr, np.zeros(spec["B"].shape[1]),
                            sampler, seeds)
            report[name][float(scale)] = dict(drc=rd, lqr=rl)
            for ctrl, r in (("DRC", rd), ("LQR", rl)):
                rows.append([name, _f(scale, 2), ctrl, _f(r["mae"]),
                             _f(r["mae_std"]), _f(r["cost"]),
                             _f(r["cost_std"]), str(r["n_valid"]),
                             str(r["n_diverged"])])
    _write_table(
        "e18_covariance_only",
        ["Scenario", "covariance scale", "Controller", "MAE", "MAE std",
         "discounted cost", "cost std", "valid seeds", "diverged seeds"],
        rows,
        notes=("Mean is held fixed at d_bar; only the Gaussian covariance is "
               "scaled by the listed factor. Ten fixed seeds are used."),
    )
    lines = ["# Covariance-only Gaussian shift\n",
             "Only the disturbance covariance is changed; the mean is fixed.</br>"]
    for name, r in report.items():
        lines.append(f"## {name}")
        for scale, v in r.items():
            lines.append(
                f"- scale={scale:.2f}: DRC cost={v['drc']['cost']:.6f}, "
                f"LQR cost={v['lqr']['cost']:.6f}.")
    with open(os.path.join(OUT, "COVARIANCE_ONLY.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  4. Covariance-matched Student-t heavy tails
# --------------------------------------------------------------------------- #
def exp_heavy_tail():
    rows = []
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        drc = core.solve_drc_pi(
            spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
            spec["alpha"], spec["lam"], spec["d_bar"])
        Klqr, _, _, _ = core.solve_discounted_lqr(
            spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"])
        seeds = spec["seeds"][:10]
        kernels = [("Gaussian", _gaussian_sampler(spec))]
        kernels += [(f"Student-t(df={df})", _student_t_sampler(spec, df))
                    for df in TAIL_DFS]
        report[name] = {}
        for label, sampler in kernels:
            rd = _mc_custom(spec, drc["K"], drc["p"], sampler, seeds)
            rl = _mc_custom(spec, -Klqr, np.zeros(spec["B"].shape[1]),
                            sampler, seeds)
            report[name][label] = dict(drc=rd, lqr=rl)
            for ctrl, r in (("DRC", rd), ("LQR", rl)):
                rows.append([name, label, ctrl, _f(r["mae"]),
                             _f(r["mae_std"]), _f(r["cost"]),
                             _f(r["cost_std"]), str(r["n_valid"]),
                             str(r["n_diverged"])])
    _write_table(
        "e19_heavy_tail",
        ["Scenario", "disturbance law", "Controller", "MAE", "MAE std",
         "discounted cost", "cost std", "valid seeds", "diverged seeds"],
        rows,
        notes=("Student-t samples are scaled by sqrt(df/(df-2)) so their "
               "covariance matches the nominal Gaussian when df>2. This is a "
               "covariance-matched tail stress test, not a Wasserstein-radius "
               "certificate."),
    )
    lines = ["# Heavy-tail disturbance stress test\n",
             "Covariance-matched Student-t kernels with the same fixed seeds.</br>"]
    for name, r in report.items():
        lines.append(f"## {name}")
        for label, v in r.items():
            lines.append(
                f"- {label}: DRC cost={v['drc']['cost']:.6f}, "
                f"LQR cost={v['lqr']['cost']:.6f}.")
    with open(os.path.join(OUT, "HEAVY_TAIL.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  5. g>1 empirical-support sample-size sensitivity
# --------------------------------------------------------------------------- #
def exp_g_sample_size():
    rows = []
    report = {}
    n_support_draws, n_mc_seeds = 5, 2
    for s_idx, (name, spec) in enumerate(
            (("humanoid", models.humanoid_spec()),
             ("microgrid", models.microgrid_spec()))):
        report[name] = {}
        for g in G_GRID:
            recs = []
            for rep in range(n_support_draws):
                if g == 1:
                    V = spec["d_bar"].reshape(1, -1)
                else:
                    rng_support = np.random.default_rng(
                        7000 + 100 * s_idx + 17 * g + rep)
                    V = spec["d_bar"] + spec["d_std"] * rng_support.standard_normal(
                        (g, spec["d_bar"].size))
                try:
                    d = core.solve_drc_pi_g(
                        spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
                        spec["alpha"], spec["lam"], V)
                    if not d["converged"] or d["rho_G"] >= 1.0:
                        recs.append(dict(valid=False, converged=d["converged"],
                                         rho=d["rho_G"]))
                        continue
                except Exception:
                    recs.append(dict(valid=False, converged=False, rho=np.nan))
                    continue
                mc_seeds = spec["seeds"][:n_mc_seeds]
                mc = _mc_custom(spec, d["K"], d["p"],
                                _gaussian_sampler(spec), mc_seeds)
                vbar_err = float(np.linalg.norm(d["v_bar"] - spec["d_bar"]))
                recs.append(dict(valid=True, converged=True, rho=d["rho_G"],
                                 vbar_err=vbar_err, psi=d["psi"],
                                 mae=mc["mae"], cost=mc["cost"],
                                 n_diverged=mc["n_diverged"]))
            good = [r for r in recs if r["valid"]]
            stable_rate = len(good) / len(recs)
            if good:
                rows.append([
                    name, str(g), str(n_support_draws), str(n_mc_seeds),
                    _f(stable_rate, 2),
                    _f(np.mean([r["vbar_err"] for r in good])),
                    _f(np.std([r["vbar_err"] for r in good])),
                    _f(np.mean([r["mae"] for r in good])),
                    _f(np.std([r["mae"] for r in good])),
                    _f(np.mean([r["cost"] for r in good])),
                    _f(np.std([r["cost"] for r in good])),
                    _f(np.mean([r["psi"] for r in good]), 3),
                    _f(np.mean([r["rho"] for r in good])),
                ])
            else:
                rows.append([name, str(g), str(n_support_draws),
                             str(n_mc_seeds), _f(stable_rate, 2), "—", "—",
                             "—", "—", "—", "—", "—", "—"])
            report[name][g] = dict(records=recs,
                                   stable_rate=stable_rate)
    _write_table(
        "e20_g_sample_size",
        ["Scenario", "g support points", "support draws", "MC seeds",
         "stable rate", "mean ||vbar-dbar||", "std ||vbar-dbar||",
         "mean MAE", "std MAE", "mean cost", "std cost", "mean psi",
         "mean rho"],
        rows,
        notes=("For g=1 the support is fixed at d_bar. For g>1, supports are "
               "drawn i.i.d. from N(d_bar, diag(d_std^2)); therefore the table "
               "reports empirical-support sample-size sensitivity, not only the "
               "algebraic mean-reduction identity."),
    )
    lines = ["# g>1 empirical-support sample-size sensitivity\n",
             "At g=1 the support is the fixed reference point d_bar. At g>1 "
             "the support points are i.i.d. draws from the nominal disturbance "
             "law and are averaged over support draws and Monte-Carlo seeds.</br>"]
    for name, r in report.items():
        lines.append(f"## {name}")
        for g, v in r.items():
            lines.append(f"- g={g}: stable_rate={v['stable_rate']:.2f}.")
    with open(os.path.join(OUT, "G_SAMPLE_SIZE.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  6. Input-saturation nonlinearity stress test
# --------------------------------------------------------------------------- #
def _control_quantile(spec, K, p, seeds, q=0.99):
    vals = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = spec["x_ref"].copy()
        for _ in range(spec["T"]):
            e = x - spec["x_ref"]
            u = K @ e + p
            vals.append(float(np.max(np.abs(u))))
            d = spec["d_bar"] + spec["d_std"] * rng.standard_normal(
                spec["d_bar"].size)
            x = spec["A"] @ x + spec["B"] @ u + spec["E"] @ d
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > DIVERGENCE_LIMIT:
                break
    return float(np.quantile(vals, q))


def exp_saturation():
    rows = []
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        drc = core.solve_drc_pi(
            spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
            spec["alpha"], spec["lam"], spec["d_bar"])
        Klqr, _, _, _ = core.solve_discounted_lqr(
            spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"])
        seeds = spec["seeds"][:10]
        ref_bound = _control_quantile(spec, drc["K"], drc["p"],
                                      spec["seeds"][:3])
        report[name] = dict(reference_bound=ref_bound, cases={})
        for factor in SAT_FACTORS:
            bound = float(factor * ref_bound)
            rd = _mc_custom(spec, drc["K"], drc["p"], _gaussian_sampler(spec),
                            seeds, sat_bound=bound)
            rl = _mc_custom(spec, -Klqr, np.zeros(spec["B"].shape[1]),
                            _gaussian_sampler(spec), seeds, sat_bound=bound)
            report[name]["cases"][float(factor)] = dict(
                bound=bound, drc=rd, lqr=rl)
            for ctrl, r in (("DRC", rd), ("LQR", rl)):
                rows.append([
                    name, _f(factor, 2), _f(bound), ctrl, _f(r["mae"]),
                    _f(r["mae_std"]), _f(r["cost"]), _f(r["cost_std"]),
                    _f(r["sat_fraction"], 4), str(r["n_valid"]),
                    str(r["n_diverged"]),
                ])
    _write_table(
        "e21_saturation",
        ["Scenario", "bound factor", "symmetric u bound", "Controller",
         "MAE", "MAE std", "discounted cost", "cost std",
         "saturation fraction", "valid seeds", "diverged seeds"],
        rows,
        notes=("The plant remains the forward-Euler LTI model; the only added "
               "nonlinearity is component-wise input saturation. The reference "
               "bound is the 99th percentile of DRC nominal max|u| over three "
               "fixed seeds; both controllers use the same bound."),
    )
    lines = ["# Input-saturation nonlinearity stress test\n",
             "This test adds component-wise actuator saturation to the LTI "
             "plant; it is not a full nonlinear-plant validation.</br>"]
    for name, r in report.items():
        lines.append(f"## {name}")
        lines.append(f"- reference bound = {r['reference_bound']:.6f}")
        for factor, v in r["cases"].items():
            lines.append(
                f"- factor={factor:.2f}, bound={v['bound']:.6f}: "
                f"DRC sat={v['drc']['sat_fraction']:.4f}, "
                f"LQR sat={v['lqr']['sat_fraction']:.4f}.")
    with open(os.path.join(OUT, "SATURATION_NONLINEAR.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


# --------------------------------------------------------------------------- #
#  Result-summary helper used by the experiment drivers.
# --------------------------------------------------------------------------- #
def append_results_sections(scan, lqr_adv, cov, heavy, gsens, sat):
    path = os.path.join(OUT, "RESULTS.md")
    lines = ["\n## 9. Robustness-extension experiments (computed this run)\n",
             "These experiments are diagnostic extensions. The radius scan is a "
             "grid/Lagrangian surrogate, not a hard-ball certificate; the "
             "penalized-dual affine rollouts are not hard worst-case "
             "distributions.\n"]

    lines.append("### 9.1 g=1 Wasserstein radius / lambda grid scan\n")
    for name, r in scan.items():
        lines.append(f"- **{name}**: valid lambda points {r['valid_lambdas']}; "
                     f"failed points {r['failed_lambdas']}.")
        for c in r["calibration"]:
            if c["best_lambda"] is None:
                lines.append(f"  - zeta={c['zeta']:.5f}: no valid grid point.")
            else:
                lines.append(
                    f"  - zeta={c['zeta']:.5f}: best dual lambda="
                    f"{c['best_lambda']:.3f}, objective={c['best_objective']:.6f}, "
                    f"adversary W2={c['best_w2']:.6f}, "
                    f"member={c['best_is_member']}; best member lambda="
                    f"{c['member_lambda']}.")
    lines.append("\nSource: `output/WASSERSTEIN_RADIUS_SCAN.md`, "
                 "`output/tables/e16a_wasserstein_lambda_scan.*`, "
                 "`output/tables/e16b_wasserstein_radius_calibration.*`.\n")

    lines.append("### 9.2 LQR-specific penalized-dual affine adversary\n")
    for name, r in lqr_adv.items():
        nd = r["nominal"]["drc"]
        nl = r["nominal"]["lqr"]
        ad = r["own_adversary"]["drc"]
        al = r["own_adversary"]["lqr"]
        lines.append(
            f"- **{name}**: nominal DRC/LQR MAE {nd['mae']:.6f}/{nl['mae']:.6f}; "
            f"own-adversary DRC/LQR MAE {ad['mae']:.6f}/{al['mae']:.6f}; "
            f"own-adversary W2 radius {ad['w2']:.6f}/{al['w2']:.6f}.")
    lines.append("\nThe LQR adversary is LQR-specific but is not a hard-ball "
                 "worst-case distribution. Source: `output/LQR_OWN_ADVERSARY.md`.\n")

    lines.append("### 9.3 Covariance-only shift and heavy tails\n")
    for name, r in cov.items():
        vals = []
        for scale, v in r.items():
            vals.append(f"s={scale:.2f}: DRC={v['drc']['cost']:.6f}, "
                        f"LQR={v['lqr']['cost']:.6f}")
        lines.append(f"- **{name} covariance-only**: " + "; ".join(vals) + ".")
    for name, r in heavy.items():
        vals = []
        for label, v in r.items():
            vals.append(f"{label}: DRC={v['drc']['cost']:.6f}, "
                        f"LQR={v['lqr']['cost']:.6f}")
        lines.append(f"- **{name} heavy-tail**: " + "; ".join(vals) + ".")
    lines.append("\nSources: `output/COVARIANCE_ONLY.md`, `output/HEAVY_TAIL.md`.\n")

    lines.append("### 9.4 g>1 empirical-support sample-size sensitivity\n")
    for name, r in gsens.items():
        lines.append(f"- **{name}**: stable rates " + ", ".join(
            f"g={g}: {v['stable_rate']:.2f}" for g, v in r.items()) + ".")
    lines.append("\nSource: `output/G_SAMPLE_SIZE.md`.\n")

    lines.append("### 9.5 Input-saturation nonlinearity stress test\n")
    for name, r in sat.items():
        lines.append(f"- **{name}**: reference bound "
                     f"{r['reference_bound']:.6f}.")
        for factor, v in r["cases"].items():
            lines.append(
                f"  - factor={factor:.2f}, bound={v['bound']:.6f}: "
                f"DRC sat={v['drc']['sat_fraction']:.4f}, "
                f"LQR sat={v['lqr']['sat_fraction']:.4f}, "
                f"DRC cost={v['drc']['cost']}, LQR cost={v['lqr']['cost']}.")
    lines.append("\nThis is an actuator-saturation stress test on the LTI "
                 "plant, not a full nonlinear-plant validation. "
                 "Source: `output/SATURATION_NONLINEAR.md`.\n")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
#  Standalone entry point
# --------------------------------------------------------------------------- #
def _run(name, fn):
    path = os.path.join(LOG, name + ".log")
    buf = io.StringIO()
    with open(path, "w", encoding="utf-8") as f:
        class Tee:
            def write(self, s):
                f.write(s)
                buf.write(s)
            def flush(self):
                f.flush()
                buf.flush()
        with contextlib.redirect_stdout(Tee()):
            result = fn()
            print(f"{name} completed")
    print(f"[{name}] log: {path}")
    return result


def main():
    print("=== CAAI robustness extensions ===")
    scan = _run("e16_wasserstein_scan", exp_wasserstein_radius_scan)
    lqr_adv = _run("e17_lqr_own_adversary", exp_lqr_own_adversary)
    cov = _run("e18_covariance_only", exp_covariance_only)
    heavy = _run("e19_heavy_tail", exp_heavy_tail)
    gsens = _run("e20_g_sample_size", exp_g_sample_size)
    sat = _run("e21_saturation", exp_saturation)
    print("Completed outputs:")
    for name in sorted(scan):
        print(f"  {name}: lambda grid + radius scan")
    print(f"  covariance scales: {len(cov)} scenarios")
    print(f"  heavy-tail kernels: {len(heavy)} scenarios")
    print(f"  g grid: {len(gsens)} scenarios")
    print(f"  saturation cases: {len(sat)} scenarios")


if __name__ == "__main__":
    main()
