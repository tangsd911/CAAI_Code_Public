#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent OD simulation suite: output feedback, Algorithm 4, hard W2, robustness."""

import os
import csv
import json
import time
import numpy as np

import models
import drc_core as core
import algorithm4
import data_driven as dd
import wasserstein_experiments as wx
import robustness_extensions as rx
import nonlinear_validation as nl
import figures_ofb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")
LOG = os.path.join(OUT, "logs")
for d in (OUT, TBL, LOG):
    os.makedirs(d, exist_ok=True)


def write_table(name, headers, rows, notes=None):
    with open(os.path.join(TBL, name + ".csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(["" if x is None else str(x) for x in row])
    with open(os.path.join(TBL, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("" if x is None else str(x)
                                      for x in row) + " |\n")
        if notes:
            f.write("\n" + notes + "\n")


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


def log(name, obj):
    with open(os.path.join(LOG, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, default=float)


def _solve(spec):
    d = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                          spec["R"], spec["alpha"], spec["lam"], spec["d_bar"])
    Klqr, rho, it, conv = core.solve_discounted_lqr(
        spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"])
    return d, Klqr, rho, it, conv


def _full(spec, K, p, d_mean=None, d_std=None):
    return core.mc_batch(
        spec["A"], spec["B"], spec["E"], K, p, spec["x_ref"],
        spec["d_bar"] if d_mean is None else d_mean,
        spec["d_std"] if d_std is None else d_std,
        spec["T"], spec["ss"], spec["alpha"], spec["mae_mode"],
        spec["Q"], spec["R"], seeds=spec["seeds"], x0=spec["x0"])


def _ofb(spec, K, p, d_mean=None, d_std=None):
    return core.mc_batch_output_feedback_kf(
        spec["A"], spec["B"], spec["C"], spec["E"], K, p, spec["x_ref"],
        spec["d_bar"] if d_mean is None else d_mean,
        spec["d_std"] if d_std is None else d_std,
        spec["T"], spec["ss"], spec["alpha"], spec["mae_mode"],
        spec["Q"], spec["R"], seeds=spec["seeds"], x0=spec["x0"])


def exp_ofb_main():
    rows, report = [], {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        d, Klqr, _, _, _ = _solve(spec)
        cases = {
            "DRC": (d["K"], d["p"]),
            "LQR": (-Klqr, np.zeros(spec["B"].shape[1])),
        }
        report[name] = {}
        for label, (K, p) in cases.items():
            full = _full(spec, K, p)
            ofb = _ofb(spec, K, p)
            report[name][label] = dict(full=full, ofb=ofb)
            rows.append([name, label, "full state", f"{full[0]:.6f}",
                         f"{full[1]:.6f}", f"{full[2]:.6f}", f"{full[3]:.6f}",
                         "—", "—", "—", "0"])
            rows.append([name, label, "unknown-history OFB (KF)",
                         f"{ofb['mae_mean']:.6f}", f"{ofb['mae_std']:.6f}",
                         f"{ofb['cost_mean']:.6f}", f"{ofb['cost_std']:.6f}",
                         f"{ofb['recon_rmse_mean']:.3e}",
                         f"{ofb['observer_rho']:.6f}",
                         f"{ofb['max_state']:.3e}", str(ofb["n_diverged"])])
    write_table(
        "od_e1_main_output_feedback",
        ["Scenario", "Controller", "Feedback", "MAE", "MAE std",
         "Discounted cost", "Cost std", "Reconstruction RMSE",
         "Observer rho", "Max state", "Diverged seeds"], rows,
        notes=("OFB uses y=Cx and a steady-state Kalman observer with no realized "
               "disturbance-history input. The full-state row uses the same "
               "20 seeds and x0. OFB is therefore the primary unknown-history "
               "closed-loop result; the exact static window requires d history."))
    log("od_e1_main_output_feedback", report)
    return report


def exp_ofb_shift():
    rows, report = [], {}
    hum = models.humanoid_spec()
    dh, Kh, _, _, _ = _solve(hum)
    report["humanoid"] = {}
    for sh in (0.0, 0.5, 1.0, 2.0, 3.0):
        dm = hum["d_bar"] * (1.0 + sh)
        d = _ofb(hum, dh["K"], dh["p"], d_mean=dm)
        l = _ofb(hum, -Kh, np.zeros(3), d_mean=dm)
        report["humanoid"][sh] = dict(drc=d, lqr=l)
        rows.append(["humanoid", sh, "DRC", f"{d['mae_mean']:.6f}",
                     f"{d['mae_std']:.6f}", f"{d['cost_mean']:.6f}",
                     f"{d['recon_rmse_mean']:.3e}", str(d["n_diverged"])])
        rows.append(["humanoid", sh, "LQR", f"{l['mae_mean']:.6f}",
                     f"{l['mae_std']:.6f}", f"{l['cost_mean']:.6f}",
                     f"{l['recon_rmse_mean']:.3e}", str(l["n_diverged"])])

    mic = models.microgrid_spec()
    dm, Km, _, _, _ = _solve(mic)
    shifts = [("nominal", mic["d_bar"], mic["d_std"]),
              ("mean_only_x1.5", 1.5 * mic["d_bar"], mic["d_std"]),
              ("mean_x1.5_var_x2", 1.5 * mic["d_bar"],
               np.sqrt(2.0) * mic["d_std"])]
    report["microgrid"] = {}
    for label, mean, std in shifts:
        d = _ofb(mic, dm["K"], dm["p"], d_mean=mean, d_std=std)
        l = _ofb(mic, -Km, np.zeros(2), d_mean=mean, d_std=std)
        report["microgrid"][label] = dict(drc=d, lqr=l)
        rows.append(["microgrid", label, "DRC", f"{d['mae_mean']:.6f}",
                     f"{d['mae_std']:.6f}", f"{d['cost_mean']:.6f}",
                     f"{d['recon_rmse_mean']:.3e}", str(d["n_diverged"])])
        rows.append(["microgrid", label, "LQR", f"{l['mae_mean']:.6f}",
                     f"{l['mae_std']:.6f}", f"{l['cost_mean']:.6f}",
                     f"{l['recon_rmse_mean']:.3e}", str(l["n_diverged"])])
    write_table(
        "od_e2_output_feedback_shift",
        ["Scenario", "Shift", "Controller", "MAE", "MAE std",
         "Discounted cost", "Reconstruction RMSE", "Diverged seeds"], rows,
        notes=("All rows use unknown-history output feedback with the same 20 "
               "seeds. The estimator uses the nominal d_bar even under shifted "
               "disturbances; therefore this is a genuine distribution-shift "
               "test, not a re-tuned observer."))
    log("od_e2_output_feedback_shift", report)
    return report


def _affine_bias(spec):
    return np.linalg.pinv(spec["B"]) @ ((np.eye(spec["A"].shape[0]) - spec["A"])
                                         @ spec["x_ref"] - spec["E"] @ spec["d_bar"])


def exp_ofb_ablation():
    rows, report = [], {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        d, Klqr, _, _, _ = _solve(spec)
        variants = {
            "LQR pure": (-Klqr, np.zeros(spec["B"].shape[1])),
            "Affine nominal LQR": (-Klqr, _affine_bias(spec)),
            "DRC without bias": (d["K"], np.zeros(spec["B"].shape[1])),
            "Affine DRC": (d["K"], d["p"]),
        }
        report[name] = {}
        for label, (K, p) in variants.items():
            r = _ofb(spec, K, p)
            report[name][label] = r
            rows.append([name, label, f"{r['mae_mean']:.6f}",
                         f"{r['mae_std']:.6f}", f"{r['cost_mean']:.6f}",
                         f"{r['cost_std']:.6f}",
                         f"{r['recon_rmse_mean']:.3e}",
                         str(r["n_diverged"])])
    write_table(
        "od_e3_output_feedback_ablation",
        ["Scenario", "Controller", "MAE", "MAE std", "Discounted cost",
         "Cost std", "Reconstruction RMSE", "Diverged seeds"], rows,
        notes=("This is an internal ablation, not a comparison with an "
               "independent robust/DRO solver. All controllers use the same "
               "unknown-history observer, seeds, weights, and rollout horizon."))
    log("od_e3_output_feedback_ablation", report)
    return report


def exp_algorithm4():
    report = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        ols = algorithm4.run_data_driven_algorithm4(
            spec, N=20000, seed=11, rounds=3, expl_scale=0.25,
            use_iv=False, state_obs_noise_std=0.0,
            input_obs_noise_std=0.0, dist_obs_noise_std=0.0,
            target_obs_noise_std=1e-4)
        iv = algorithm4.run_data_driven_algorithm4(
            spec, N=20000, seed=11, rounds=2, expl_scale=0.25,
            use_iv=True, n_lags=3, state_obs_noise_std=0.0,
            input_obs_noise_std=0.0, dist_obs_noise_std=0.0,
            target_obs_noise_std=1e-4)
        report[name] = dict(ols=ols, iv=iv)
    # Slope scans use a controlled disturbance-measurement noise so the
    # statistical rate is identifiable; noise-free synthetic data would give
    # machine-precision errors and an undefined slope.
    report["slopes"] = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        report["slopes"][name] = {
            "ols": algorithm4.alg4_slope_scan(
                spec, Ns=(1000, 2000, 4000, 8000, 16000), seeds=range(5),
                use_iv=False, target_obs_noise_std=1e-3, rounds=2),
            "iv": algorithm4.alg4_slope_scan(
                spec, Ns=(1000, 2000, 4000, 8000, 16000), seeds=range(5),
                use_iv=True, n_lags=3, target_obs_noise_std=1e-3, rounds=2),
        }
    rows = []
    for name in ("humanoid", "microgrid"):
        for method in ("ols", "iv"):
            r = report[name][method]
            e = r["errors"]
            rows.append([name, method.upper(), r["N"], str(r["accepted"]),
                         f"{e['eA']:.4e}", f"{e['eB']:.4e}",
                         f"{e['eE']:.4e}", f"{e['eK']:.4e}",
                         f"{e['ep']:.4e}", f"{r['true_rho_G']:.6f}",
                         r["sdpi"]["stop_reason"],
                         f"{r['identifiability']['rank']}/{r['identifiability']['reg_dim']}",
                         f"{r['identifiability']['rank_instruments']}/{r['identifiability']['instrument_dim']}"])
    write_table(
        "od_e4_algorithm4_end_to_end",
        ["Scenario", "Method", "N", "accepted", "eA", "eB", "eE", "eK",
         "ep", "true rho(G)", "SDPI stop", "rank(W)", "rank(Z)"], rows,
        notes=("Fresh data are collected each outer round. OLS identifies all "
               "of A,B,E from measured state/input/disturbance data; IV uses "
               "lagged instruments. IV is reported as an implemented robustness "
               "diagnostic, not as a claim that it always outperforms OLS."))
    slope_rows = []
    for name in ("humanoid", "microgrid"):
        for method in ("ols", "iv"):
            s = report["slopes"][name][method]["slopes"]
            slope_rows.append([name, method.upper()] +
                              [f"{s[k]:+.4f}" for k in
                               ("eA", "eB", "eE", "eK", "ep", "eN", "ez")])
    write_table(
        "od_e5_algorithm4_rates",
        ["Scenario", "Method", "slope eA", "slope eB", "slope eE",
         "slope eK", "slope ep", "slope eN", "slope ez"], slope_rows,
        notes=("Slopes are finite-sample log-log regressions at N=1000,2000,"
               "4000,8000,16000 with 5 seeds and independent next-state "
               "measurement noise 1e-3. A value near -0.5 is consistent with "
               "the stated O(N^-1/2) asymptotic order; deviations are reported, "
               "not hidden."))
    log("od_e4_e5_algorithm4", report)
    return report


def exp_hard_wasserstein():
    """Outer-lambda hard-ball feasibility scan for zeta in both scenarios.

    DRC and LQR are evaluated separately, each with its own penalized-dual
    affine adversary.  Feasibility is the sampled finite-horizon condition
    W2_max <= zeta on the induced atomwise worst-case distribution.
    """
    report = {}
    rows = []
    audit_rows = []
    grids = {
        "humanoid": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        "microgrid": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    }
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        V = spec["d_bar"].reshape(1, -1)
        T = 1000 if name == "humanoid" else 2000
        rr = {}
        for z in grids[name]:
            pair = {}
            for ctrl in ("drc", "lqr"):
                res = wx.find_ball_feasible_policy(
                    spec, z, ctrl, T=T, V=V, max_iter=5000)
                pair[ctrl] = res
                sel = res.get("selected")
                if sel is None:
                    rows.append([
                        name, f"{z:.5f}", ctrl.upper(), "infeasible",
                        "—", "—", "—", "—", "—", "—", "—", "—",
                        "—", "—", "—", "—", "—", str(T),
                        "semicolon-separated counts: " + "; ".join(
                            f"{k}={v}" for k, v in sorted(
                                res["failure_counts"].items())),
                        wx.HARD_BALL_SCOPE,
                    ])
                else:
                    rows.append([
                        name, f"{z:.5f}", ctrl.upper(), "feasible",
                        f"{sel['lam']:.6f}",
                        "—" if sel.get("boundary_lambda") is None
                        else f"{sel['boundary_lambda']:.6f}",
                        f"{sel['w2_rms']:.6e}", f"{sel['w2_max']:.6e}",
                        f"{sel['atom_w2_max']:.6e}",
                        f"{sel['constraint_slack']:.3e}",
                        f"{sel['complementarity']:.3e}",
                        str(bool(sel['constraint_active'])),
                        "—" if sel.get("boundary_complementarity") is None
                        else f"{sel['boundary_complementarity']:.3e}",
                        "—" if sel.get("boundary_active") is None
                        else str(bool(sel['boundary_active'])),
                        f"{sel['trajectory']['cost']:.6f}",
                        f"{sel['trajectory']['mae']:.6f}",
                        f"{sel['rho_G']:.6f}", str(T),
                        "none", wx.HARD_BALL_SCOPE,
                    ])
                for cand in res["candidates"]:
                    audit_rows.append([
                        name, f"{z:.5f}", ctrl.upper(),
                        f"{cand['lam']:.9g}", cand["status"],
                        cand["failure_reason"],
                        str(bool(cand.get("solver_converged", False))),
                        str(cand.get("solver_iters", "—")),
                        "—" if cand.get("rho_G") is None
                        else f"{cand['rho_G']:.6e}",
                        "—" if cand.get("H_uu_min") is None
                        else f"{cand['H_uu_min']:.6e}",
                        "—" if cand.get("H_ww_min") is None
                        else f"{cand['H_ww_min']:.6e}",
                        "—" if cand.get("w2_rms") is None
                        else f"{cand['w2_rms']:.6e}",
                        "—" if cand.get("w2_max") is None
                        else f"{cand['w2_max']:.6e}",
                        "—" if cand.get("constraint_slack") is None
                        else f"{cand['constraint_slack']:.3e}",
                        str(bool(cand.get("constraint_active", False))),
                        "—" if cand.get("complementarity") is None
                        else f"{cand['complementarity']:.3e}",
                        "—" if cand.get("outer_objective") is None
                        else f"{cand['outer_objective']:.6e}",
                        str(T), wx.HARD_BALL_SCOPE,
                    ])
            rr[z] = pair
        report[name] = rr
    write_table(
        "od_e6_hard_wasserstein_own_adversary",
        ["Scenario", "zeta", "Controller", "status", "selected lambda",
         "boundary lambda", "W2 rms", "W2 max", "atom W2 max",
         "constraint slack", "selected complementarity", "selected active",
         "boundary complementarity", "boundary active",
         "own-adversary cost", "own-adversary MAE", "rho_G", "horizon",
         "failure reasons", "scope"], rows,
        notes=("Each controller uses its own affine penalized-dual adversary. "
               "The outer lambda grid is searched independently for every row; "
               "selected lambda minimizes the finite-horizon Lagrangian among "
               "feasible sampled candidates, while boundary lambda is the "
               "refined infeasible-to-feasible transition. Membership is "
               "W2_max<=zeta at every sampled trajectory point. This is a "
               "finite-horizon sampled certificate, not a global Bellman proof."))
    write_table(
        "od_e6_lambda_candidate_audit",
        ["Scenario", "zeta", "Controller", "lambda", "status",
         "failure reason", "solver converged", "iterations", "rho_G",
         "H_uu min", "H_ww min", "W2 rms", "W2 max", "constraint slack",
         "constraint active", "complementarity", "finite-horizon Lagrangian",
         "horizon", "scope"], audit_rows,
        notes=("Every row is computed by wasserstein_experiments.py. Candidate "
               "failure reasons include solver exceptions, non-convergence, "
               "instability, non-positive input Hessian, non-negative adversary "
               "Hessian, rollout divergence, atom-identity failure, and "
               "W2>zeta."))
    log("od_e6_hard_wasserstein", report)
    return report


def exp_robustness_suite():
    t0 = time.time()
    cov = rx.exp_covariance_only()
    heavy = rx.exp_heavy_tail()
    g = rx.exp_g_sample_size()
    sat = rx.exp_saturation()
    out = dict(covariance_only=cov, heavy_tail=heavy, g_sample_size=g,
               saturation=sat, wall_seconds=time.time() - t0)
    log("od_e7_robustness_suite", out)
    return out


def _support_draw(spec, g, draw, scenario_index):
    """Generate a reproducible empirical disturbance support."""
    if g == 1:
        return spec["d_bar"].reshape(1, -1).copy()
    rng = np.random.default_rng(91000 + 1000 * scenario_index + 100 * g + draw)
    return (spec["d_bar"][None, :]
            + spec["d_std"][None, :] * rng.standard_normal(
                (g, spec["d_bar"].size)))


def exp_g_w2_membership():
    """Outer-lambda hard-ball search for g=1,2,5,10,20 support draws.

    Both controllers are fitted to the same empirical support, but each uses
    its own adversary.  Every support draw gets a separate membership check;
    infeasible draws and candidate failure reasons are retained.
    """
    rows, detail_rows = [], []
    report = {}
    g_grid = (1, 2, 5, 10, 20)
    n_draws = 10
    zeta = 0.1
    for s_idx, (name, spec) in enumerate(
            (("humanoid", models.humanoid_spec()),
             ("microgrid", models.microgrid_spec()))):
        T = 1000 if name == "humanoid" else 2000
        report[name] = {}
        for g in g_grid:
            report[name][g] = {}
            for ctrl in ("drc", "lqr"):
                records = []
                for draw in range(n_draws):
                    V = _support_draw(spec, g, draw, s_idx)
                    res = wx.find_ball_feasible_policy(
                        spec, zeta, ctrl, T=T, V=V, max_iter=5000)
                    sel = res.get("selected")
                    rec = dict(
                        draw=draw, feasible=bool(res["feasible"]),
                        support_mean_shift=float(np.linalg.norm(
                            V.mean(axis=0) - spec["d_bar"])),
                        failure_counts=res["failure_counts"],
                        selected=None,
                    )
                    if sel is not None:
                        rec["selected"] = {
                            "lam": float(sel["lam"]),
                            "boundary_lambda": (
                                None if sel.get("boundary_lambda") is None
                                else float(sel["boundary_lambda"])),
                            "w2_rms": float(sel["w2_rms"]),
                            "w2_max": float(sel["w2_max"]),
                            "atom_w2_max": float(sel["atom_w2_max"]),
                            "constraint_slack": float(sel["constraint_slack"]),
                            "complementarity": float(sel["complementarity"]),
                            "constraint_active": bool(sel["constraint_active"]),
                            "boundary_complementarity": (
                                None if sel.get("boundary_complementarity") is None
                                else float(sel["boundary_complementarity"])),
                            "boundary_active": bool(sel.get("boundary_active", False)),
                            "mae": float(sel["trajectory"]["mae"]),
                            "cost": float(sel["trajectory"]["cost"]),
                            "rho_G": float(sel["rho_G"]),
                            "atom_dispersion": float(sel["atom_dispersion"]),
                            "mean_identity_error": float(
                                sel["mean_identity_error"]),
                        }
                    records.append(rec)
                    failures = "; ".join(
                        f"{k}={v}" for k, v in sorted(
                            res["failure_counts"].items()))
                    if sel is None:
                        detail_rows.append([
                            name, g, ctrl.upper(), draw, f"{zeta:.5f}",
                            "infeasible", "—", "—", "—", "—", "—",
                            "—", "—", "—", "—", "—", failures, str(T),
                            wx.HARD_BALL_SCOPE,
                        ])
                    else:
                        detail_rows.append([
                            name, g, ctrl.upper(), draw, f"{zeta:.5f}",
                            "feasible", f"{sel['lam']:.6f}",
                            "—" if sel.get("boundary_lambda") is None
                            else f"{sel['boundary_lambda']:.6f}",
                            f"{sel['w2_rms']:.6e}", f"{sel['w2_max']:.6e}",
                            f"{sel['constraint_slack']:.3e}",
                            str(bool(sel['constraint_active'])),
                            f"{sel['complementarity']:.3e}",
                            "—" if sel.get("boundary_complementarity") is None
                            else f"{sel['boundary_complementarity']:.3e}",
                            str(bool(sel.get("boundary_active", False))),
                            f"{sel['trajectory']['mae']:.6f}",
                            f"{sel['trajectory']['cost']:.6f}",
                            f"{sel['rho_G']:.6f}", str(T),
                            wx.HARD_BALL_SCOPE,
                        ])

                good = [r for r in records if r["selected"] is not None]
                reason_counts = {}
                for r in records:
                    for reason, count in r["failure_counts"].items():
                        reason_counts[reason] = reason_counts.get(reason, 0) + count
                if good:
                    sel_lams = np.asarray([r["selected"]["lam"] for r in good])
                    bnd_lams = np.asarray([
                        r["selected"]["boundary_lambda"] for r in good
                        if r["selected"]["boundary_lambda"] is not None])
                    w2r = np.asarray([r["selected"]["w2_rms"] for r in good])
                    w2m = np.asarray([r["selected"]["w2_max"] for r in good])
                    comps = np.asarray([r["selected"]["complementarity"]
                                        for r in good])
                    bcomps = np.asarray([
                        r["selected"]["boundary_complementarity"] for r in good
                        if r["selected"]["boundary_complementarity"] is not None])
                    maes = np.asarray([r["selected"]["mae"] for r in good])
                    costs = np.asarray([r["selected"]["cost"] for r in good])
                    rhs = np.asarray([r["selected"]["rho_G"] for r in good])
                    active_rate = float(np.mean([
                        r["selected"]["constraint_active"] for r in good]))
                    bactive_rate = float(np.mean([
                        r["selected"]["boundary_active"] for r in good]))
                    summary = dict(
                        feasible_rate=float(len(good) / len(records)),
                        mean_selected_lambda=float(np.mean(sel_lams)),
                        min_selected_lambda=float(np.min(sel_lams)),
                        max_selected_lambda=float(np.max(sel_lams)),
                        mean_boundary_lambda=(None if len(bnd_lams) == 0
                                              else float(np.mean(bnd_lams))),
                        mean_w2_rms=float(np.mean(w2r)),
                        worst_w2_max=float(np.max(w2m)),
                        selected_active_rate=active_rate,
                        boundary_active_rate=bactive_rate,
                        mean_selected_complementarity=float(np.mean(comps)),
                        max_selected_complementarity=float(np.max(comps)),
                        mean_boundary_complementarity=(
                            None if len(bcomps) == 0 else float(np.mean(bcomps))),
                        max_boundary_complementarity=(
                            None if len(bcomps) == 0 else float(np.max(bcomps))),
                        mean_mae=float(np.mean(maes)),
                        mean_cost=float(np.mean(costs)),
                        max_rho_G=float(np.max(rhs)),
                        failure_counts=reason_counts,
                    )
                    rows.append([
                        name, str(g), ctrl.upper(), str(n_draws),
                        f"{zeta:.5f}", "feasible",
                        f"{summary['feasible_rate']:.2f}",
                        f"{summary['mean_selected_lambda']:.6f}",
                        f"{summary['min_selected_lambda']:.6f}",
                        f"{summary['max_selected_lambda']:.6f}",
                        "—" if summary["mean_boundary_lambda"] is None
                        else f"{summary['mean_boundary_lambda']:.6f}",
                        f"{summary['mean_w2_rms']:.6e}",
                        f"{summary['worst_w2_max']:.6e}",
                        f"{summary['selected_active_rate']:.2f}",
                        f"{summary['boundary_active_rate']:.2f}",
                        f"{summary['mean_selected_complementarity']:.3e}",
                        f"{summary['max_selected_complementarity']:.3e}",
                        "—" if summary["mean_boundary_complementarity"] is None
                        else f"{summary['mean_boundary_complementarity']:.3e}",
                        "—" if summary["max_boundary_complementarity"] is None
                        else f"{summary['max_boundary_complementarity']:.3e}",
                        f"{summary['mean_mae']:.6f}",
                        f"{summary['mean_cost']:.6f}",
                        f"{summary['max_rho_G']:.6f}",
                        "; ".join(f"{k}={v}" for k, v in sorted(
                            reason_counts.items())),
                        str(T), wx.HARD_BALL_SCOPE,
                    ])
                else:
                    summary = dict(
                        feasible_rate=0.0, failure_counts=reason_counts)
                    rows.append([
                        name, str(g), ctrl.upper(), str(n_draws),
                        f"{zeta:.5f}", "infeasible", "0.00", "—", "—", "—",
                        "—", "—", "—", "0.00", "0.00", "—", "—", "—", "—",
                        "—", "—", "—",
                        "; ".join(f"{k}={v}" for k, v in sorted(
                            reason_counts.items())),
                        str(T), wx.HARD_BALL_SCOPE,
                    ])
                report[name][g][ctrl] = dict(
                    summary=summary,
                    records=records,
                )
    write_table(
        "od_e7_g_w2_membership",
        ["Scenario", "g", "Controller", "support draws", "zeta", "status",
         "feasible rate", "mean selected lambda", "min selected lambda",
         "max selected lambda", "mean boundary lambda", "mean W2 rms",
         "worst W2 max", "selected active rate", "boundary active rate",
         "mean selected complementarity", "max selected complementarity",
         "mean boundary complementarity", "max boundary complementarity",
         "mean MAE", "mean cost", "max rho_G", "candidate failure counts",
         "horizon", "scope"], rows,
        notes=("For every support draw and both controllers, lambda is searched "
               "on the outer grid and the infeasible-to-feasible transition is "
               "refined. The selected lambda minimizes the sampled finite-horizon "
               "Lagrangian among ball-feasible candidates; boundary lambda is "
               "reported for complementary-slackness diagnostics. Membership is "
               "checked on the explicit atomwise worst-case distribution "
               "W2_max<=zeta. This is not a global Bellman proof."))
    write_table(
        "od_e7_g_w2_support_draws",
        ["Scenario", "g", "Controller", "support draw", "zeta", "status",
         "selected lambda", "boundary lambda", "W2 rms", "W2 max",
         "constraint slack", "selected active", "selected complementarity",
         "boundary complementarity", "boundary active", "MAE", "cost",
         "rho_G", "horizon", "scope"], detail_rows,
        notes=("Each row is one empirical support draw. Failure counts list all "
               "outer-lambda candidates that did not produce a ball-feasible "
               "stable policy. No failed draw is dropped."))
    log("od_e7_g_w2_membership", report)
    return report


def exp_nonlinear_validation():
    """Run the reduced nonlinear/contact/saturation/operating-range validation."""
    r = nl.run_nonlinear_validation()
    rows = []
    for (scenario, controller), v in r.items():
        rows.append([scenario, controller,
                     "—" if v["mae"] is None else f"{v['mae']:.6f}",
                     "—" if v["cost"] is None else f"{v['cost']:.6f}",
                     f"{v['max_state']:.6f}", f"{v['saturated']:.4f}",
                     "—" if v["contact_fraction"] is None
                     else f"{v['contact_fraction']:.4f}",
                     "—" if v["freq_violation"] is None
                     else f"{v['freq_violation']:.4f}",
                     str(v["n_diverged"])])
    write_table(
        "od_e8_nonlinear_contact_saturation",
        ["Scenario", "Controller", "MAE", "Discounted cost", "Max |state|",
         "Saturation fraction", "Contact/limit fraction",
         "Frequency violation fraction", "Diverged seeds"], rows,
        notes=("Humanoid uses gravity, friction, torque saturation, and compliant "
               "joint-limit contact; microgrid uses quadratic frequency damping, "
               "governor saturation, and frequency-range violation. These are "
               "reduced nonlinear stress tests, not high-fidelity plant models."))
    log("od_e8_nonlinear_validation", {f"{k[0]}::{k[1]}": v for k, v in r.items()})
    return r


def main():
    t0 = time.time()
    result = {}
    result["output_feedback_main"] = exp_ofb_main()
    result["output_feedback_shift"] = exp_ofb_shift()
    result["output_feedback_ablation"] = exp_ofb_ablation()
    result["algorithm4"] = exp_algorithm4()
    result["hard_wasserstein"] = exp_hard_wasserstein()
    result["robustness"] = exp_robustness_suite()
    result["g_w2_membership"] = exp_g_w2_membership()
    result["nonlinear"] = exp_nonlinear_validation()
    result["figures"] = figures_ofb.make_all()
    result["wall_seconds"] = time.time() - t0
    log("od_all_summary", result)
    print(json.dumps({"wall_seconds": result["wall_seconds"],
                      "tables": sorted(x for x in os.listdir(TBL)
                                       if x.startswith("od_") or
                                       x.startswith("e18_") or
                                       x.startswith("e19_") or
                                       x.startswith("e20_") or
                                       x.startswith("e21_"))}, indent=2))


if __name__ == "__main__":
    main()
