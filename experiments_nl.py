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
    report = {}
    rows = []
    grids = {"humanoid": (0.01, 0.02, 0.05, 0.1, 0.2),
             "microgrid": (0.01, 0.02, 0.05, 0.1)}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        # Shorten only the diagnostic trajectory; reported values are from the
        # same deterministic adversarial map, with the horizon recorded.
        T = 3000 if name == "humanoid" else 10000
        rr = {}
        for z in grids[name]:
            pair = {}
            for ctrl in ("drc", "lqr"):
                res = wx.find_ball_feasible_policy(spec, z, ctrl, T=T)
                pair[ctrl] = res
                sel = res.get("selected")
                if sel is None:
                    rows.append([name, z, ctrl, "infeasible", "—", "—", "—",
                                 "—", "—", "—", "—"])
                else:
                    tr = sel["trajectory"]
                    rows.append([name, z, ctrl, "feasible", f"{sel['lam']:.6f}",
                                 f"{sel['w2_rms']:.6e}", f"{sel['w2_max']:.6e}",
                                 f"{sel['complementarity']:.3e}",
                                 f"{tr['cost']:.6f}", f"{tr['mae']:.6f}",
                                 sel["selection_kind"]])
            rr[z] = pair
        report[name] = rr
    write_table(
        "od_e6_hard_wasserstein_own_adversary",
        ["Scenario", "zeta", "Controller", "status", "lambda",
         "W2 rms", "W2 max", "complementarity", "adversarial cost",
         "adversarial MAE", "selection"], rows,
        notes=("Each controller uses its own affine penalized-dual adversary. "
               "The decoder searches the outer lambda grid and refines a "
               "ball-feasible boundary. A finite-dimensional sampled trajectory "
               "certificate is reported; this is not a proof of global "
               "hard-Bellman optimality."))
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




def exp_g_w2_membership():
    """General-g support sensitivity with an outer-lambda W2 feasibility check."""
    rows, report = [], {}
    lam_grid = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    zeta = 0.1
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        report[name] = {}
        for g in (1, 2, 5, 10, 20, 50):
            w2s, maes, costs, lams, feasible_flags = [], [], [], [], []
            for draw in range(5):
                rng = np.random.default_rng(7000 + 100 * g + draw)
                if g == 1:
                    V = spec["d_bar"].reshape(1, -1)
                else:
                    V = spec["d_bar"][None, :] + spec["d_std"][None, :] *                         rng.standard_normal((g, len(spec["d_bar"])))
                vbar = V.mean(axis=0)
                T = 3000 if name == "humanoid" else 5000
                chosen = None
                for lam in lam_grid:
                    try:
                        sol = core.solve_drc_pi_g(
                            spec["A"], spec["B"], spec["E"], spec["Q"],
                            spec["R"], spec["alpha"], lam, V)
                        pol = {k: sol[k] for k in ("K", "p", "N", "z")}
                        rad = wx.w2_radius_along_trajectory(
                            spec, pol, vbar=vbar, T=T)
                        if rad["w2_max"] <= zeta + 1e-12:
                            chosen = (lam, sol, pol, rad)
                            break
                    except Exception:
                        continue
                if chosen is None:
                    feasible_flags.append(0.0)
                    w2s.append(float("nan")); maes.append(float("nan"))
                    costs.append(float("nan")); lams.append(float("nan"))
                    continue
                lam, sol, pol, rad = chosen
                mc = core.mc_batch(
                    spec["A"], spec["B"], spec["E"], sol["K"], sol["p"],
                    spec["x_ref"], vbar, spec["d_std"], spec["T"], spec["ss"],
                    spec["alpha"], spec["mae_mode"], spec["Q"], spec["R"],
                    seeds=spec["seeds"][:2], x0=spec["x0"])
                feasible_flags.append(1.0)
                lams.append(lam); w2s.append(rad["w2_max"])
                maes.append(mc[0]); costs.append(mc[2])
            valid = np.isfinite(w2s)
            report[name][g] = dict(
                feasible_rate=float(np.mean(feasible_flags)),
                lam_mean=float(np.mean(np.asarray(lams)[np.isfinite(lams)]))
                if np.any(np.isfinite(lams)) else None,
                w2_max=float(np.nanmax(w2s)) if np.any(valid) else None,
                mae=float(np.nanmean(maes)) if np.any(valid) else None,
                cost=float(np.nanmean(costs)) if np.any(valid) else None)
            rows.append([name, g, "5", zeta,
                         f"{np.mean(feasible_flags):.2f}",
                         "—" if not np.any(np.isfinite(lams))
                         else f"{np.mean(np.asarray(lams)[np.isfinite(lams)]):.3f}",
                         "—" if not np.any(valid) else f"{np.nanmax(w2s):.6e}",
                         "—" if not np.any(valid) else f"{np.nanmean(maes):.6f}",
                         "—" if not np.any(valid) else f"{np.nanmean(costs):.6f}"])
    write_table(
        "od_e7_g_w2_membership",
        ["Scenario", "g", "support draws", "zeta", "feasible rate",
         "mean selected lambda", "worst W2 max", "mean MAE", "mean cost"], rows,
        notes=("For each empirical support draw, lambda is chosen from a grid "
               "so that the policy's own atomwise worst-case distribution "
               "satisfies W2<=zeta on the sampled adversarial trajectory. "
               "Failure to find a feasible lambda is reported, not hidden."))
    log("od_e7_g_w2_membership", report)
    return report


def exp_nonlinear_validation():
    """Run the unified nonlinear/contact/saturation validation.

    The implementation and formal table writer live in
    ``nonlinear_validation.py``.  Keeping the experiment there avoids a second
    table-generation path that could drift from the rollout code.
    """
    r = nl.run_nonlinear_validation(write_outputs=True)
    log("od_e8_nonlinear_validation",
        {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in r.items()})
    return r


def exp_tail_only_validation():
    """Run the formal tail-only/heavy-tail comparison table."""
    r = nl.run_tail_only_validation(write_outputs=True)
    log("od_e9_tail_only", {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in r.items()})
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
    result["tail_only"] = exp_tail_only_validation()
    result["figures"] = figures_ofb.make_all()
    result["wall_seconds"] = time.time() - t0
    log("od_all_summary", result)
    print(json.dumps({"wall_seconds": result["wall_seconds"],
                      "tables": sorted(x for x in os.listdir(TBL)
                                       if x.startswith("od_") or
                                       x.startswith("e18_") or
                                       x.startswith("e19_") or
                                       x.startswith("e20_") or
                                       x.startswith("e21_") or
                                       x.startswith("od_e8_") or
                                       x.startswith("od_e9_"))}, indent=2))


if __name__ == "__main__":
    main()
