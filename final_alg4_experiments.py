#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the Algorithm 4 / IV / SDPI tables under ``output/tables``."""

import csv
import json
import os
import time
import numpy as np

import algorithm4 as a4
import data_driven as dd
import drc_core as core
import models
import sdpi

HERE = os.path.dirname(os.path.abspath(__file__))
TBL = os.path.join(HERE, "output", "tables")
LOG = os.path.join(HERE, "output", "logs")
os.makedirs(TBL, exist_ok=True)
os.makedirs(LOG, exist_ok=True)


def _mean_std(vals):
    x = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                   dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std())


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


def write_table(name, headers, rows, notes=None):
    with open(os.path.join(TBL, name + ".csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(["" if x is None else x for x in row])
    with open(os.path.join(TBL, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("" if x is None else str(x)
                                      for x in row) + " |\n")
        if notes:
            f.write("\n" + notes + "\n")


def run_end_to_end(N=12000, seeds=range(5), rounds=2):
    runs = {}
    rows = []
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        runs[name] = {}
        for iv in (False, True):
            method = "IV/2SLS" if iv else "OLS"
            rr = []
            for seed in seeds:
                r = a4.run_data_driven_algorithm4(
                    spec, N=N, seed=seed, rounds=rounds, use_iv=iv,
                    n_lags=1, instrument_mode="probe",
                    dist_probe_scale=0.5, expl_scale=0.75)
                rr.append(r)
            runs[name][method] = rr
            keys = ("eA", "eB", "eE", "eN", "ez", "eK", "ep",
                    "eN_policy", "ez_policy")
            stats = {k: _mean_std([r["errors"].get(k) for r in rr]) for k in keys}
            ef = _mean_std([r["identification"].get("E_first_stage_min_F")
                            for r in rr])
            nf = _mean_std([r["identification"].get("N_first_stage_min_F")
                            for r in rr])
            rhos = np.asarray([r["true_rho_G"] for r in rr], dtype=float)
            certs = [r["certificate"]["E_status"] if r["certificate"] else "none"
                     for r in rr]
            n_certs = [r["certificate"]["N_status"] if r["certificate"] else "none"
                       for r in rr]
            stops = sorted(set(h.get("sdpi_stop", "not_run")
                                for r in rr for h in r["history"]))
            rows.append([
                name, method, len(rr), N, rounds,
                f"{certs.count('identifiable')}/{len(certs)}",
                f"{n_certs.count('identifiable')}/{len(n_certs)}",
                "—" if not np.isfinite(ef[0]) else f"{ef[0]:.2f}±{ef[1]:.2f}",
                "—" if not np.isfinite(nf[0]) else f"{nf[0]:.2f}±{nf[1]:.2f}",
                f"{stats['eA'][0]:.3e}±{stats['eA'][1]:.3e}",
                f"{stats['eB'][0]:.3e}±{stats['eB'][1]:.3e}",
                f"{stats['eE'][0]:.3e}±{stats['eE'][1]:.3e}",
                f"{stats['eN'][0]:.3e}±{stats['eN'][1]:.3e}",
                f"{stats['ez'][0]:.3e}±{stats['ez'][1]:.3e}",
                f"{stats['eK'][0]:.3e}±{stats['eK'][1]:.3e}",
                f"{stats['ep'][0]:.3e}±{stats['ep'][1]:.3e}",
                f"{stats['eN_policy'][0]:.3e}±{stats['eN_policy'][1]:.3e}",
                f"{np.nanmean(rhos):.6f}",
                f"{np.nanmin(rhos):.6f}",
                f"{sum(bool(r['accepted']) for r in rr)}/{len(rr)}",
                ";".join(stops)])
    write_table(
        "od_e4_algorithm4_end_to_end",
        ["Scenario", "Method", "Seeds", "N", "Outer rounds",
         "E identifiable", "N identifiable", "E first-stage F", "N first-stage F",
         "eA", "eB", "eE", "eN", "ez", "eK", "ep", "eN policy",
         "true rho mean", "true rho min", "post-hoc stable", "SDPI stops"],
        rows,
        notes=("End-to-end fresh-data Algorithm 4 with input exploration and an "
               "independent disturbance-channel probe q. eN is the direct "
               "adversary-block identification error; eN policy is measured after "
               "SDPI. The true-model radius is computed only after each update as a "
               "post-hoc diagnostic and does not gate the algorithm. Values are "
               "mean±std over independent seeds; no table entry is used as an "
               "algorithm input."))
    return runs


def run_rates(Ns=(4000, 8000, 16000, 32000), seeds=range(5),
              rounds=1):
    out = {}
    rows = []
    point_rows = []
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        out[name] = {}
        for iv in (False, True):
            method = "IV/2SLS" if iv else "OLS"
            r = a4.alg4_slope_scan(
                spec, Ns=Ns, seeds=seeds, use_iv=iv, rounds=rounds,
                dist_probe_scale=0.5, expl_scale=1.0)
            out[name][method] = r
            sl = r["slopes"]
            rows.append([name, method,
                         f"{sl['eA']:.3f}", f"{sl['eB']:.3f}",
                         f"{sl['eE']:.3f}", f"{sl['eP']:.3f}",
                         f"{sl['eK']:.3f}", f"{sl['ep']:.3f}",
                         f"{sl['eN_policy']:.3f}", f"{sl['ez_policy']:.3f}"])
            for N, v in r["per_n"].items():
                point_rows.append([
                    name, method, N,
                    f"{v['mean']['eA']:.6e}", f"{v['std']['eA']:.6e}",
                    f"{v['mean']['eB']:.6e}", f"{v['mean']['eE']:.6e}",
                    f"{v['mean']['eP']:.6e}", f"{v['mean']['eK']:.6e}",
                    f"{v['mean']['ep']:.6e}",
                    f"{v['mean']['eN_policy']:.6e}"])
    write_table(
        "od_e5_algorithm4_rates",
        ["Scenario", "Method", "slope eA", "slope eB", "slope eE",
         "slope eP", "slope eK", "slope ep",
         "slope eN policy", "slope ez policy"], rows,
        notes=("Finite-sample log-log slopes for N=4000--32000, five seeds, one "
               "fixed-policy evaluation/improvement round, matching the local "
               "one-step theorem rather than claiming a global-rate result for "
               "the outer loop. The designed IV uses the independent "
               "disturbance probe as the excluded instrument; OLS is kept as a "
               "biased comparison. Deviations from -1/2 are reported, not "
               "smoothed or constrained."))
    write_table(
        "od_e5_algorithm4_rates_points",
        ["Scenario", "Method", "N", "mean eA", "std eA", "mean eB",
         "mean eE", "mean eP", "mean eK", "mean ep", "mean eN policy"],
        point_rows,
        notes="Point estimates underlying od_e5_algorithm4_rates.")
    return out


def run_adversary_rates(Ns=(4000, 8000, 16000, 32000), seeds=range(5)):
    """Rate check for the affine adversary block under a fixed policy.

    The policy is the model-based DRC reference used only to generate data.
    This isolates estimation of ``d-q=Nx+z+xi`` from the outer policy loop.
    """
    rows = []
    out = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        opt = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                                spec["R"], spec["alpha"], spec["lam"],
                                spec["d_bar"])
        out[name] = {}
        for use_iv, method in ((False, "OLS"), (True, "IV/2SLS")):
            means = {N: [] for N in Ns}
            for N in Ns:
                for seed in seeds:
                    data = dd.collect_closed_loop(
                        spec, opt["K"], opt["p"], opt["N"], opt["z"], N,
                        seed=seed, expl_scale=1.0,
                        state_obs_noise_std=1e-4,
                        input_obs_noise_std=1e-5,
                        dist_obs_noise_std=1e-5,
                        target_obs_noise_std=1e-4,
                        dist_probe_std=0.5 * spec["d_std"])
                    a = dd.identify_adversary_block(
                        data, use_iv=use_iv, n_lags=1)
                    means[N].append(dict(
                        eN=float(np.linalg.norm(a["N_hat"] - opt["N"]) /
                                 max(np.linalg.norm(opt["N"]), 1e-15)),
                        ez=float(np.linalg.norm(a["z_hat"] - opt["z"]))))
            eN_mean = [np.mean([x["eN"] for x in means[N]]) for N in Ns]
            ez_mean = [np.mean([x["ez"] for x in means[N]]) for N in Ns]
            out[name][method] = dict(
                Ns=list(Ns), eN_mean=eN_mean, ez_mean=ez_mean,
                slope_eN=dd.loglog_slope(Ns, eN_mean),
                slope_ez=dd.loglog_slope(Ns, ez_mean))
            rows.append([name, method,
                         f"{out[name][method]['slope_eN']:.3f}",
                         f"{out[name][method]['slope_ez']:.3f}",
                         ",".join(f"{x:.3e}" for x in eN_mean),
                         ",".join(f"{x:.3e}" for x in ez_mean)])
    write_table(
        "od_e5b_adversary_block_rates",
        ["Scenario", "Method", "slope eN", "slope ez", "eN means",
         "ez means"], rows,
        notes=("Adversary-block rate check with data generated under the fixed "
               "model-based DRC policy and N=4000,8000,16000,32000, five seeds. "
               "The direct N estimate uses d-q=Nx+z+xi; it is separate from the "
               "policy error after SDPI."))
    return out


def run_sdpi_diagnostics(seed=0, N=8000):
    rows = []
    details = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        opt = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                                spec["R"], spec["alpha"], spec["lam"],
                                spec["d_bar"])
        V = spec["d_bar"].reshape(1, -1)
        # Positive IV/identified-model run with one outer round.  This isolates
        # SDPI diagnostics from a later fresh-batch rejection.
        r = a4.run_data_driven_algorithm4(
            spec, N=N, seed=seed, rounds=1, use_iv=True,
            instrument_mode="probe", dist_probe_scale=0.5)
        h = r["history"][0]
        s = r["sdpi"]
        fs = s["final_schur"]
        fp = s["fixed_point"]
        mp = r["history"][0]["scaling"]
        rows.append([
            name, "identified IV, normal", s["stop_reason"],
            s["phase1_steps"], s["initialization"].get("beta_scale", 1.0),
            s["initialization"].get("source", "—"),
            f"{s['rho_G']:.6e}", f"{h['true_rho_G']:.6e}",
            f"{max(s['final_delta'].values()):.3e}",
            f"{fs.get('cond_S_u', np.nan):.3e}",
            f"{fs.get('cond_S_m', np.nan):.3e}",
            f"{fp['residual_P']:.3e}", f"{fp['residual_chi']:.3e}",
            f"{fp['residual_psi']:.3e}",
            f"{mp.get('mapping_identity_residual', 0.0):.3e}",
            "well posed" if not fs.get("boundary", True) else "boundary"])
        # Boundary stop test on the true model.  This is a deliberate
        # assumption-failure test, not a reported controller result.
        b = sdpi.solve_sdpi(
            spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
            spec["alpha"], 0.01, V, K0=opt["K"], p0=opt["p"],
            N0=opt["N"], z0=opt["z"], max_phase2=5, n_value_samples=10)
        bfs = b["final_schur"]
        rows.append([
            name, "deliberate Schur-boundary test", b["stop_reason"],
            b["phase1_steps"], b["initialization"].get("beta_scale", 1.0),
            b["initialization"].get("source", "—"),
            f"{b['rho_G']:.6e}", f"{b['rho_G']:.6e}",
            f"{max(b['final_delta'].values()):.3e}",
            f"{bfs.get('cond_S_u', np.nan):.3e}",
            f"{bfs.get('cond_S_m', np.nan):.3e}",
            f"{b['fixed_point']['residual_P']:.3e}",
            f"{b['fixed_point']['residual_chi']:.3e}",
            f"{b['fixed_point']['residual_psi']:.3e}",
            "—", bfs.get("reason", "—")])
        details[name] = dict(normal=r, boundary=b)
    write_table(
        "od_e9_sdpi_diagnostics",
        ["Scenario", "Case", "stop reason", "phase-I steps", "beta scale",
         "initialization", "rho identified", "rho true", "max delta",
         "cond S_u", "cond S_m", "residual P", "residual chi",
         "residual psi", "mapping residual", "boundary status"], rows,
        notes=("SDPI diagnostics. The scaling back-map is the identity in "
               "(K,p,N,z) coordinates followed by an explicit check on the "
               "original A,B,E; beta*G_orig is verified separately. The lambda="
               "0.01 rows deliberately exercise the singular/indefinite Schur "
               "boundary and must stop rather than use a pseudo-inverse."))
    return details


def run_identifiability_negative(N=8000, seeds=range(5)):
    rows = []
    details = {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        details[name] = []
        for seed in seeds:
            for method in ("OLS", "IV/2SLS"):
                pol = a4._zero_policy(spec["A"].shape[0], spec["B"].shape[1],
                                      spec["E"].shape[1])
                data = dd.collect_closed_loop(
                    spec, pol["K"], pol["p"], pol["N"], pol["z"], N,
                    seed=seed, expl_scale=0.25,
                    state_obs_noise_std=1e-4, input_obs_noise_std=1e-5,
                    dist_obs_noise_std=1e-5, target_obs_noise_std=1e-4,
                    dist_probe_std=0.0)
                if method == "OLS":
                    est = dd.identify_full_ols(data)
                    adv = dd.identify_adversary_block(data, use_iv=False)
                else:
                    est = dd.identify_full_iv(data, n_lags=1,
                                              instrument_mode="lagged")
                    adv = dd.identify_adversary_block(data, use_iv=True)
                est = a4._merge_estimates(est, adv)
                cert = dd.identifiability_certificate(data, est)
                details[name].append(dict(seed=seed, method=method,
                                          certificate=cert,
                                          rank=est["rank"],
                                          reg_dim=est["reg_dim"],
                                          e_f=est.get("E_first_stage_min_F"),
                                          n_f=est.get("N_first_stage_min_F")))
        ds = details[name]
        ols = [d for d in ds if d["method"] == "OLS"]
        iv = [d for d in ds if d["method"] == "IV/2SLS"]
        rows.append([name, "OLS", len(ols),
                     f"{sum(d['certificate']['E_status'] != 'identifiable' for d in ols)}/{len(ols)}",
                     f"{np.mean([d['rank'] == d['reg_dim'] for d in ols]):.2f}",
                     "—", "confounded"
                     if ols[0]["certificate"]["E_status"] != "identifiable" else "identified"])
        rows.append([name, "IV/2SLS, lagged only", len(iv),
                     f"{sum(d['certificate']['E_status'] != 'identifiable' for d in iv)}/{len(iv)}",
                     f"{np.mean([d['rank'] == d['reg_dim'] for d in iv]):.2f}",
                     "—" if not np.isfinite(np.mean([d['e_f'] for d in iv
                                                      if d['e_f'] is not None]))
                     else f"{np.mean([d['e_f'] for d in iv if d['e_f'] is not None]):.3f}",
                     "not identifiable without exogenous disturbance excitation"])
    write_table(
        "od_e10_identifiability_negative",
        ["Scenario", "Method", "Seeds", "E non-identifiable", "rank full",
         "E first-stage F", "Conclusion"], rows,
        notes=("Negative control with q=0. The matrix rank can be full while E "
               "is still not identified: d=Nx+z+xi is feedback-correlated with "
               "x, and xi enters the transition error. This result is retained "
               "to prevent a false claim that generic closed-loop data identify "
               "the pair (E,N). Exogenous disturbance-channel excitation (or "
               "prior knowledge) is required."))
    return details


def main():
    t0 = time.time()
    result = {}
    result["e4"] = run_end_to_end()
    result["e5"] = run_rates()
    result["e5b"] = run_adversary_rates()
    result["e9"] = run_sdpi_diagnostics()
    result["e10"] = run_identifiability_negative()
    result["wall_seconds"] = time.time() - t0
    with open(os.path.join(LOG, "final_alg4_experiments.json"), "w",
              encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2, default=float)
    print(json.dumps({"wall_seconds": result["wall_seconds"],
                      "tables": ["od_e4_algorithm4_end_to_end",
                                 "od_e5_algorithm4_rates",
                                 "od_e5_algorithm4_rates_points",
                                 "od_e5b_adversary_block_rates",
                                 "od_e9_sdpi_diagnostics",
                                 "od_e10_identifiability_negative"]}, indent=2))


if __name__ == "__main__":
    main()
