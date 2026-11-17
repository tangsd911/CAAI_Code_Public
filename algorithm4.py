#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identification-based Algorithm 4 with an explicit identifiability gate.

Pipeline implemented here
-------------------------
1. Independent pilot data estimate the disturbance support ``V``.
2. Each outer round collects a fresh closed-loop batch under the current
   ``(K,p,N,z)``.  The batch contains input exploration and, when requested,
   an independent disturbance-channel probe ``q``.
3. ``A,B,E,c`` are estimated by OLS or 2SLS.  The affine adversary block
   ``d-q=Nx+z+xi`` is estimated separately.  With the probe, the IV estimate
   is identifiable; without it, the code returns a negative identifiability
   result instead of fitting a confounded model and pretending otherwise.
4. Model-based SDPI evaluates/improves the policy on the identified model.
   The scaling constructor records its back-mapping to the original matrices.
5. The updated policy is fed back without consulting the true model.  The true
   spectral radius and value function are computed after each update only as
   post-hoc diagnostics for the simulation study.
6. The loop stops on policy stationarity or the configured round limit.

The true matrices are used only for data generation and post-hoc scoring; they
do not gate, reject, or modify an Algorithm 4 update.
"""

import numpy as np

import data_driven as dd
import drc_core as core
import sdpi


def _zero_policy(nx, nu, nd):
    return dict(K=np.zeros((nu, nx)), p=np.zeros(nu),
                N=np.zeros((nd, nx)), z=np.zeros(nd))


def _policy_diff(a, b):
    return dict(K=float(np.linalg.norm(a["K"] - b["K"], "fro")),
                p=float(np.linalg.norm(a["p"] - b["p"])),
                N=float(np.linalg.norm(a["N"] - b["N"], "fro")),
                z=float(np.linalg.norm(a["z"] - b["z"])))


def _merge_estimates(est, adv):
    out = dict(est)
    e_f = est.get("first_stage_min_F")
    out["E_first_stage_min_F"] = e_f
    out["N_first_stage_min_F"] = adv.get("first_stage_min_F")
    out.update(adv)
    # Keep both diagnostics: the two structural equations use different
    # excluded instruments.  A single overwritten F-statistic would be
    # misleading in the end-to-end table.
    out["E_first_stage_min_F"] = e_f
    out["N_first_stage_min_F"] = adv.get("first_stage_min_F")
    return out


def solve_identified_policy(A, B, E, Q, R, alpha, lam, V, init,
                            max_phase2=150, eps=1e-8):
    """Run SDPI on a fully identified model from the current policy."""
    return sdpi.solve_sdpi(
        A, B, E, Q, R, alpha, lam, V,
        K0=init["K"], p0=init["p"], N0=init["N"], z0=init["z"],
        max_phase2=max_phase2, eps=eps, n_value_samples=100)


def identifiability_negative_result(spec, N=4000, seed=0, expl_scale=0.25,
                                    n_lags=1):
    """Run the feedback-endogenous design with ``q=0`` and report the limit.

    This is a genuine negative result: the full regressor matrix may be full
    rank, but ``E`` is not identified because the only current variation in
    ``d`` is the innovation ``xi`` that also drives the state.  No SDPI step is
    attempted on the confounded estimate.
    """
    pol = _zero_policy(spec["A"].shape[0], spec["B"].shape[1],
                       spec["E"].shape[1])
    opt = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                            spec["R"], spec["alpha"], spec["lam"],
                            spec["d_bar"])
    data = dd.collect_closed_loop(
        spec, pol["K"], pol["p"], pol["N"], pol["z"], N, seed=seed,
        expl_scale=expl_scale, state_obs_noise_std=1e-4,
        input_obs_noise_std=1e-5, dist_obs_noise_std=1e-5,
        target_obs_noise_std=1e-4, dist_probe_std=0.0)
    est = dd.identify_full_ols(data)
    adv = dd.identify_adversary_block(data, use_iv=False)
    est = _merge_estimates(est, adv)
    cert = dd.identifiability_certificate(data, est)
    return dict(status="negative_unidentifiable", N=int(N), seed=int(seed),
                certificate=cert,
                rank=est["rank"], reg_dim=est["reg_dim"],
                eN=float(np.linalg.norm(est["N_hat"] - opt["N"]) /
                         max(np.linalg.norm(opt["N"]), 1e-15)))


def run_data_driven_algorithm4(spec, N=8000, seed=0, rounds=2,
                               expl_scale=0.25, use_iv=True, n_lags=1,
                               instrument_mode="probe",
                               dist_probe_scale=0.5,
                               support_n=2000, support_g=1,
                               state_obs_noise_std=1e-4,
                               input_obs_noise_std=1e-5,
                               dist_obs_noise_std=1e-5,
                               target_obs_noise_std=1e-4,
                               max_phase2=150,
                               require_identifiable=True,
                               stop_tol=1e-7):
    """Run one fresh-data identification/SDPI trajectory for one seed.

    The ``X`` returned by ``collect_closed_loop`` is treated as the exact
    reconstructed equivalent-state sample ``x^T=x``.  This full-state
    specialization isolates the identification theorem from reconstruction
    error; the separate ``data_driven_reconstruction.py`` experiment verifies
    the reconstruction error itself.  No true matrix is used in synthesis.
    """
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    nx, nu, nd = A.shape[0], B.shape[1], E.shape[1]

    # Support data are independent of the closed-loop policy batches.  This is
    # an offline data set, not access to the true support in the controller.
    pilot = dd.collect_support_points(spec, support_n, seed=seed + 900_001)
    V_hat = dd.estimate_support(pilot.T, g=support_g)
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])

    pol = _zero_policy(nx, nu, nd)
    history = []
    est = None
    certificate = None
    last_sdpi = None
    status = "not_started"

    probe_std = (np.asarray(spec["d_std"], dtype=float) * dist_probe_scale
                 if dist_probe_scale > 0 else 0.0)
    if instrument_mode == "probe" and not np.any(probe_std > 0):
        raise ValueError("probe instrument mode requires dist_probe_scale > 0")

    for r in range(rounds):
        data = dd.collect_closed_loop(
            spec, pol["K"], pol["p"], pol["N"], pol["z"], N,
            seed=seed + 1000 * r, expl_scale=expl_scale,
            state_obs_noise_std=state_obs_noise_std,
            input_obs_noise_std=input_obs_noise_std,
            dist_obs_noise_std=dist_obs_noise_std,
            target_obs_noise_std=target_obs_noise_std,
            dist_probe_std=probe_std)
        if use_iv:
            est = dd.identify_full_iv(data, n_lags=n_lags,
                                      instrument_mode=instrument_mode)
        else:
            est = dd.identify_full_ols(data)
        adv = dd.identify_adversary_block(
            data, use_iv=use_iv, n_lags=n_lags)
        est = _merge_estimates(est, adv)
        certificate = dd.identifiability_certificate(data, est)

        if use_iv and require_identifiable and \
                certificate["E_status"] != "identifiable":
            history.append(dict(
                round=r, status="identification_gate_failed",
                certificate=certificate, rank=est["rank"],
                rank_instruments=est.get("rank_instruments"),
                reg_dim=est["reg_dim"],
                instrument_dim=est.get("instrument_dim"),
                E_first_stage_min_F=est.get("E_first_stage_min_F"),
                N_first_stage_min_F=est.get("N_first_stage_min_F"),
                accepted=False, true_rho_G=float("nan"),
                sdpi_stop="not_run"))
            status = "identification_gate_failed"
            break

        init = dict(K=pol["K"], p=pol["p"],
                    N=adv["N_hat"], z=adv["z_hat"])
        try:
            res = solve_identified_policy(
                est["A_hat"], est["B_hat"], est["E_hat"], Q, R, alpha, lam,
                V_hat, init, max_phase2=max_phase2)
        except Exception as exc:
            history.append(dict(
                round=r, status="identified_sdpi_failed",
                certificate=certificate, rank=est["rank"],
                rank_instruments=est.get("rank_instruments"),
                reg_dim=est["reg_dim"],
                instrument_dim=est.get("instrument_dim"),
                E_first_stage_min_F=est.get("E_first_stage_min_F"),
                N_first_stage_min_F=est.get("N_first_stage_min_F"),
                accepted=False, true_rho_G=float("nan"),
                sdpi_stop="identified_sdpi_failed", error=str(exc)))
            last_sdpi = dict(stop_reason="identified_sdpi_failed", history=[])
            status = "identified_sdpi_failed"
            break

        new_pol = {k: np.array(res[k], dtype=float, copy=True)
                   for k in ("K", "p", "N", "z")}
        delta = _policy_diff(new_pol, pol)
        rho_true = float(sdpi.rho(A + B @ new_pol["K"] + E @ new_pol["N"]))
        accepted = bool(np.isfinite(rho_true) and rho_true < 1.0)

        true_eval = None
        if accepted:
            ev = core.policy_evaluate(A, B, E, Q, R, alpha, lam,
                                      spec["d_bar"].reshape(1, -1),
                                      new_pol["K"], new_pol["p"],
                                      new_pol["N"], new_pol["z"])
            true_eval = sdpi._fixed_point_diagnostics(
                A, B, E, Q, R, alpha, lam, spec["d_bar"].reshape(1, -1),
                new_pol, ev["P"], ev["chi"], ev["psi"])
            true_eval["P_error_vs_reference"] = float(
                np.linalg.norm(ev["P"] - opt["P"]) / np.linalg.norm(opt["P"]))
        # ``accepted`` is a post-hoc diagnostic only.  It is not an update gate.

        ident_errors = dd.full_identification_errors(spec, est, opt)
        history.append(dict(
            round=r, status="updated",
            certificate=certificate, rank=est["rank"],
            rank_instruments=est.get("rank_instruments"),
            reg_dim=est["reg_dim"], instrument_dim=est.get("instrument_dim"),
            E_first_stage_min_F=est.get("E_first_stage_min_F"),
            N_first_stage_min_F=est.get("N_first_stage_min_F"),
            cond_Phi=est.get("cond_Phi"),
            eA=ident_errors["eA"], eB=ident_errors["eB"],
            eE=ident_errors["eE"], eN=ident_errors.get("eN"),
            ez=ident_errors.get("ez"),
            V_error=float(np.linalg.norm(V_hat.mean(axis=0) - spec["d_bar"]) /
                          max(np.linalg.norm(spec["d_bar"]), 1e-15)),
            identified_rho_G=float(res["rho_G"]),
            true_rho_G=rho_true, accepted=accepted,
            sdpi_stop=res["stop_reason"], delta=delta,
            scaling=res["initialization"],
            final_schur=res["final_schur"],
            fixed_point=res["fixed_point"], true_eval=true_eval,
            policy_error={k: float(np.linalg.norm(new_pol[k] - opt[k]))
                          for k in ("K", "p", "N", "z")}))

        pol = new_pol
        last_sdpi = res
        if max(delta.values()) < stop_tol:
            status = "converged"
            break
        status = "round_limit"
    else:
        status = "round_limit"

    if est is None or last_sdpi is None:
        errors = dict(eA=float("nan"), eB=float("nan"), eE=float("nan"),
                      eN=float("nan"), ez=float("nan"),
                      eK=float("nan"), ep=float("nan"),
                      eN_policy=float("nan"), ez_policy=float("nan"),
                      eP=float("nan"))
        true_rho = float("nan")
        accepted = False
    else:
        errors = dict(
            eA=history[-1].get("eA", float("nan")),
            eB=history[-1].get("eB", float("nan")),
            eE=history[-1].get("eE", float("nan")),
            eN=history[-1].get("eN", float("nan")),
            ez=history[-1].get("ez", float("nan")),
            eK=float(np.linalg.norm(pol["K"] - opt["K"], "fro")),
            ep=float(np.linalg.norm(pol["p"] - opt["p"])),
            eN_policy=float(np.linalg.norm(pol["N"] - opt["N"], "fro")),
            ez_policy=float(np.linalg.norm(pol["z"] - opt["z"])),
            eP=float(np.linalg.norm(last_sdpi.get("P", opt["P"]) - opt["P"]) /
                     np.linalg.norm(opt["P"])))
        true_rho = float(sdpi.rho(A + B @ pol["K"] + E @ pol["N"]))
        accepted = bool(true_rho < 1.0)

    return dict(
        policy=pol, estimate=est, opt=opt, V_hat=V_hat, history=history,
        sdpi=last_sdpi, certificate=certificate, status=status,
        errors=errors, true_rho_G=true_rho,
        model_based_rho_G=float(opt["rho_G"]),
        use_iv=bool(use_iv), N=int(N), seed=int(seed),
        rounds=int(rounds), accepted=accepted,
        identification=dict(
            rank=None if est is None else est.get("rank"),
            rank_instruments=None if est is None else est.get("rank_instruments"),
            reg_dim=None if est is None else est.get("reg_dim"),
            instrument_dim=None if est is None else est.get("instrument_dim"),
            E_first_stage_min_F=None if est is None else est.get("E_first_stage_min_F"),
            N_first_stage_min_F=None if est is None else est.get("N_first_stage_min_F"),
            overid_p_min=None if est is None else est.get("overid_p_min")),
        data_design=dict(expl_scale=float(expl_scale),
                         dist_probe_scale=float(dist_probe_scale),
                         instrument_mode=instrument_mode,
                         support_g=int(support_g)))


def alg4_slope_scan(spec, Ns=(1000, 2000, 4000, 8000, 16000),
                    seeds=range(5), use_iv=True, n_lags=1,
                    expl_scale=0.25, dist_probe_scale=0.5,
                    state_obs_noise_std=1e-4,
                    input_obs_noise_std=1e-5,
                    dist_obs_noise_std=1e-5,
                    target_obs_noise_std=1e-4,
                    rounds=2):
    """Scan identification and policy errors over data length.

    The slope is a finite-sample log-log regression.  No slope is forced: a
    weak-instrument outcome remains visible as a slope far from the expected
    ``-1/2`` or as an identification-gate failure.
    """
    per_n = {}
    for N in Ns:
        rows = []
        for seed in seeds:
            r = run_data_driven_algorithm4(
                spec, N=N, seed=seed, rounds=rounds, expl_scale=expl_scale,
                use_iv=use_iv, n_lags=n_lags, instrument_mode="probe",
                dist_probe_scale=dist_probe_scale,
                state_obs_noise_std=state_obs_noise_std,
                input_obs_noise_std=input_obs_noise_std,
                dist_obs_noise_std=dist_obs_noise_std,
                target_obs_noise_std=target_obs_noise_std)
            rows.append(r["errors"])
        keys = sorted(rows[0])
        per_n[int(N)] = dict(
            mean={k: float(np.nanmean([row[k] for row in rows])) for k in keys},
            std={k: float(np.nanstd([row[k] for row in rows])) for k in keys},
            n_failed=int(sum(not np.isfinite(row.get("eA", np.nan))
                             for row in rows)))
    slopes = {}
    for key in per_n[list(per_n)[0]]["mean"]:
        slopes[key] = dd.loglog_slope(
            list(per_n.keys()), [per_n[N]["mean"][key] for N in per_n])
    return dict(per_n=per_n, slopes=slopes, use_iv=bool(use_iv),
                noise=dict(state=state_obs_noise_std,
                           input=input_obs_noise_std,
                           disturbance=dist_obs_noise_std,
                           target=target_obs_noise_std),
                data_design=dict(expl_scale=expl_scale,
                                 dist_probe_scale=dist_probe_scale))
