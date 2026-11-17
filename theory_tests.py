#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression / theory-consistency tests for the verified DRC package.

Checks:
  T1  Fixed-point residuals: P Lyapunov equation, chi linear equation, psi
      constant term, for both scenarios.
  T2  General-g equivalence:
        (a) g=1 solver reproduces the established baseline scalars;
        (b) g=5 support points whose mean equals d_bar give K, p, N, z_bar
            within 1e-8 of the g=1 solution (the mean-reduction identity in
            drc_core.py);
        (c) psi is invariant when the support has the same mean; the atomwise
            transport displacement is the same for every support point.
  T3  Noiseless output reconstruction and forward propagation with the full
      measured disturbance history; a negative check confirms that omitting
      the disturbance-propagation term is not an identity.
  T4  Hessian signs at nominal lambda: H_uu ≻ 0, H_ww ≺ 0.

Run:  python3 theory_tests.py
Exit code 0 only if every assertion holds.
"""

import numpy as np

import drc_core as core
import models
import sdpi
import data_driven as dd
import thm3
import wasserstein_experiments as wx
import new_experiments as nex

TOL = 1e-8


def check_close(name, a, b, tol=TOL):
    err = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    status = "OK" if err <= tol else "FAIL"
    print(f"  [{status}] {name}: error={err:.3e}")
    assert err <= tol, (name, err)


def test_fixed_point(name, spec):
    print(f"T1 fixed-point residuals: {name}")
    d = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                          spec["R"], spec["alpha"], spec["lam"], spec["d_bar"])
    K, p, N, z, P, chi, G, F = (d["K"], d["p"], d["N"], d["z"], d["P"],
                                d["chi"], d["G"], d["F"])
    PhiP = spec["Q"] + K.T @ spec["R"] @ K - spec["lam"] * (N.T @ N)
    P_res = P - spec["alpha"] * (G.T @ P @ G) - PhiP
    check_close("P Lyapunov residual", P_res, np.zeros_like(P), 2e-7)

    PhiChi = p @ spec["R"] @ K - spec["lam"] * (z @ N) \
        + spec["lam"] * (spec["d_bar"] @ N)
    chi_rhs = PhiChi.T + spec["alpha"] * (G.T @ P @ F)
    chi_res = (np.eye(spec["A"].shape[0]) - spec["alpha"] * G.T) @ chi \
        - chi_rhs
    check_close("chi linear residual", chi_res, np.zeros(spec["A"].shape[0]),
                2e-7)

    psi_ref = (p @ spec["R"] @ p
               - spec["lam"] * ((z - spec["d_bar"]) @ (z - spec["d_bar"]))
               + spec["alpha"] * (F @ P @ F) + 2 * spec["alpha"] * (F @ chi)) \
        / (1 - spec["alpha"])
    check_close("psi formula", d["psi"], psi_ref, 1e-10)


def test_g_equivalence(name, spec):
    print(f"T2 general-g equivalence: {name}")
    g1 = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                            spec["R"], spec["alpha"], spec["lam"],
                            spec["d_bar"])
    # g=1 general entry (V as (1, nd)) must be bit-identical to solve_drc_pi
    V1 = spec["d_bar"].reshape(1, -1)
    g1b = core.solve_drc_pi_g(spec["A"], spec["B"], spec["E"], spec["Q"],
                              spec["R"], spec["alpha"], spec["lam"], V1)
    for key in ("K", "p", "N", "z"):
        check_close(f"g=1 bit-identical {key}", g1[key], g1b[key], 1e-12)

    # g=5 split: support points v_j = d_bar + delta_j, mean delta = 0
    rng = np.random.default_rng(123)
    delta = rng.standard_normal((5, len(spec["d_bar"])))
    delta -= delta.mean(axis=0)
    V5 = spec["d_bar"][None, :] + delta
    assert np.linalg.norm(V5.mean(axis=0) - spec["d_bar"]) < 1e-12
    g5 = core.solve_drc_pi_g(spec["A"], spec["B"], spec["E"], spec["Q"],
                             spec["R"], spec["alpha"], spec["lam"], V5)
    for key in ("K", "p", "N", "z"):
        check_close(f"g=5 vs g=1 {key}", g5[key], g1[key], TOL)

    # For a fixed support mean, atomwise transport is m-v_bar for every atom;
    # hence psi carries no support-spread constant.
    check_close("psi mean-reduction identity", g5["psi"], g1["psi"], 1e-8)


def test_reconstruction(name, spec):
    print(f"T3 output reconstruction and forward propagation: {name}")
    A, B, C, E = spec["A"], spec["B"], spec["C"], spec["E"]
    ell = 3
    rng = np.random.default_rng(3)
    x_delayed = rng.standard_normal(A.shape[0])
    us = rng.standard_normal((ell, B.shape[1]))
    ds = rng.standard_normal((ell, E.shape[1]))
    ys = []
    xcur = x_delayed.copy()
    for i in range(ell):
        ys.append(C @ xcur)
        xcur = A @ xcur + B @ us[i] + E @ ds[i]
    x_true = xcur
    ywin = np.concatenate(ys)

    z = core.reconstruct_delayed_state(A, B, C, E, ywin, us.reshape(-1),
                                       ds.reshape(-1), ell)
    check_close("noiseless delayed-state reconstruction", z, x_delayed, 1e-9)

    xhat = core.propagate_reconstructed_state(A, B, E, z, us, ds)
    check_close("forward propagation with disturbance history", xhat, x_true,
                1e-9)

    xhat_no_dist = core.propagate_reconstructed_state(
        A, B, E, z, us, np.zeros_like(ds))
    omitted_error = float(np.linalg.norm(xhat_no_dist - x_true))
    print(f"  [NEG] omitted disturbance-propagation error={omitted_error:.3e}")
    assert omitted_error > 1e-3, omitted_error


def test_cost_indexing():
    """The stage cost must use (x_t,u_t), not (x_{t+1},u_t)."""
    print("T3 stage-cost time indexing")
    A = np.array([[2.0]])
    B = np.array([[1.0]])
    E = np.array([[1.0]])
    K = np.array([[0.0]])
    p = np.array([0.3])
    Q = np.array([[1.0]])
    R = np.array([[2.0]])
    x_ref = np.array([0.0])
    d_mean = np.array([0.0])
    d_std = np.array([0.0])
    _, stage_cost = core.rollout(
        A, B, E, K, p, x_ref, d_mean, d_std, T=1, ss=0, seed=0,
        Q=Q, R=R, mae_mode="full")
    expected = float(p @ R @ p)  # x_0=0, so only u_0'Ru_0 contributes
    check_close("one-step stage cost uses current state", stage_cost, expected,
                1e-12)


def test_output_feedback_closed_loop(name, spec):
    """Exercise the full output-window reconstructor inside the closed loop."""
    print(f"T4 output-feedback closed loop: {name}")
    T, ss = 600, 300
    seeds = range(1000, 1002)
    A, B, C, E = spec["A"], spec["B"], spec["C"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    full = core.mc_batch(
        A, B, E, opt["K"], opt["p"], spec["x_ref"], spec["d_bar"],
        spec["d_std"], T, ss, alpha, spec["mae_mode"], Q, R, seeds=seeds)
    known = core.mc_batch_output_feedback(
        A, B, C, E, opt["K"], opt["p"], spec["x_ref"], spec["d_bar"],
        spec["d_std"], T, ss, alpha, spec["mae_mode"], Q, R,
        seeds=seeds, ell=3, d_history="known")
    mean = core.mc_batch_output_feedback(
        A, B, C, E, opt["K"], opt["p"], spec["x_ref"], spec["d_bar"],
        spec["d_std"], T, ss, alpha, spec["mae_mode"], Q, R,
        seeds=seeds, ell=3, d_history="mean")

    check_close("known-history OFB MAE vs full state", known["mae_mean"],
                full[0], 2e-10)
    check_close("known-history OFB cost vs full state", known["cost_mean"],
                full[2], 2e-8)
    assert known["rank_O"] == A.shape[0]
    assert known["warmup_steps"] == 3
    assert known["reconstructed_steps"] == T - ss
    assert mean["recon_rmse_mean"] > 0.0
    print(f"  rank(O_3)={known['rank_O']}/{A.shape[0]}, "
          f"known-history recon RMSE={known['recon_rmse_mean']:.2e}, "
          f"mean-history recon RMSE={mean['recon_rmse_mean']:.2e}")


def test_hessian_signs(name, spec):
    print(f"T4 Hessian signs: {name}")
    d = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                          spec["R"], spec["alpha"], spec["lam"], spec["d_bar"])
    print(f"  H_uu min eig = {d['H_uu_min_eig']:.4f} (>0)")
    print(f"  H_ww min eig = {d['H_ww_min_eig']:.4f} (<0)")
    assert d["H_uu_min_eig"] > 0
    assert d["H_ww_min_eig"] < 0


# --------------------------------------------------------------------------- #
#  T1 — general-g per-support dual expansion and g scan
# --------------------------------------------------------------------------- #
def test_per_support_identity(name, spec):
    print(f"T1 per-support dual expansion: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    g1 = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    rng = np.random.default_rng(7)
    for g in (1, 2, 5, 10):
        delta = rng.standard_normal((g, len(spec["d_bar"])))
        delta -= delta.mean(axis=0)
        V = spec["d_bar"][None, :] + delta
        imp = core.per_support_improvement(A, B, E, Q, R, alpha, lam, V,
                                           g1["P"], g1["chi"])
        # N identical across atoms
        for j, Nj in enumerate(imp["N_list"]):
            check_close(f"g={g} N_{j} shared", Nj, imp["N"], 1e-12)
        # z^j = v_j - v_bar + z_bar
        zj_ref = V - imp["v_bar"][None, :] + imp["z"][None, :]
        check_close(f"g={g} per-atom z^j identity", imp["zj"], zj_ref, 1e-12)
        # g scan: K, p, N, z_bar coincide with g=1 to 1e-8
        gx = core.solve_drc_pi_g(A, B, E, Q, R, alpha, lam, V)
        for key in ("K", "p", "N", "z"):
            check_close(f"g={g} scan vs g=1 {key}", gx[key], g1[key], TOL)


# --------------------------------------------------------------------------- #
#  T2 — SDPI convergence, baseline agreement, diagnostics
# --------------------------------------------------------------------------- #
def test_sdpi_baseline(name, spec):
    print(f"T2 SDPI vs naive fixed point: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    V = spec["d_bar"].reshape(1, -1)
    base = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    res = sdpi.solve_sdpi(A, B, E, Q, R, alpha, lam, V)
    print(f"  Phase I rounds={res['phase1_steps']}, switch step="
          f"{res['switched_at']}, Phase II steps={res['phase2_steps']}, "
          f"rho(G)={res['rho_G']:.5f}, value violations="
          f"{res['monotonic_violations']}")
    check_close("SDPI K vs baseline", res["K"], base["K"], 1e-8)
    check_close("SDPI p vs baseline", res["p"], base["p"], 1e-8)
    check_close("SDPI N vs baseline", res["N"], base["N"], 1e-8)
    check_close("SDPI z vs baseline", res["z"], base["z"], 1e-8)
    check_close("SDPI P vs baseline", res["P"], base["P"], 1e-7)
    print(f"  initialization={res['initialization']['source']}, "
          f"stop={res['stop_reason']}, final deltas="
          f"{res['final_delta']}")
    assert res["stop_reason"] in {"stationarity", "interpolation_step",
                                  "max_phase2"}
    assert set(res["final_delta"]) == {"K", "p", "N", "z"}
    if res["stop_reason"] == "stationarity":
        assert max(res["final_delta"].values()) < 1e-8


def test_sdpi_degenerate(name, spec):
    print(f"T2 SDPI degenerate at baseline policy: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    V = spec["d_bar"].reshape(1, -1)
    base = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    res = sdpi.solve_sdpi(A, B, E, Q, R, alpha, lam, V,
                          K0=base["K"], p0=base["p"], N0=base["N"],
                          z0=base["z"])
    check_close("degenerate K", res["K"], base["K"], 1e-8)
    check_close("degenerate p", res["p"], base["p"], 1e-8)
    check_close("degenerate P", res["P"], base["P"], 1e-7)


def test_sdpi_arbitrary_initial(name, spec):
    """A deliberately bad initial gain must be repaired by verified init + PI."""
    print(f"T2 SDPI arbitrary initial policy: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    V = spec["d_bar"].reshape(1, -1)
    K0 = 10.0 * np.ones((B.shape[1], A.shape[0]))
    res = sdpi.solve_sdpi(A, B, E, spec["Q"], spec["R"], spec["alpha"],
                          spec["lam"], V, K0=K0, max_phase1=2,
                          max_phase2=100)
    assert res["rho_G"] < 1.0
    assert res["initialization"]["source"] in {
        "scaled_candidate_verified_on_original", "discounted_lqr_fallback"}
    assert res["stop_reason"] == "stationarity"
    assert max(res["final_delta"].values()) < 1e-8
    print(f"  initial rho={res['initialization']['initial_rho']:.4f}, "
          f"source={res['initialization']['source']}, "
          f"final rho={res['rho_G']:.5f}")


# --------------------------------------------------------------------------- #
#  T3 — identification light test: full rank, no truth in regressors
# --------------------------------------------------------------------------- #
def test_identification_light(name, spec):
    print(f"T3 identification rank/PE: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    data = dd.collect_closed_loop(spec, opt["K"], opt["p"], opt["N"], opt["z"],
                                   2000, seed=0, expl_scale=0.1)
    cl = dd.identify_closed_loop(data)
    ol = dd.identify_open_loop(data)
    assert cl["rank"] == cl["reg_dim"], (name, cl["rank"], cl["reg_dim"])
    assert ol["rank"] == ol["reg_dim"]
    assert cl["lam_min_Phi"] > 0
    print(f"  rank [X;E_e]={cl['rank']}/{cl['reg_dim']}, "
          f"lambda_min Phi={cl['lam_min_Phi']:.2e}")


# --------------------------------------------------------------------------- #
#  T4 — Thm.3 X criterion
# --------------------------------------------------------------------------- #
def test_thm3(name, spec):
    print(f"T4 Thm.3 X criterion: {name}")
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    PhiP = Q + opt["K"].T @ R @ opt["K"] - lam * (opt["N"].T @ opt["N"])
    rep = thm3.check_criterion(opt["G"], opt["P"], PhiP, alpha, name)
    assert rep["rho_lt1"]
    assert rep["construction"]["feasible"]
    assert rep["sdp"]["feasible"]
    assert rep["lyapunov_sdp"]["feasible"]
    G_bad, _ = thm3.destabilize(opt["G"], 1.02)
    rep_bad = thm3.check_criterion(G_bad, opt["P"], PhiP, alpha, name + "-bad")
    assert rep_bad["necessary_infeasible"]
    assert rep_bad["lyapunov_sdp"]["feasible"] is False
    print(f"  stable rho={rep['rho_G']:.4f}: construction+SDP feasible; "
          f"destabilized rho={rep_bad['rho_G']:.4f}: Lyapunov SDP "
          f"{rep_bad['lyapunov_sdp']['status']}")


# --------------------------------------------------------------------------- #
#  T5 — model-free truth-usage audit
# --------------------------------------------------------------------------- #
def test_truth_audit():
    print("T5 truth-usage audit")
    audit = nex.audit_truth_usage()
    assert audit["compliant"], audit["unexpected"]
    print(f"  truth A/B/E referenced only in: "
          f"{list(audit['truth_reference_locations'])}")


# --------------------------------------------------------------------------- #
#  T6 - atomwise W2 identity and controller-specific adversaries
# --------------------------------------------------------------------------- #
def test_wasserstein_adversaries(name, spec):
    print(f"T6 atomwise W2 and separate adversaries: {name}")
    nd = len(spec["d_bar"])
    rng = np.random.default_rng(20261112)
    V = spec["d_bar"][None, :] + spec["d_std"][None, :] * rng.standard_normal(
        (5, nd))
    drc = wx._policy_for_controller(spec, "drc", 10.0, V)
    lqr = wx._policy_for_controller(spec, "lqr", 10.0, V)
    for label, pol in (("drc", drc), ("lqr", lqr)):
        rad = wx.atomwise_w2_diagnostics(spec, pol, V=V, T=20)
        assert rad["atom_dispersion"] < 1e-10, (label, rad["atom_dispersion"])
        assert rad["mean_identity_error"] < 1e-10, (
            label, rad["mean_identity_error"])
        xs, _, _, _ = wx._rollout_policy(spec, pol, T=20)
        direct = np.sqrt(np.mean(np.sum(
            (xs @ pol["N"].T + pol["z"][None, :] - V.mean(axis=0)[None, :]) ** 2,
            axis=1)))
        check_close(f"{label} atomwise W2 formula", rad["w2_rms"], direct,
                    1e-10)
    # LQR must satisfy its own first-order adversary equation.
    x = rng.standard_normal(lqr["N"].shape[1])
    d = lqr["N"] @ x + lqr["z"]
    Ey = lqr["E"].T @ lqr["P"]
    Acl = lqr["Acl"]
    residual = ((10.0 * np.eye(nd) - spec["alpha"] * Ey @ lqr["E"]) @ d
                - 10.0 * V.mean(axis=0)
                - spec["alpha"] * Ey @ Acl
                @ (x - spec["x_ref"]))
    check_close("LQR own-adversary FOC", np.linalg.norm(residual), 0.0, 1e-9)
    assert not np.allclose(drc["N"], lqr["N"])
    print("  atom identities exact; LQR FOC residual="
          f"{np.linalg.norm(residual):.3e}; DRC and LQR adversaries distinct")


if __name__ == "__main__":
    hum = models.humanoid_spec()
    mic = models.microgrid_spec()
    test_fixed_point("humanoid", hum)
    test_fixed_point("microgrid", mic)
    test_g_equivalence("humanoid", hum)
    test_g_equivalence("microgrid", mic)
    test_reconstruction("humanoid", hum)
    test_reconstruction("microgrid", mic)
    test_cost_indexing()
    test_output_feedback_closed_loop("humanoid", hum)
    test_output_feedback_closed_loop("microgrid", mic)
    test_hessian_signs("humanoid", hum)
    test_hessian_signs("microgrid", mic)
    test_per_support_identity("humanoid", hum)
    test_per_support_identity("microgrid", mic)
    test_sdpi_baseline("humanoid", hum)
    test_sdpi_baseline("microgrid", mic)
    test_sdpi_degenerate("humanoid", hum)
    test_sdpi_degenerate("microgrid", mic)
    test_sdpi_arbitrary_initial("humanoid", hum)
    test_sdpi_arbitrary_initial("microgrid", mic)
    test_identification_light("humanoid", hum)
    test_identification_light("microgrid", mic)
    test_thm3("humanoid", hum)
    test_thm3("microgrid", mic)
    test_wasserstein_adversaries("humanoid", hum)
    test_wasserstein_adversaries("microgrid", mic)
    test_truth_audit()
    print("\nALL TESTS PASSED")
