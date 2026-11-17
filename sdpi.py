#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stabilizing discounted policy iteration (SDPI) for affine DRC.

Implementation scope
--------------------
The released code uses the scaling construction only as a *candidate
initializer*.  It does not assume that copying gains from a scaled problem is a
valid optimality-preserving map back to the original problem: after the
scaled candidate is formed, stability is checked directly on the original
matrix ``G=A+BK+EN``.  If the candidate is not stabilizing, the routine falls
back to the discounted LQR gain, which is also checked directly on the original
system.  Once a verified stabilizing policy exists, Phase II is ordinary
policy iteration on the original (unscaled) system with the conservative
stability interpolation.  This is deliberately narrower than an unproved
claim that scaling itself certifies original-system convergence.

Diagnostics
-----------
The interpolation is constrained by stability only.  The code records the
Bellman residual and pointwise value changes, but does not claim per-step cost
monotonicity or global convergence.  The stopping test uses all four policy
blocks ``(K,p,N,z)``; stopping on ``K,p`` alone is not accepted.
"""

import numpy as np

import drc_core as core


# --------------------------------------------------------------------------- #
#  Spectral radius
# --------------------------------------------------------------------------- #
def rho(G):
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def _interpolate(pol_cur, pol_dag, beta):
    """Blend policy: beta*dag + (1-beta)*current, on K,p,N,z_bar."""
    return dict(
        K=beta * pol_dag["K"] + (1 - beta) * pol_cur["K"],
        p=beta * pol_dag["p"] + (1 - beta) * pol_cur["p"],
        N=beta * pol_dag["N"] + (1 - beta) * pol_cur["N"],
        z=beta * pol_dag["z"] + (1 - beta) * pol_cur["z"],
    )


def _closed_loop(A, B, E, pol):
    return A + B @ pol["K"] + E @ pol["N"]


def scaled_policy_to_original(pol_scaled, beta):
    """Map a policy solved on ``(beta A,beta B,beta E,beta V)`` back.

    The scaling is applied only to the next-state transition, while the input
    and disturbance coordinates remain unchanged.  Hence the gain variables
    keep their coordinates::

        K_orig = K_scaled,  N_orig = N_scaled,
        p_orig = p_scaled,  z_orig = z_scaled.

    The returned dictionary also contains the explicit verification identity
    ``beta*G_orig = beta*(A+B K_orig+E N_orig)``.  The identity is checked by
    the caller on the original matrices; a copied candidate is never accepted
    merely because it is stable for the scaled model.
    """
    out = {k: np.array(pol_scaled[k], dtype=float, copy=True)
           for k in ("K", "p", "N", "z")}
    out["beta"] = float(beta)
    return out


def _schur_diagnostics(A, B, E, Q, R, alpha, lam, P,
                       eig_tol=1e-10, cond_max=1e12):
    """Condition numbers of the two saddle-point Schur complements.

    They are diagnostics only: singularity means the unconstrained improvement
    is not defined and the SDPI step is stopped rather than regularized
    silently.
    """
    H_uu = R + alpha * B.T @ P @ B
    H_um = alpha * B.T @ P @ E
    H_mm = -lam * np.eye(E.shape[1]) + alpha * E.T @ P @ E
    eig_uu = float(np.min(np.linalg.eigvalsh(H_uu)))
    eig_mm = float(np.max(np.linalg.eigvalsh(H_mm)))
    base = dict(H_uu_min=eig_uu, H_mm_max=eig_mm,
                H_uu_positive=bool(eig_uu > eig_tol),
                H_mm_negative=bool(eig_mm < -eig_tol))
    if not (base["H_uu_positive"] and base["H_mm_negative"]):
        base.update(cond_S_u=float("inf"), cond_S_m=float("inf"),
                    S_u_well_posed=False, S_m_well_posed=False,
                    boundary=True,
                    reason="H_uu or H_mm has the wrong definiteness")
        return base
    try:
        Su = H_uu - H_um @ np.linalg.solve(H_mm, H_um.T)
        Sm = H_mm - H_um.T @ np.linalg.solve(H_uu, H_um)
        c_su = float(np.linalg.cond(Su))
        c_sm = float(np.linalg.cond(Sm))
        eig_su = float(np.min(np.linalg.eigvalsh(Su)))
        eig_sm = float(np.max(np.linalg.eigvalsh(Sm)))
    except np.linalg.LinAlgError:
        base.update(cond_S_u=float("inf"), cond_S_m=float("inf"),
                    S_u_well_posed=False, S_m_well_posed=False,
                    boundary=True, reason="singular block in Schur complement")
        return base
    boundary = (not np.isfinite(c_su) or not np.isfinite(c_sm) or
                c_su > cond_max or c_sm > cond_max or
                eig_su <= eig_tol or eig_sm >= -eig_tol)
    base.update(cond_S_u=c_su, cond_S_m=c_sm,
                eig_S_u_min=eig_su, eig_S_m_max=eig_sm,
                S_u_well_posed=bool(eig_su > eig_tol and np.isfinite(c_su)),
                S_m_well_posed=bool(eig_sm < -eig_tol and np.isfinite(c_sm)),
                boundary=bool(boundary),
                reason="boundary/ill-conditioned Schur complement" if boundary
                else "well posed")
    return base


def _max_stable_interp(A, B, E, pol_cur, pol_dag, rho_limit=1.0,
                       bisect_tol=1e-10):
    """Largest beta in (0, 1] with rho(G_beta) < rho_limit (bisection).

    G_beta = beta G_dag + (1-beta) G_cur is explicit in beta (Remark L1254);
    no Lyapunov equation is re-solved during the stability line search.
    """
    G_cur = _closed_loop(A, B, E, pol_cur)
    G_dag = _closed_loop(A, B, E, pol_dag)

    def stable(beta):
        Gb = beta * G_dag + (1 - beta) * G_cur
        return rho(Gb) < rho_limit

    if not stable(0.0):
        raise RuntimeError("current policy is not stabilizing")
    if stable(1.0):
        return 1.0, G_dag
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < bisect_tol:
            break
    beta = lo
    Gb = beta * G_dag + (1 - beta) * G_cur
    return beta, Gb


def _one_stage_residual(A, B, E, Q, R, alpha, lam, v_bar, pol, P, chi, psi,
                        Xs):
    """Bellman residual L(x) = c̃(x) + alpha V_old(x') - V_old(x) for the
    candidate policy evaluated against the FIXED old value (Thm.4 proof,
    L1220-1227). Diagnostic only; the Alg.3 line search uses stability.
    """
    K, p, N, z = pol["K"], pol["p"], pol["N"], pol["z"]
    G = A + B @ K + E @ N
    F = B @ p + E @ z
    res = np.empty(len(Xs))
    for i, x in enumerate(Xs):
        u = K @ x + p
        w = N @ x + z
        xn = G @ x + F
        c = x @ Q @ x + u @ R @ u - lam * (w - v_bar) @ (w - v_bar)
        v_next = xn @ P @ xn + 2 * chi @ xn + psi
        v_cur = x @ P @ x + 2 * chi @ x + psi
        res[i] = c + alpha * v_next - v_cur
    return res


def _initial_stabilizing_policy(A, B, E, Q, R, alpha, lam, V, pol,
                                kappa_sc=0.9, max_rounds=30):
    """Return a stabilizing policy verified on the original system.

    The scaled DRC solve is used only to generate a candidate.  The acceptance
    test is always ``rho(A+BK+EN) < 1`` on the original matrices.  If the
    scaled candidates fail, the discounted LQR policy is used as a certified
    fallback and is checked by the same test.
    """
    beta = 1.0
    rho0 = rho(_closed_loop(A, B, E, pol))
    if rho0 < 1.0:
        return pol, dict(source="provided_stable", beta_scale=1.0,
                         rounds=0, initial_rho=rho0)

    candidate = pol
    attempts = []
    for i in range(1, max_rounds + 1):
        G = _closed_loop(A, B, E, candidate)
        rho_i = rho(G)
        beta_new = kappa_sc / max(rho_i, np.finfo(float).eps)
        beta *= beta_new
        sub = core.solve_drc_pi_g(beta * A, beta * B, beta * E,
                                   Q, R, alpha, lam, beta * V)
        mapped = scaled_policy_to_original(sub, beta)
        candidate = {k: mapped[k] for k in ("K", "p", "N", "z")}
        G_orig = _closed_loop(A, B, E, candidate)
        rho_orig = rho(G_orig)
        rho_scaled = rho(beta * G_orig)
        attempts.append(dict(round=i, rho_before=rho_i, beta=float(beta),
                             rho_scaled=float(rho_scaled),
                             rho_original=float(rho_orig)))
        if rho_orig < 1.0:
            return candidate, dict(
                source="scaled_candidate_verified_on_original",
                beta_scale=float(beta), rounds=i, initial_rho=rho0,
                mapping="K,p,N,z copied in unchanged coordinates",
                mapping_identity_residual=float(np.linalg.norm(
                    beta * G_orig - beta * _closed_loop(A, B, E, candidate))),
                scaled_candidate_attempts=attempts)

    Klqr, _, _, _ = core.solve_discounted_lqr(A, B, Q, R, alpha)
    fallback = dict(K=-Klqr, p=np.zeros(B.shape[1]),
                    N=np.zeros((E.shape[1], A.shape[0])),
                    z=np.zeros(E.shape[1]))
    if rho(_closed_loop(A, B, E, fallback)) >= 1.0:
        raise RuntimeError("no stabilizing initializer found on the original system")
    return fallback, dict(source="discounted_lqr_fallback", beta_scale=1.0,
                          rounds=max_rounds, initial_rho=rho0,
                          mapping="discounted LQR fallback; no scaled gain copied",
                          mapping_identity_residual=0.0,
                          scaled_candidate_attempts=attempts)


def _fixed_point_diagnostics(A, B, E, Q, R, alpha, lam, V, pol,
                             P, chi, psi):
    """Residuals of the three policy-evaluation equations."""
    G = _closed_loop(A, B, E, pol)
    F = B @ pol["p"] + E @ pol["z"]
    v_bar = np.atleast_2d(V).mean(axis=0)
    phiP = (Q + pol["K"].T @ R @ pol["K"]
            - lam * pol["N"].T @ pol["N"])
    resP = P - alpha * G.T @ P @ G - phiP
    rhs_c = (pol["K"].T @ R @ pol["p"]
             - lam * pol["N"].T @ (pol["z"] - v_bar)
             + alpha * G.T @ P @ F)
    reschi = (np.eye(A.shape[0]) - alpha * G.T) @ chi - rhs_c
    # At the constrained mean, every atom is displaced by m-v_bar.  Hence the
    # minimum dual penalty is -lambda||z_bar-v_bar||^2 and there is no extra
    # support-spread constant in psi.
    psi_rhs = ((pol["p"] @ R @ pol["p"]
                - lam * (pol["z"] - v_bar) @ (pol["z"] - v_bar)
                + alpha * F @ P @ F
                + 2 * alpha * F @ chi) / (1 - alpha))
    return dict(residual_P=float(np.linalg.norm(resP)),
                residual_chi=float(np.linalg.norm(reschi)),
                residual_psi=float(abs(psi - psi_rhs)),
                rho_G=float(rho(G)),
                stable=bool(rho(G) < 1.0))


def solve_sdpi(A, B, E, Q, R, alpha, lam, V,
               K0=None, p0=None, N0=None, z0=None,
               kappa_sc=0.9, eps=1e-8, max_phase1=30, max_phase2=300,
               bisect_tol=1e-10, n_value_samples=200, value_seed=0,
               track_history=True):
    """Run SDPI (verified initialization + Phase II stabilizing PI).

    Returns a dict with the final policy/value parameters, phase switch step,
    per-iteration log (rho(G), beta_scale, beta_interp, residual max, value
    monotonicity violations), and diagnostics.
    """
    nx, nu, nd = A.shape[0], B.shape[1], E.shape[1]
    V = np.atleast_2d(np.asarray(V, dtype=float))
    v_bar = V.mean(axis=0)

    # --- arbitrary initial policy (default: zero, deliberately non-stabilizing)
    K = np.zeros((nu, nx)) if K0 is None else np.array(K0, dtype=float)
    p = np.zeros(nu) if p0 is None else np.array(p0, dtype=float)
    N = np.zeros((nd, nx)) if N0 is None else np.array(N0, dtype=float)
    z = np.zeros(nd) if z0 is None else np.array(z0, dtype=float)

    beta_scale = 1.0
    history = []
    phase1_steps = 0
    switched_at = None

    # value-function sample grid for the Thm.4 monotonicity diagnostic
    rng = np.random.default_rng(value_seed)
    Xs = rng.standard_normal((n_value_samples, nx))

    def value_at(P, chi, psi, x):
        return float(x @ P @ x + 2 * chi @ x + psi)

    # ------------------- Phase I: verified initialization ------------------- #
    pol = dict(K=K, p=p, N=N, z=z)
    pol, init_info = _initial_stabilizing_policy(
        A, B, E, Q, R, alpha, lam, V, pol,
        kappa_sc=kappa_sc, max_rounds=max_phase1)
    beta_scale = float(init_info["beta_scale"])
    phase1_steps = int(init_info["rounds"])
    switched_at = len(history)
    history.append(dict(step=len(history), phase="I", beta_scale=beta_scale,
                         beta_interp=None,
                         rho_G=rho(_closed_loop(A, B, E, pol)),
                         residual_max=None, violations=None))

    # Evaluate the Phase-I stabilizing policy exactly
    ev = core.policy_evaluate(A, B, E, Q, R, alpha, lam, V,
                              pol["K"], pol["p"], pol["N"], pol["z"])
    P, chi, psi = ev["P"], ev["chi"], ev["psi"]

    monotonic_violations = 0
    stop_reason = "not_started"
    prev_val = np.array([value_at(P, chi, psi, x) for x in Xs])
    last_delta = dict(K=float("nan"), p=float("nan"),
                      N=float("nan"), z=float("nan"))

    # ------------------------- Phase II: stabilizing PI -------------------- #
    for i in range(max_phase2):
        # unconstrained improvement at the current value.  A singular Schur
        # complement is an assumption failure for this step; stop and report
        # it instead of silently using a pseudo-inverse.
        try:
            schur = _schur_diagnostics(A, B, E, Q, R, alpha, lam, P)
            if not (schur["S_u_well_posed"] and schur["S_m_well_posed"]):
                stop_reason = ("schur_complement_boundary"
                               if schur.get("boundary")
                               else "schur_complement_not_well_posed")
                break
            dag = core.policy_improve(A, B, E, Q, R, alpha, lam, V, P, chi)
        except np.linalg.LinAlgError:
            stop_reason = "schur_complement_singular"
            break
        pol_dag = dict(K=dag["K"], p=dag["p"], N=dag["N"], z=dag["z"])
        # one-stage residual of the greedy improvement against fixed V_old
        res_dag = _one_stage_residual(A, B, E, Q, R, alpha, lam, v_bar,
                                      pol_dag, P, chi, psi, Xs)
        residual_max = float(res_dag.max())
        # Alg. 3 conservative interpolation: largest stable beta
        beta_interp, G_new = _max_stable_interp(
            A, B, E, pol, pol_dag, rho_limit=1.0, bisect_tol=bisect_tol)
        pol_new = _interpolate(pol, pol_dag, beta_interp)
        ev_new = core.policy_evaluate(A, B, E, Q, R, alpha, lam, V,
                                      pol_new["K"], pol_new["p"],
                                      pol_new["N"], pol_new["z"])
        cur_val = np.array([value_at(ev_new["P"], ev_new["chi"], ev_new["psi"], x)
                            for x in Xs])
        viol = int(np.sum(cur_val > prev_val + 1e-7))
        monotonic_violations += viol

        dK = np.linalg.norm(pol_new["K"] - pol["K"], "fro")
        dp = np.linalg.norm(pol_new["p"] - pol["p"])
        dN = np.linalg.norm(pol_new["N"] - pol["N"], "fro")
        dz = np.linalg.norm(pol_new["z"] - pol["z"])
        delta_max = max(dK, dp, dN, dz)
        last_delta = dict(K=float(dK), p=float(dp),
                          N=float(dN), z=float(dz))
        history.append(dict(step=len(history), phase="II",
                            beta_scale=beta_scale, beta_interp=beta_interp,
                            rho_G=rho(G_new), residual_max=residual_max,
                            delta_K=dK, delta_p=dp, delta_N=dN, delta_z=dz,
                            delta_max=delta_max, violations=viol,
                            schur=schur))
        pol, P, chi, psi = pol_new, ev_new["P"], ev_new["chi"], ev_new["psi"]
        ev = ev_new
        prev_val = cur_val

        if beta_interp < eps:
            stop_reason = "interpolation_step"
            break
        if delta_max < eps:
            stop_reason = "stationarity"
            break
    else:
        stop_reason = "max_phase2"

    G = _closed_loop(A, B, E, pol)
    final_schur = _schur_diagnostics(A, B, E, Q, R, alpha, lam, P)
    fp = _fixed_point_diagnostics(A, B, E, Q, R, alpha, lam, V, pol,
                                  P, chi, psi)
    return dict(K=pol["K"], p=pol["p"], N=pol["N"], z=pol["z"],
                P=P, chi=chi, psi=psi, G=G, F=B @ pol["p"] + E @ pol["z"],
                rho_G=rho(G), phase1_steps=phase1_steps,
                switched_at=switched_at, history=history,
                monotonic_violations=monotonic_violations,
                phase2_steps=len(history) - phase1_steps - 1,
                initialization=init_info, stop_reason=stop_reason,
                final_delta=last_delta, final_schur=final_schur,
                fixed_point=fp,
                stopping=dict(reason=stop_reason, eps=float(eps),
                              max_phase2=int(max_phase2),
                              final_delta=last_delta))
