#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard-Wasserstein diagnostics and controller-specific adversarial rollouts.

For an empirical support ``V = (v_1,...,v_g)`` and an affine adversary
``omega_j = N x + z_j`` with
``z_j = v_j - v_bar + z_bar``, every atom has the same transport displacement
``omega_j-v_j = N x + z_bar - v_bar``.  Consequently the empirical 2-Wasserstein
radius of the induced atomwise worst-case distribution is

    W2(x) = ||N x + z_bar - v_bar||_2.

The functions below verify that identity directly on the support atoms, compute
both the trajectory RMS and maximum radius, search the outer penalty ``lambda``
for ball-feasible candidates, and build a separate adversary for DRC and for
the LQR baseline.  All hard-ball statements produced here are finite-horizon,
sample-trajectory certificates, not proofs of global Bellman optimality.
"""

import warnings

import numpy as np

import drc_core as core

HARD_BALL_SCOPE = (
    "finite_horizon_sampled_trajectory_certificate; "
    "not_a_global_Bellman_optimality_proof"
)


def _as_support(V, nd):
    V = np.asarray(V, dtype=float).reshape(-1, nd)
    if V.shape[0] == 0:
        raise ValueError("the empirical support must contain at least one atom")
    return V, V.mean(axis=0)


def lqr_value_policy(spec, max_iter=20000, tol=1e-13):
    """Discounted LQR policy and value matrix, with ``u=-K(x-x_ref)``."""
    A, B, Q, R, alpha = spec["A"], spec["B"], spec["Q"], spec["R"], spec["alpha"]
    nx = A.shape[0]
    P = np.eye(nx)
    K_ctrl = None
    for _ in range(max_iter):
        K_step = np.linalg.solve(
            R + alpha * B.T @ P @ B, alpha * B.T @ P @ A)
        K_ctrl = -K_step
        Acl = A + B @ K_ctrl
        Pn = Q + K_ctrl.T @ R @ K_ctrl + alpha * Acl.T @ P @ Acl
        if np.linalg.norm(Pn - P) < tol:
            P = Pn
            break
        P = Pn
    if K_ctrl is None or not np.all(np.isfinite(P)):
        raise RuntimeError("LQR Riccati iteration failed")
    return K_ctrl, P


def lqr_own_adversary(spec, lam, V=None):
    """LQR-specific affine maximizer of its fixed-``lambda`` inner problem.

    The adversary is derived from the LQR value matrix and the LQR gain; it is
    never copied from the DRC solution.  For a general empirical support V, its
    per-atom offsets satisfy ``z_j = v_j-v_bar+z_bar``.
    """
    A, B, E = spec["A"], spec["B"], spec["E"]
    nd = E.shape[1]
    V, vbar = _as_support(spec["d_bar"] if V is None else V, nd)
    x_ref = np.asarray(spec["x_ref"], dtype=float).reshape(-1)
    K_ctrl, P = lqr_value_policy(spec)
    Acl = A + B @ K_ctrl
    Ey = E.T @ P
    lam = float(lam)
    lam_crit = float(spec["alpha"] * np.max(np.linalg.eigvalsh(Ey @ E)))
    M = np.eye(nd) - (spec["alpha"] / lam) * (Ey @ E)
    if float(np.min(np.linalg.eigvalsh(M))) <= 0.0:
        raise ValueError("LQR penalized inner problem is not strictly concave")
    N_e = np.linalg.solve(M, (spec["alpha"] / lam) * Ey @ Acl)
    # d = M^{-1}vbar + N_e (x-x_ref) = N_e x + zbar.
    d0 = np.linalg.solve(M, vbar)
    zbar = d0 - N_e @ x_ref
    zj = V - vbar[None, :] + zbar[None, :]
    G = A + B @ K_ctrl + E @ N_e
    F = E @ zbar
    return dict(
        K=K_ctrl, p=np.zeros(B.shape[1]), N=N_e, z=zbar, zj=zj,
        P=P, G=G, F=F, Acl=Acl, E=E, v_bar=vbar, lam=lam,
        lam_crit=lam_crit, inner_well_posed=bool(lam > lam_crit),
        policy_type="lqr_own_penalized_dual_adversary",
    )


def _policy_for_controller(spec, controller, lam, V, max_iter=5000):
    if controller == "drc":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sol = core.solve_drc_pi_g(
                    spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
                    spec["alpha"], float(lam), V, max_iter=max_iter)
        pol = {k: sol[k] for k in ("K", "p", "N", "z", "zj", "P", "chi",
                                   "psi", "G", "F", "v_bar", "rho_G",
                                   "H_uu_min_eig", "H_ww_min_eig")}
        pol.update(lam=float(lam), policy_type="drc_own_penalized_dual_adversary")
        pol["solver_converged"] = bool(sol["converged"])
        pol["solver_iters"] = int(sol["iters"])
        pol["lambda_crit"] = float(sol["lambda_crit"])
        return pol
    if controller == "lqr":
        pol = lqr_own_adversary(spec, float(lam), V=V)
        pol["solver_converged"] = True
        pol["solver_iters"] = 0
        pol["rho_G"] = float(np.max(np.abs(np.linalg.eigvals(pol["G"]))))
        pol["H_uu_min_eig"] = float("nan")
        pol["H_ww_min_eig"] = float("nan")
        return pol
    raise ValueError("controller must be 'drc' or 'lqr'")


def _rollout_policy(spec, pol, T=None):
    """Deterministic rollout under the policy's own affine adversary."""
    A, B, E = spec["A"], spec["B"], spec["E"]
    x_ref = np.asarray(spec["x_ref"], dtype=float)
    x = x_ref.copy()
    T = spec["T"] if T is None else int(T)
    xs, us, ds = [], [], []
    diverged = False
    for _ in range(T):
        e = x - x_ref
        u = pol["K"] @ e + pol["p"]
        d = pol["N"] @ x + pol["z"]
        xs.append(x.copy())
        us.append(np.asarray(u, dtype=float).copy())
        ds.append(np.asarray(d, dtype=float).copy())
        x = A @ x + B @ u + E @ d
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e8:
            diverged = True
            break
    if not xs:
        return (np.empty((0, A.shape[0])), np.empty((0, B.shape[1])),
                np.empty((0, E.shape[1])), True)
    return np.asarray(xs), np.asarray(us), np.asarray(ds), diverged


def atomwise_w2_diagnostics(spec, pol, V=None, T=None):
    """Check W2 membership explicitly on every support atom.

    For every time ``k`` the returned ``w2_max`` is the maximum over the
    trajectory of the empirical atomwise distribution radius; ``w2_rms`` is
    ``sqrt(mean_k W2(x_k)^2)``.  Membership uses ``w2_max <= zeta``.
    """
    nd = spec["E"].shape[1]
    V, vbar = _as_support(spec["d_bar"] if V is None else V, nd)
    xs, _, dmean, diverged = _rollout_policy(spec, pol, T=T)
    if xs.size == 0 or diverged:
        return dict(
            w2_rms=float("inf"), w2_max=float("inf"),
            w2_final=float("inf"), atom_w2_max=float("inf"),
            atom_dispersion=float("inf"), mean_identity_error=float("inf"),
            T=int(len(xs)), diverged=True,
            support_size=int(V.shape[0]), scope=HARD_BALL_SCOPE,
        )
    zbar = np.asarray(pol["z"], dtype=float).reshape(1, -1)
    N = np.asarray(pol["N"], dtype=float)
    mean_adv = xs @ N.T + zbar
    # All support atoms share the same displacement.  Constructing the atoms
    # explicitly catches an inconsistent z_j implementation or a bad support.
    omega = mean_adv[:, None, :] + (V - vbar)[None, :, :]
    delta = omega - V[None, :, :]
    atom_sq = np.sum(delta ** 2, axis=2)
    w2_k = np.sqrt(np.mean(atom_sq, axis=1))
    atom_disp = np.max(np.ptp(delta, axis=1)) if delta.shape[1] > 1 else 0.0
    mean_identity_error = float(
        np.max(np.abs(np.mean(omega, axis=1) - mean_adv)))
    return dict(
        w2_rms=float(np.sqrt(np.mean(w2_k ** 2))),
        w2_max=float(np.max(w2_k)),
        w2_final=float(w2_k[-1]),
        atom_w2_max=float(np.max(np.sqrt(np.max(atom_sq, axis=1)))),
        atom_dispersion=float(atom_disp),
        mean_identity_error=mean_identity_error,
        T=int(len(xs)), diverged=False,
        support_size=int(V.shape[0]), scope=HARD_BALL_SCOPE,
    )


def w2_radius_along_trajectory(spec, pol, vbar=None, T=None, V=None):
    """Backward-compatible wrapper around :func:`atomwise_w2_diagnostics`."""
    if V is None:
        center = spec["d_bar"] if vbar is None else vbar
        V = np.asarray(center, dtype=float).reshape(1, -1)
    return atomwise_w2_diagnostics(spec, pol, V=V, T=T)


def trajectory_cost(spec, pol, T=None):
    """Discounted stage cost along the policy's own adversarial trajectory."""
    xs, us, _, diverged = _rollout_policy(spec, pol, T=T)
    if xs.size == 0 or diverged:
        return dict(cost=float("inf"), mae=float("inf"),
                    diverged=True, T=int(len(xs)))
    x_ref = spec["x_ref"]
    e = xs - x_ref
    stage = np.einsum("ki,ij,kj->k", e, spec["Q"], e)
    stage += np.einsum("ki,ij,kj->k", us, spec["R"], us)
    alpha = spec["alpha"]
    weights = alpha ** np.arange(len(stage))
    cost = float(np.sum(weights * stage) / (1.0 - alpha))
    if spec["mae_mode"] == "freq":
        mae = float(np.mean(np.linalg.norm(e[:, [0, 2]], axis=1)))
    elif spec["mae_mode"] == "angle":
        mae = float(np.mean(np.linalg.norm(e[:, [0, 2, 4]], axis=1)))
    else:
        mae = float(np.mean(np.linalg.norm(e, axis=1)))
    return dict(cost=cost, mae=mae, diverged=False, T=int(len(xs)))


def finite_horizon_lagrangian(spec, pol, zeta, T=None):
    """Finite-horizon Lagrangian surrogate for the hard-ball outer problem.

    It is the discounted finite-horizon cost with the atomwise transport
    penalty and terminal value, plus ``lambda*zeta^2``.  It is used only to rank
    sampled ball-feasible candidates; it is not a global Bellman certificate.
    """
    xs, us, ds, diverged = _rollout_policy(spec, pol, T=T)
    if xs.size == 0 or diverged:
        return float("inf")
    x_ref = np.asarray(spec["x_ref"], dtype=float)
    e = xs - x_ref
    stage = np.einsum("ki,ij,kj->k", e, spec["Q"], e)
    stage += np.einsum("ki,ij,kj->k", us, spec["R"], us)
    # The atomwise empirical transport cost reduces exactly to the mean
    # displacement cost; this is also checked by atomwise_w2_diagnostics.
    transport_sq = np.sum((ds - np.asarray(pol.get("v_bar", spec["d_bar"]))
                           [None, :]) ** 2, axis=1)
    alpha = spec["alpha"]
    weights = alpha ** np.arange(len(stage))
    xT = xs[-1]
    if "chi" in pol and "psi" in pol:
        terminal = float(xT @ pol["P"] @ xT + 2.0 * pol["chi"] @ xT
                         + pol["psi"])
    else:
        eT = xT - x_ref
        terminal = float(eT @ pol["P"] @ eT)
    obj = (float(pol["lam"]) * float(zeta) ** 2
           + float(np.sum(weights * (stage - float(pol["lam"]) * transport_sq)))
           + float(alpha ** len(stage) * terminal))
    return float(obj)


def default_lambda_grid(spec, controller):
    """Outer-lambda grid (experimental design, not result values)."""
    if controller == "drc" and spec.get("name") == "microgrid":
        return np.asarray([3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0, 15.0,
                           20.0, 35.0, 50.0, 75.0, 100.0, 200.0, 500.0,
                           1000.0, 2000.0, 5000.0, 10000.0, 20000.0])
    if controller == "lqr" and spec.get("name") == "microgrid":
        return np.asarray([2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0,
                           35.0, 50.0, 75.0, 100.0, 200.0, 500.0, 1000.0,
                           2000.0, 5000.0, 10000.0, 20000.0])
    return np.asarray([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0,
                       20.0, 35.0, 50.0, 75.0, 100.0, 200.0, 500.0,
                       1000.0, 2000.0, 5000.0, 10000.0, 20000.0])


def _evaluate_candidate(spec, controller, lam, zeta, V, T, max_iter):
    rec = dict(
        lam=float(lam), feasible=False, status="failed",
        failure_reason=None, w2_rms=None, w2_max=None, w2_final=None,
        atom_w2_max=None, atom_dispersion=None, mean_identity_error=None,
        rho_G=None, H_uu_min=None, H_ww_min=None,
        constraint_slack=None, complementarity=None,
        constraint_active=False, trajectory=None, outer_objective=None,
        T=int(T), scope=HARD_BALL_SCOPE,
    )
    try:
        pol = _policy_for_controller(spec, controller, lam, V,
                                     max_iter=max_iter)
    except Exception as exc:
        rec["failure_reason"] = f"solver_exception:{type(exc).__name__}"
        return rec, None

    rec.update(
        rho_G=float(pol["rho_G"]),
        H_uu_min=float(pol["H_uu_min_eig"]),
        H_ww_min=float(pol["H_ww_min_eig"]),
        solver_converged=bool(pol["solver_converged"]),
        solver_iters=int(pol["solver_iters"]),
    )
    if not pol["solver_converged"]:
        rec["failure_reason"] = "fixed_point_not_converged"
    elif not np.all(np.isfinite(pol["P"])) or not np.all(np.isfinite(pol["N"])):
        rec["failure_reason"] = "nonfinite_policy"
    elif not np.isfinite(rec["rho_G"]) or rec["rho_G"] >= 1.0:
        rec["failure_reason"] = "closed_loop_unstable"
    elif np.isfinite(rec["H_uu_min"]) and rec["H_uu_min"] <= 0.0:
        rec["failure_reason"] = "input_Hessian_not_positive"
    elif np.isfinite(rec["H_ww_min"]) and rec["H_ww_min"] >= 0.0:
        rec["failure_reason"] = "adversary_Hessian_not_negative"

    rad = atomwise_w2_diagnostics(spec, pol, V=V, T=T)
    rec.update(
        w2_rms=rad["w2_rms"], w2_max=rad["w2_max"],
        w2_final=rad["w2_final"], atom_w2_max=rad["atom_w2_max"],
        atom_dispersion=rad["atom_dispersion"],
        mean_identity_error=rad["mean_identity_error"],
    )
    if rad["diverged"]:
        rec["failure_reason"] = rec["failure_reason"] or "adversarial_rollout_diverged"
        return rec, pol

    if rec["mean_identity_error"] > 1e-10:
        rec["failure_reason"] = rec["failure_reason"] or "atom_mean_identity_failed"
    if rec["atom_dispersion"] > 1e-9:
        rec["failure_reason"] = rec["failure_reason"] or "atom_displacement_mismatch"

    tol = 1e-10 * max(1.0, float(zeta))
    if rec["w2_max"] > float(zeta) + tol:
        rec["failure_reason"] = rec["failure_reason"] or "W2_exceeds_zeta"

    tr = trajectory_cost(spec, pol, T=T)
    rec["trajectory"] = tr
    if tr["diverged"]:
        rec["failure_reason"] = rec["failure_reason"] or "cost_rollout_diverged"
    else:
        rec["outer_objective"] = finite_horizon_lagrangian(
            spec, pol, zeta, T=T)

    slack = float(zeta) - float(rec["w2_max"])
    rec["constraint_slack"] = slack
    rec["complementarity"] = float(lam) * max(0.0, slack)
    active_tol = 1e-3 * max(float(zeta), 1e-12)
    rec["constraint_active"] = bool(abs(slack) <= active_tol)
    rec["feasible"] = bool(rec["failure_reason"] is None)
    rec["status"] = "feasible" if rec["feasible"] else "infeasible"
    if rec["feasible"]:
        rec["failure_reason"] = "none"
    return rec, pol


def _public_record(rec, pol):
    out = {k: v for k, v in rec.items() if k != "policy"}
    out["policy"] = pol
    out["lam"] = float(rec["lam"])
    out["objective"] = rec["outer_objective"]
    out["selection_kind"] = ("feasible_min_finite_horizon_lagrangian")
    return out


def find_ball_feasible_policy(spec, zeta, controller="drc", lam_grid=None,
                              T=None, V=None, max_iter=5000,
                              refine_boundary=True):
    """Search outer ``lambda`` for finite-horizon hard-ball feasible candidates.

    ``W2<=zeta`` is checked on every support atom along the sampled adversarial
    trajectory.  Separate calls with ``controller='drc'`` and
    ``controller='lqr'`` construct the two controllers' own adversaries.
    """
    nd = spec["E"].shape[1]
    V, vbar = _as_support(spec["d_bar"] if V is None else V, nd)
    T = int(spec["T"] if T is None else T)
    if lam_grid is None:
        lam_grid = default_lambda_grid(spec, controller)
    lam_grid = np.unique(np.asarray(lam_grid, dtype=float))

    cache = {}

    def get(lam):
        key = float(lam)
        if key not in cache:
            cache[key] = _evaluate_candidate(
                spec, controller, key, float(zeta), V, T, max_iter)
        return cache[key]

    initial = []
    for lam in lam_grid:
        rec, pol = get(lam)
        initial.append((rec, pol))

    ordered = sorted(initial, key=lambda x: x[0]["lam"])
    boundary = []
    if refine_boundary:
        for (lo_rec, _), (hi_rec, _) in zip(ordered[:-1], ordered[1:]):
            if (not lo_rec["feasible"]) and hi_rec["feasible"]:
                a, b = float(lo_rec["lam"]), float(hi_rec["lam"])
                best = hi_rec
                for _ in range(32):
                    mid = float(np.sqrt(a * b))
                    fm, _ = get(mid)
                    if fm["feasible"]:
                        b, best = mid, fm
                    else:
                        a = mid
                boundary.append(best)

    all_recs = [rec for rec, _ in initial] + boundary
    feasible = [r for r in all_recs if r["feasible"]]
    failure_counts = {}
    for r in all_recs:
        reason = r.get("failure_reason") or "unknown"
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    if not feasible:
        return dict(
            feasible=False, zeta=float(zeta), controller=controller,
            V=V, v_bar=vbar, selected=None, boundary=None,
            candidates=[r for r, _ in initial] + boundary,
            failure_counts=failure_counts, T=T, scope=HARD_BALL_SCOPE,
        )

    key = lambda r: (float(r["outer_objective"]), float(r["lam"]))
    best = min(feasible, key=key)
    pol = get(best["lam"])[1]
    selected = _public_record(best, pol)

    boundary_best = None
    if boundary:
        bbest = min(boundary, key=key)
        boundary_best = _public_record(bbest, get(bbest["lam"])[1])

    selected["boundary_lambda"] = (
        None if boundary_best is None else boundary_best["lam"])
    selected["boundary_w2_rms"] = (
        None if boundary_best is None else boundary_best["w2_rms"])
    selected["boundary_w2_max"] = (
        None if boundary_best is None else boundary_best["w2_max"])
    selected["boundary_complementarity"] = (
        None if boundary_best is None else boundary_best["complementarity"])
    selected["boundary_active"] = (
        False if boundary_best is None else boundary_best["constraint_active"])
    selected["selection_kind"] = "feasible_min_finite_horizon_lagrangian"
    selected["scope"] = HARD_BALL_SCOPE
    return dict(
        feasible=True, zeta=float(zeta), controller=controller,
        V=V, v_bar=vbar, selected=selected, boundary=boundary_best,
        policy=pol, candidates=[r for r, _ in initial] + boundary,
        failure_counts=failure_counts, T=T, scope=HARD_BALL_SCOPE,
    )


def compare_controllers_on_balls(spec, zetas, V=None, T=None):
    """At each radius, solve the DRC and LQR problems with separate adversaries."""
    out = {}
    for z in zetas:
        out[float(z)] = dict(
            drc=find_ball_feasible_policy(spec, z, "drc", V=V, T=T),
            lqr=find_ball_feasible_policy(spec, z, "lqr", V=V, T=T),
        )
    return out
