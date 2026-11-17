#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theorem 3 (L1103-1163) necessary-and-sufficient stability criterion.

Given the optimal policy value parameter P* ≻ 0, G* = A + B K* + E N* and
Phi_P* = Q + K*' R K* - lambda N*' N*, the criterion is

    rho(G*) < 1  <=>  exists symmetric X such that
        alpha P* + X  ≻  0,
        Phi_P* + (alpha - 1) P* + X - G*' X G*  ≻  0.

Two independent checks are implemented:

  1. Construction direction (proof L1125-1143): when rho(G*) < 1, solve the
     undiscounted Lyapunov equation P0 - G*' P0 G* = Q0 (Q0 ≻ 0, default I) and
     set X = P0 - alpha P*. Then alpha P* + X = P0 ≻ 0 and the second LMI
     equals Q0 ≻ 0 exactly. This proves the feasible direction constructively.

  2. SDP feasibility (cvxpy + CLARABEL/SCS): the two LMIs are posed directly as
     a feasibility problem. For rho(G*) < 1 it is feasible; for a deliberately
     destabilized G_bad it is reported infeasible.

Honest scope: the infeasible direction relies on (a) the pure Lyapunov SDP
``Ptilde - G'Ptilde G ≻ 0`` being reported infeasible, together with (b) the
standard discrete Lyapunov necessary condition: if such a PD Ptilde existed
then rho(G) < 1, contradicting the measured rho(G) ≥ 1. The original-form SDP
keeps P*, PhiP* fixed while G is perturbed; the matched identity
PhiP* - P* + alpha G*' P* G* = 0 no longer holds for the perturbed G, so its
residual bracket can mask instability and its feasible status alone is not a
Thm.3 contradiction — this is reported separately, not used as the infeasibility
evidence. We do not claim a separate S-procedure dual certificate beyond the
solver status.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:  # pragma: no cover
    _HAS_CVXPY = False


def _psd_min_eig(M):
    return float(np.min(np.linalg.eigvalsh((M + M.T) / 2)))


def construct_feasible_X(G, P, PhiP, alpha, Q0=None):
    """Proof construction (L1125-1143). Requires rho(G) < 1."""
    nx = G.shape[0]
    if Q0 is None:
        Q0 = np.eye(nx)
    # scipy solves P0 - A_l P0 A_l' = Q0; choose A_l = G'
    P0 = solve_discrete_lyapunov(G.T, Q0)
    X = P0 - alpha * P
    C1 = alpha * P + X
    C2 = PhiP + (alpha - 1) * P + X - G.T @ X @ G
    return dict(X=X, P0=P0, Q0=Q0,
                eig_C1=_psd_min_eig(C1), eig_C2=_psd_min_eig(C2),
                C1=C1, C2=C2)


def sdp_feasibility(G, P, PhiP, alpha, solver=None):
    """Direct SDP feasibility of the two LMIs (Theorem 3, original form).

    Returns dict with status, solved X (if any), min eigenvalues.
    """
    if not _HAS_CVXPY:
        return dict(status="cvxpy-not-installed", feasible=None, X=None)
    nx = G.shape[0]
    X = cp.Variable((nx, nx), symmetric=True)
    C1 = alpha * P + X
    C2 = PhiP + (alpha - 1) * P + X - G.T @ X @ G
    margin = 1e-8 * np.eye(nx)
    constraints = [C1 >> margin, C2 >> margin]
    prob = cp.Problem(cp.Minimize(0), constraints)
    if solver is None:
        solver = "CLARABEL" if "CLARABEL" in cp.installed_solvers() else "SCS"
    try:
        prob.solve(solver=solver)
    except Exception as exc:  # pragma: no cover
        return dict(status=f"solver-error: {exc}", feasible=None, X=None)
    feasible = prob.status in ("optimal", "optimal_inaccurate")
    X_val = X.value if X.value is not None else None
    eig1 = eig2 = None
    if X_val is not None:
        eig1 = _psd_min_eig(C1.value)
        eig2 = _psd_min_eig(C2.value)
    return dict(status=prob.status, feasible=feasible, X=X_val,
                eig_C1=eig1, eig_C2=eig2, solver=solver)


def lyapunov_sdp(G, margin=1e-8, eig_tol=1e-7, solvers=("CLARABEL", "SCS")):
    """Pure Lyapunov feasibility: exists Ptilde ≻ 0 with Ptilde - G'Ptilde G ≻ 0.

    This is the necessity-clean form of the criterion: when the matched
    identity PhiP - P + alpha G'PG = 0 holds, the second Thm.3 LMI reduces to
    Ptilde - G'Ptilde G ≻ 0. For rho(G) >= 1 this problem is infeasible by the
    standard discrete Lyapunov necessary condition.

    Post-solve the PD margin and the residual Lyapunov inequality are checked
    directly; a solver "optimal" whose matrix eigenvalues are at the numerical
    floor is reported infeasible-marginal, not feasible.
    """
    rho_G = float(np.max(np.abs(np.linalg.eigvals(G))))
    if not _HAS_CVXPY:
        return dict(status="cvxpy-not-installed", feasible=None, rho_G=rho_G)
    nx = G.shape[0]
    Pt = cp.Variable((nx, nx), symmetric=True)
    M = margin * np.eye(nx)
    constraints = [Pt >> M, Pt - G.T @ Pt @ G >> M]
    prob = cp.Problem(cp.Minimize(0), constraints)
    last = None
    for sv in solvers:
        if sv not in cp.installed_solvers():
            continue
        try:
            prob.solve(solver=sv)
        except Exception as exc:
            last = f"{sv}-error: {exc}"
            continue
        last = prob.status
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            return dict(status=prob.status, feasible=False, solver=sv,
                        rho_G=rho_G)
        if prob.status in ("optimal", "optimal_inaccurate") and Pt.value is not None:
            Pt_v = (Pt.value + Pt.value.T) / 2
            eig_Pt = float(np.min(np.linalg.eigvalsh(Pt_v)))
            eig_res = float(np.min(np.linalg.eigvalsh(
                Pt_v - G.T @ Pt_v @ G)))
            if eig_Pt > eig_tol and eig_res > eig_tol:
                return dict(status=prob.status, feasible=True, solver=sv,
                            eig_Pt_min=eig_Pt, eig_res_min=eig_res, rho_G=rho_G)
            last = f"{sv}-marginal"
            return dict(status=last, feasible=False, solver=sv,
                        eig_Pt_min=eig_Pt, eig_res_min=eig_res, rho_G=rho_G)
    return dict(status=last, feasible=False, solver=None, rho_G=rho_G)


def check_criterion(G, P, PhiP, alpha, label=""):
    """Run both construction and SDP checks; return a report dict."""
    rho_G = float(np.max(np.abs(np.linalg.eigvals(G))))
    rep = dict(label=label, rho_G=rho_G, rho_lt1=rho_G < 1.0)

    if rho_G < 1.0:
        con = construct_feasible_X(G, P, PhiP, alpha)
        rep["construction"] = dict(
            eig_C1=con["eig_C1"], eig_C2=con["eig_C2"],
            feasible=bool(con["eig_C1"] > 0 and con["eig_C2"] > 0))
    else:
        rep["construction"] = None

    sdp = sdp_feasibility(G, P, PhiP, alpha)
    rep["sdp"] = sdp
    rep["lyapunov_sdp"] = lyapunov_sdp(G)
    # Necessary-condition verdict (standard discrete Lyapunov theorem):
    # rho(G) >= 1 => no PD Ptilde with Ptilde - G'Ptilde G >> 0.
    rep["necessary_infeasible"] = rho_G >= 1.0
    return rep


def destabilize(G, rho_target=1.02):
    """Scale G so its spectral radius exceeds 1: G_bad = rho_target G / rho(G)."""
    rho_G = float(np.max(np.abs(np.linalg.eigvals(G))))
    return rho_target * G / rho_G, rho_G
