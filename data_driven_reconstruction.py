#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-driven output-window reconstruction used by the paper's Theorem 1.

The paper identifies an observability subspace from input-output windows.  In
general the identified subspace differs from the physical one by an
invertible similarity transform T.  Therefore the mathematically correct
identities are

    M_y O_l = T^{-1},
    M_u + M_y D_l = 0,
    T_d = M_y E_l,

in the equivalent coordinate x^T = T^{-1}x.  The earlier wording
``M_y O_l = I`` is valid only after fixing T=I.  This module implements the
general-coordinate version and returns the transform explicitly.

Disturbance-to-state map.  The paper's original phrase "estimate T_d from the
residual covariance" is not constructive: a covariance alone does not identify
an input-to-state map.  The implementable estimator regresses the output-window
residual on the measured disturbance windows,

    E_l_hat = tilde Y D_w^T (D_w D_w^T)^+,
    T_d_hat = M_y_hat E_l_hat,

and then projects it onto the identified state subspace.  This uses the
offline disturbance samples explicitly and is consistent under independent
excitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

import drc_core as core


@dataclass
class WindowDataset:
    U: np.ndarray          # (ell*nu, N)
    Y: np.ndarray          # (ell*ny, N)
    D: np.ndarray          # (ell*nd, N)
    X: np.ndarray          # (nx, N) delayed true states
    nx: int
    nu: int
    ny: int
    nd: int
    ell: int


def simulate_windows(spec: Dict, n_windows: int, ell: int, seed: int,
                     state_scale: float = 3.0,
                     input_scale: float = 1.5,
                     dist_scale: float = 1.0) -> WindowDataset:
    """Generate independent persistently excited input-output windows."""
    rng = np.random.default_rng(seed)
    A, B, C, E = spec["A"], spec["B"], spec["C"], spec["E"]
    nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    nd = E.shape[1]
    Us, Ys, Ds, Xs = [], [], [], []
    for _ in range(int(n_windows)):
        x = state_scale * rng.standard_normal(nx)
        Xs.append(x.copy())
        uwin = input_scale * rng.standard_normal((ell, nu))
        dwin = (spec["d_bar"][None, :]
                + dist_scale * spec["d_std"][None, :]
                * rng.standard_normal((ell, nd)))
        ywin = []
        xcur = x.copy()
        for k in range(ell):
            ywin.append(C @ xcur)
            xcur = A @ xcur + B @ uwin[k] + E @ dwin[k]
        Us.append(uwin.reshape(-1))
        Ys.append(np.concatenate(ywin))
        Ds.append(dwin.reshape(-1))
    return WindowDataset(
        U=np.asarray(Us).T, Y=np.asarray(Ys).T, D=np.asarray(Ds).T,
        X=np.asarray(Xs).T, nx=nx, nu=nu, ny=ny, nd=nd, ell=ell)


def fit_reconstruction(data: WindowDataset, spec: Dict,
                       rcond: float = 1e-10) -> Dict:
    """Fit M_u, M_y, T_d and the similarity transform from window data.

    The subspace step follows the paper's Hankel/SVD construction.  The
    disturbance map is estimated by the residual-vs-disturbance regression
    described in the module docstring.  The true O_l is used only to align the
    identified basis and report the equivalent-coordinate transform T; it is
    not used in reconstruction.
    """
    A, B, C, E = spec["A"], spec["B"], spec["C"], spec["E"]
    ell = data.ell
    O, D_l, E_l = core.output_window_maps(A, B, C, E, ell)

    # Regress out the known input window.
    D_hat = data.Y @ data.U.T @ np.linalg.pinv(
        data.U @ data.U.T, rcond=rcond)
    tildeY = data.Y - D_hat @ data.U
    U_s, s, _ = np.linalg.svd(tildeY, full_matrices=False)
    O_hat = U_s[:, :data.nx]

    # The disturbance-window residual contains O_l X + E_l D.  Since X and D
    # are independent under the excitation design, the D-regression estimates
    # E_l consistently.  Project the result through M_y.
    E_l_hat = tildeY @ data.D.T @ np.linalg.pinv(
        data.D @ data.D.T, rcond=rcond)
    M_y = np.linalg.pinv(O_hat, rcond=rcond)
    M_u = -M_y @ D_hat
    T_d = M_y @ E_l_hat

    # The identified basis is aligned as O_hat = O_l T.
    T = np.linalg.pinv(O, rcond=rcond) @ O_hat
    T_inv = np.linalg.pinv(T, rcond=rcond)
    # The ideal maps are defined in the reconstructed equivalent coordinate.
    # Comparing to these maps, rather than only checking an algebraic relation,
    # is what makes the N^-1/2 error test meaningful.
    M_y_star = T_inv @ np.linalg.pinv(O, rcond=rcond)
    M_u_star = -M_y_star @ D_l
    T_d_star = M_y_star @ E_l
    relation_y = M_y @ O - T_inv
    relation_u = M_u + M_y @ D_l
    return dict(
        M_y=M_y, M_u=M_u, M_y_star=M_y_star, M_u_star=M_u_star,
        T=T, T_inv=T_inv, T_d=T_d, T_d_star=T_d_star,
        O_hat=O_hat, O_true=O, D_l=D_l, E_l=E_l,
        singular_values=s,
        rank_hat=int(np.linalg.matrix_rank(O_hat)),
        rank_true=int(np.linalg.matrix_rank(O)),
        relation_identity_y=float(np.linalg.norm(relation_y)),
        relation_identity_u=float(np.linalg.norm(relation_u)),
        residual_identity_y=float(np.linalg.norm(M_y - M_y_star)),
        residual_identity_u=float(np.linalg.norm(M_u - M_u_star)),
        residual_identity_Td=float(np.linalg.norm(T_d - T_d_star)),
    )


def reconstruct(data: WindowDataset, fit: Dict, zeta: float = 0.0,
                support: Optional[np.ndarray] = None,
                physical: bool = False) -> Dict:
    """Apply the identified map and the minimax squared-loss center.

    For a squared loss and a Wasserstein ball, the minimax center is the
    empirical mean of the target.  The radius changes the worst-case risk but
    does not translate the center.  ``zeta`` is retained for API and reporting
    compatibility and does not alter the estimate.

    Parameters
    ----------
    data : WindowDataset
        Windowed data.  Any column can be used as a test window.
    fit : dict
        Output of :func:`fit_reconstruction`.
    zeta : float
        Wasserstein radius retained for the risk audit.
    support : ndarray, shape (g, nd), optional
        Empirical one-step support.  Its repeated window mean is used.
    physical : bool
        If true, multiply the equivalent state estimate by the identified T.
    """
    xhat_eq = fit["M_u"] @ data.U + fit["M_y"] @ data.Y
    if support is None:
        support = np.zeros((1, data.nd))
    support = np.atleast_2d(np.asarray(support, dtype=float))
    vbar = support.mean(axis=0)
    vbar_window = np.tile(vbar, data.ell)
    Td = fit["T_d"]
    correction = Td @ vbar_window
    xdr_eq = xhat_eq - correction[:, None]
    if physical:
        xdr = fit["T"] @ xdr_eq
        x_nom = fit["T"] @ xhat_eq
    else:
        xdr = xdr_eq
        x_nom = xhat_eq
    x_ref = data.X if physical else fit["T_inv"] @ data.X
    err_nom = np.linalg.norm(x_nom - x_ref, axis=0)
    err_rob = np.linalg.norm(xdr - x_ref, axis=0)
    return dict(x_hat=xhat_eq, x_dr=xdr_eq, x_physical=xdr,
                x_ref=x_ref, err_nominal=err_nom, err_robust=err_rob,
                mean_err_nominal=float(np.mean(err_nom)),
                mean_err_robust=float(np.mean(err_rob)),
                ustar=np.zeros(Td.shape[1]),
                vbar_window=vbar_window,
                correction_norm=float(np.linalg.norm(correction)),
                zeta_center_shift=0.0)


def reconstruction_rate_scan(spec: Dict, Ns: Sequence[int] = (1000, 2000,
                                                             4000, 8000),
                             seeds: Sequence[int] = (100, 101, 102),
                             ell: int = 3) -> Dict:
    """Empirical N^-1/2 scan for the statistical reconstruction term."""
    per_n = {}
    for N in Ns:
        sub_err, y_err, u_err, td_err = [], [], [], []
        for seed in seeds:
            train = simulate_windows(spec, N, ell, seed)
            fit = fit_reconstruction(train, spec)
            test = simulate_windows(spec, 300, ell, seed + 10000)
            rec = reconstruct(test, fit, zeta=0.0,
                              support=spec["d_bar"].reshape(1, -1))
            # Equivalent-coordinate reference for the identified basis.
            xeq = fit["T_inv"] @ test.X
            xhat = fit["M_u"] @ test.U + fit["M_y"] @ test.Y
            e = np.linalg.norm(xhat - xeq, axis=0)
            sub_err.append(np.linalg.norm(fit["M_y"] - fit["M_y_star"]))
            y_err.append(float(np.mean(e)))
            u_err.append(fit["residual_identity_u"])
            td_err.append(fit["residual_identity_Td"])
        per_n[int(N)] = dict(
            subspace=float(np.mean(sub_err)),
            reconstruction=float(np.mean(y_err)),
            input_identity=float(np.mean(u_err)),
            disturbance_identity=float(np.mean(td_err)),
        )
    from data_driven import loglog_slope
    slopes = {k: loglog_slope(list(per_n),
                              [per_n[N][k] for N in per_n])
              for k in next(iter(per_n.values()))}
    return dict(per_n=per_n, slopes=slopes, ell=int(ell),
                Ns=list(Ns), seeds=list(seeds))
