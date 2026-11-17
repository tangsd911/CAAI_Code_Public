#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core solver for the affine distributionally-robust control (DRC) verification
package.

Model (discrete-time LTI, forward-Euler discretization used by the paper)::

    x_{k+1} = A x_k + B u_k + E d_k

Affine control policy  u = K x + p ;  adversarial affine disturbance
omega^j = N x + z^j  (j = 1..g support points), and the dynamics consume the
support-point mean  omega_bar = (1/g) sum_j omega^j.

Mathematics (kept identical to the independently derived fixed-point iteration
in the legacy verify_independent.py; only the implicit global Q/R state and the
g=1 hard coding are removed):

  Relaxed one-stage minimax (fixed Lagrangian penalty lambda > 0)::

      J = x'Qx + u'Ru - (lambda/g) sum_j ||omega^j - v_j||^2 + alpha V(x+)
      x+ = A x + B u + E omega_bar

  Three-term value function  V(x) = x' P x + 2 chi' x + psi  (the "2 chi"
  convention is equivalent to the paper's V=x'Px+chi'x+psi; it rescales chi by
  2 and leaves K,p,N,z unchanged).

  Block Hessian for the mean adversarial variable m = omega_bar::

      H_uu = R + alpha B'PB,  H_um = alpha B'PE,
      H_mm = -lambda I + alpha E'PE

  FOC block system::

      [H_uu H_um; H_um' H_mm] [K; N] = -[alpha A'PB; alpha A'PE]
      [H_uu H_um; H_um' H_mm] [p; z] = -[alpha B'chi; alpha E'chi + lambda v_bar]

  Policy evaluation of the closed loop G = A+BK+EN, F = Bp + E z_bar::

      Phi_P = Q + K'RK - lambda N'N
      P - alpha G' P G = Phi_P
      (I - alpha G') chi = K'Rp - lambda N'(z_bar - v_bar) + alpha G' P F
      psi = [p'Rp - lambda(z_bar-v_bar)'(z_bar-v_bar)
             + alpha F'PF + 2 alpha F'chi] / (1-alpha)

General-g reduction (rigorous; this is why the general-g solution coincides
with g=1 when the support means agree):

  For a fixed mean m = omega_bar, the separable penalty is minimized under the
  constraint (1/g) sum_j omega^j = m. The minimizer is
      omega^j = v_j + (m - v_bar),
  and the minimum penalty equals
      -(lambda/g) sum_j ||omega^j-v_j||^2 = -lambda ||m - v_bar||^2.
  There is no additional support-spread term: every displaced atom has the
  same transport displacement m - v_bar. Hence the optimal (K, p, N, z_bar)
  depend solely on v_bar, and the per-support affine offsets are
      z^j = v_j - v_bar + z_bar.
"""

import numpy as np

# --------------------------------------------------------------------------- #
#  DRC fixed-point iteration (general g)
# --------------------------------------------------------------------------- #
def solve_drc_pi_g(A, B, E, Q, R, alpha, lam, V, max_iter=20000, tol=1e-12,
                   track_history=False):
    """General-support affine DRC fixed-point (Riccati) iteration.

    Parameters
    ----------
    A, B, E : ndarray
        Discrete-time system matrices.
    Q, R : ndarray
        State / input weighting matrices (explicit, no global state).
    alpha : float
        Discount factor in (0, 1).
    lam : float
        Fixed Wasserstein Lagrangian penalty lambda.
    V : ndarray, shape (g, n_d)
        Empirical support points v_j. g=1 reproduces the legacy solver exactly.

    Returns
    -------
    dict with P, chi, psi, K, p, N, z (z_bar), zj (per-support offsets),
    G, F, v_bar, iters, converged, rho_G, lam_min_P, H_uu_min_eig,
    H_ww_min_eig, lambda_crit.
    """
    nx = A.shape[0]
    nu = B.shape[1]
    nd = E.shape[1]
    V = np.atleast_2d(np.asarray(V, dtype=float))
    g = V.shape[0]
    v_bar = V.mean(axis=0)

    P = np.eye(nx)
    chi = np.zeros(nx)
    converged = False
    it = 0
    history = []

    for it in range(1, max_iter + 1):
        H_uu = R + alpha * B.T @ P @ B
        H_um = alpha * B.T @ P @ E
        H_mm = -lam * np.eye(nd) + alpha * E.T @ P @ E
        Hb = np.block([[H_uu, H_um],
                       [H_um.T, H_mm]])
        # quadratic (x) coefficients: K, N
        rhs_q = -np.vstack([(alpha * A.T @ P @ B).T,
                            (alpha * A.T @ P @ E).T])
        sol_q = np.linalg.solve(Hb, rhs_q)
        K = sol_q[:nu, :]
        N = sol_q[nu:, :]
        # affine (constant) coefficients: p, z_bar
        Xi_u = alpha * B.T @ chi
        Xi_w = alpha * E.T @ chi + lam * v_bar
        rhs_l = -np.hstack([Xi_u, Xi_w])
        sol_l = np.linalg.solve(Hb, rhs_l)
        p = sol_l[:nu]
        z_bar = sol_l[nu:]
        # closed loop
        G = A + B @ K + E @ N
        F = B @ p + E @ z_bar
        # policy evaluation: P = Phi_P + alpha G' P G  (iterated Lyapunov)
        P_new = Q + K.T @ R @ K - lam * (N.T @ N) + alpha * G.T @ P @ G
        rhs_c = (K.T @ R @ p
                 - lam * N.T @ (z_bar - v_bar)
                 + alpha * G.T @ P_new @ F)
        chi_new = np.linalg.solve(np.eye(nx) - alpha * G.T, rhs_c)
        err = np.linalg.norm(P_new - P) + np.linalg.norm(chi_new - chi)
        if track_history:
            history.append(err)
        if err < tol:
            P, chi = P_new, chi_new
            converged = True
            break
        P, chi = P_new, chi_new

    # per-support affine offsets  z^j = v_j - v_bar + z_bar
    zj = V - v_bar + z_bar[None, :]

    # Constant term psi (reported only; it does not feed back into K, p).
    # The atomwise transport penalty is ||m-v_bar||^2; no support-spread
    # constant is added (see the derivation in the module docstring).
    psi = ((p @ R @ p
            - lam * ((z_bar - v_bar) @ (z_bar - v_bar))
            + alpha * (F @ P @ F) + 2 * alpha * (F @ chi)) / (1 - alpha))

    lam_min_P = float(np.min(np.linalg.eigvalsh(P)))
    rho_G = float(np.max(np.abs(np.linalg.eigvals(G))))
    H_uu = R + alpha * B.T @ P @ B
    H_mm = -lam * np.eye(nd) + alpha * E.T @ P @ E
    H_uu_min = float(np.min(np.linalg.eigvalsh(H_uu)))
    H_ww_min = float(np.min(np.linalg.eigvalsh(H_mm)))
    # critical lambda for H_mm ≺ 0:  lam > alpha * lambda_max(E'PE)
    lambda_crit = float(alpha * np.max(np.linalg.eigvalsh(E.T @ P @ E)))

    return dict(P=P, chi=chi, psi=psi, K=K, p=p, N=N, z=z_bar, zj=zj,
                G=G, F=F, v_bar=v_bar, iters=it, converged=converged,
                rho_G=rho_G, lam_min_P=lam_min_P, H_uu_min_eig=H_uu_min,
                H_ww_min_eig=H_ww_min, lambda_crit=lambda_crit,
                history=history)


def solve_drc_pi(A, B, E, Q, R, alpha, lam, d_bar, **kwargs):
    """g=1 specialization (backward compatible with the legacy solver)."""
    nd = E.shape[1]
    V = np.asarray(d_bar, dtype=float).reshape(1, nd)
    return solve_drc_pi_g(A, B, E, Q, R, alpha, lam, V, **kwargs)


# --------------------------------------------------------------------------- #
#  Discounted pure-feedback LQR baseline
# --------------------------------------------------------------------------- #
def solve_discounted_lqr(A, B, Q, R, alpha, max_iter=20000, tol=1e-13):
    """Standard discounted pure-feedback LQR.

    Returns K such that u = -K (x - x_ref), p = 0, plus closed-loop spectral
    radius and iteration count.
    """
    nx = A.shape[0]
    P = np.eye(nx)
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        K = np.linalg.solve(R + alpha * B.T @ P @ B, alpha * B.T @ P @ A)
        Acl = A - B @ K
        P_new = Q + K.T @ R @ K + alpha * Acl.T @ P @ Acl
        if np.linalg.norm(P_new - P) < tol:
            P = P_new
            converged = True
            break
        P = P_new
    rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    return K, rho, it, converged


# --------------------------------------------------------------------------- #
#  Discretization
# --------------------------------------------------------------------------- #
def euler_discretize(Ac, Bc, Ec, Ts):
    """Forward-Euler discretization (the method used throughout the paper)."""
    A = np.eye(Ac.shape[0]) + Ts * Ac
    B = Ts * Bc
    E = Ts * Ec
    return A, B, E


# --------------------------------------------------------------------------- #
#  Static output-window reconstruction
# --------------------------------------------------------------------------- #
def output_window_maps(A, B, C, E, ell):
    """Build the stacked observability and Toeplitz maps for an output window.

    For ``y_[0,ell-1]`` generated from a delayed state x_0,

        y_[0,ell-1] = O_ell x_0 + D_ell u_[0,ell-1]
                      + El_ell d_[0,ell-1].

    The returned maps are used only for the algebraic reconstruction check; the
    released numerical benchmark itself remains model-based full-state feedback.
    """
    ny, nx = C.shape
    nu = B.shape[1]
    nd = E.shape[1]
    if ell < 1:
        raise ValueError("ell must be positive")

    O = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(ell)])
    D = np.zeros((ell * ny, ell * nu))
    El = np.zeros((ell * ny, ell * nd))
    for i in range(ell):
        for m in range(i):
            D[i * ny:(i + 1) * ny, m * nu:(m + 1) * nu] = (
                C @ np.linalg.matrix_power(A, i - 1 - m) @ B)
            El[i * ny:(i + 1) * ny, m * nd:(m + 1) * nd] = (
                C @ np.linalg.matrix_power(A, i - 1 - m) @ E)
    return O, D, El


def reconstruct_delayed_state(A, B, C, E, y_window, u_window, d_window,
                              ell):
    """Recover the delayed state when the disturbance history is known.

    This is the noiseless algebraic identity underlying the static reconstructor.
    It deliberately requires ``d_window``: omitting the disturbance contribution
    is not an identity and is tested as a negative check in ``theory_tests.py``.
    """
    O, D, El = output_window_maps(A, B, C, E, ell)
    y_window = np.asarray(y_window, dtype=float).reshape(-1)
    u_window = np.asarray(u_window, dtype=float).reshape(-1)
    d_window = np.asarray(d_window, dtype=float).reshape(-1)
    rhs = y_window - D @ u_window - El @ d_window
    return np.linalg.pinv(O) @ rhs


def propagate_reconstructed_state(A, B, E, delayed_state, u_window, d_window):
    """Propagate a delayed state through the complete measured input history.

    The disturbance propagation term is essential:

        x_k = A^ell x_{k-ell}
              + sum_{i=0}^{ell-1} A^i B u_{k-1-i}
              + sum_{i=0}^{ell-1} A^i E d_{k-1-i}.

    ``u_window`` and ``d_window`` are ordered from oldest to newest.
    """
    ell = len(u_window)
    if len(d_window) != ell:
        raise ValueError("u_window and d_window must have the same horizon")
    xhat = np.linalg.matrix_power(A, ell) @ np.asarray(delayed_state,
                                                       dtype=float)
    for i in range(ell):
        xhat += np.linalg.matrix_power(A, i) @ B @ u_window[ell - 1 - i]
        xhat += np.linalg.matrix_power(A, i) @ E @ d_window[ell - 1 - i]
    return xhat


# --------------------------------------------------------------------------- #
#  Monte-Carlo rollout
# --------------------------------------------------------------------------- #
def rollout(A, B, E, K, p, x_ref, d_mean, d_std, T, ss, seed, Q, R,
            mae_mode="full", N=None, z=None, disturb="nominal", x0=None):
    """Simulate one closed-loop run.

    Parameters
    ----------
    K, p : affine control  u = K (x - x_ref) + p.
    d_mean, d_std : nominal Gaussian disturbance mean / component-wise std.
    T : total steps; ss : steady-state start step (inclusive).
    Q, R : explicit stage-cost weights (replaces the legacy global Q_loc/R_loc).
    mae_mode : 'full'  -> mean ||x-x_ref|| over steady window
                'angle'-> mean ||x[[0,2,4]]-x_ref[[0,2,4]]||
                'freq' -> mean ||x[[0,2]]-x_ref[[0,2]]||
    disturb : 'nominal'     -> Gaussian disturbance d = d_mean + d_std*eps
              'adversarial' -> deterministic affine d = N x + z supplied by
                               the caller (used for the common-dual sensitivity
                               test, not a hard-ball worst-case certificate).

    Returns
    -------
    (mae, mean_stage_cost) : steady-window averages (cost NOT divided by
    (1-alpha) here; the discount division is applied at batch level).
    """
    rng = np.random.default_rng(seed)
    nx = A.shape[0]
    x = x_ref.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    mae_list, cost_list = [], []
    for t in range(T):
        # The stage cost at time t uses the current state x_t and input u_t.
        # Compute it before updating the state to x_{t+1}; otherwise Q is
        # incorrectly paired with u_t from the previous step.
        e = x - x_ref
        u = K @ e + p
        if disturb == "adversarial":
            if N is None or z is None:
                raise ValueError("adversarial rollout needs N and z")
            d = N @ x + z
        else:
            d = d_mean + d_std * rng.standard_normal(len(d_mean))
        if t >= ss:
            if mae_mode == "full":
                mae_list.append(float(np.linalg.norm(e)))
            elif mae_mode == "angle":
                mae_list.append(float(np.linalg.norm(e[[0, 2, 4]])))
            elif mae_mode == "freq":
                mae_list.append(float(np.linalg.norm(e[[0, 2]])))
            else:
                raise ValueError(mae_mode)
            cost_list.append(float(e @ Q @ e + u @ R @ u))
        x = A @ x + B @ u + E @ d
    return float(np.mean(mae_list)), float(np.mean(cost_list))


# --------------------------------------------------------------------------- #
#  Reusable policy evaluation / improvement primitives (used by SDPI, Alg.4
#  and the theory tests). Equations follow the FOC block system documented at
#  the top of this file; the chi convention here matches the solver above
#  (V = x'Px + 2 chi'x + psi), i.e. the paper's chi is rescaled by 2.
# --------------------------------------------------------------------------- #
def policy_improve(A, B, E, Q, R, alpha, lam, V, P, chi):
    """One unconstrained saddle-point improvement given value params (P, chi).

    Returns dict with K, p, N, z (z_bar), zj (per-support offsets), G, F.
    The matrix blocks K, N do not depend on the support index; only the affine
    offset z^j differs atom-by-atom (paper L839: Xi_omega^j = alpha E'chi +
    lambda v_j).
    """
    nx = A.shape[0]
    nu = B.shape[1]
    nd = E.shape[1]
    V = np.atleast_2d(np.asarray(V, dtype=float))
    v_bar = V.mean(axis=0)

    H_uu = R + alpha * B.T @ P @ B
    H_um = alpha * B.T @ P @ E
    H_mm = -lam * np.eye(nd) + alpha * E.T @ P @ E
    Hb = np.block([[H_uu, H_um],
                   [H_um.T, H_mm]])
    rhs_q = -np.vstack([(alpha * A.T @ P @ B).T,
                        (alpha * A.T @ P @ E).T])
    sol_q = np.linalg.solve(Hb, rhs_q)
    K = sol_q[:nu, :]
    N = sol_q[nu:, :]

    Xi_u = alpha * B.T @ chi
    Xi_w = alpha * E.T @ chi + lam * v_bar
    rhs_l = -np.hstack([Xi_u, Xi_w])
    sol_l = np.linalg.solve(Hb, rhs_l)
    p = sol_l[:nu]
    z_bar = sol_l[nu:]

    zj = V - v_bar[None, :] + z_bar[None, :]
    G = A + B @ K + E @ N
    F = B @ p + E @ z_bar
    return dict(K=K, p=p, N=N, z=z_bar, zj=zj, G=G, F=F, v_bar=v_bar)


def policy_evaluate(A, B, E, Q, R, alpha, lam, V, K, p, N, z_bar):
    """Exact policy evaluation under a stabilizing policy.

    Solves P - alpha G'PG = Phi_P by the discrete Lyapunov equation
    (A_l = sqrt(alpha) G), the chi linear equation, and the constant psi.
    Requires rho(G) < 1/sqrt(alpha) for the Lyapunov solve to be well posed.
    """
    from scipy.linalg import solve_discrete_lyapunov
    nx = A.shape[0]
    V = np.atleast_2d(np.asarray(V, dtype=float))
    v_bar = V.mean(axis=0)

    G = A + B @ K + E @ N
    F = B @ p + E @ z_bar
    PhiP = Q + K.T @ R @ K - lam * (N.T @ N)
    # scipy solves P - A_l P A_l' = PhiP; to obtain P - alpha G' P G = PhiP,
    # pass A_l = sqrt(alpha) G' (so A_l P A_l' = alpha G' P G).
    A_l = np.sqrt(alpha) * G.T
    P = solve_discrete_lyapunov(A_l, PhiP)
    rhs_c = (K.T @ R @ p
             - lam * N.T @ (z_bar - v_bar)
             + alpha * G.T @ P @ F)
    chi = np.linalg.solve(np.eye(nx) - alpha * G.T, rhs_c)
    psi = ((p @ R @ p
            - lam * ((z_bar - v_bar) @ (z_bar - v_bar))
            + alpha * (F @ P @ F) + 2 * alpha * (F @ chi)) / (1 - alpha))
    return dict(P=P, chi=chi, psi=psi, G=G, F=F, v_bar=v_bar)


def per_support_improvement(A, B, E, Q, R, alpha, lam, V, P, chi):
    """Explicit atom-by-atom dual decomposition (T1, tex L313-356).

    The next-state cost depends on the dynamics only through the mean
    omega_bar = (1/g) sum_j omega^j (eq. omega_avg); the penalty is separable
    atom-by-atom. For a fixed mean m, minimizing the separable penalty under
    the mean constraint (1/g) sum_j omega^j = m gives

        omega^j = v_j + (m - v_bar),

    so the quadratic gain N is shared across all atoms, and the affine offsets
    satisfy  z^j = v_j - v_bar + z_bar  exactly.

    This routine solves the shared saddle block for (K, N, p, z_bar) and then
    returns the per-atom N_j (copies of the shared N) and z^j, so the two
    identities can be checked directly:
        N_j = N  for every j,
        z^j - z_bar = v_j - v_bar.
    """
    nu = B.shape[1]
    nd = E.shape[1]
    V = np.atleast_2d(np.asarray(V, dtype=float))
    g = V.shape[0]
    v_bar = V.mean(axis=0)

    H_uu = R + alpha * B.T @ P @ B
    H_um = alpha * B.T @ P @ E
    H_mm = -lam * np.eye(nd) + alpha * E.T @ P @ E
    Hb = np.block([[H_uu, H_um],
                   [H_um.T, H_mm]])
    rhs_q = -np.vstack([(alpha * A.T @ P @ B).T,
                        (alpha * A.T @ P @ E).T])
    sol_q = np.linalg.solve(Hb, rhs_q)
    K = sol_q[:nu, :]
    N = sol_q[nu:, :]

    Xi_u = alpha * B.T @ chi
    Xi_w = alpha * E.T @ chi + lam * v_bar
    rhs_l = -np.hstack([Xi_u, Xi_w])
    sol_l = np.linalg.solve(Hb, rhs_l)
    p = sol_l[:nu]
    z_bar = sol_l[nu:]

    # mean-constrained atom offsets: z^j = v_j - v_bar + z_bar
    zj = V - v_bar[None, :] + z_bar[None, :]
    N_list = [N.copy() for _ in range(g)]
    return dict(K=K, p=p, N=N, N_list=N_list, z=z_bar, zj=zj,
                v_bar=v_bar)


def mc_batch(A, B, E, K, p, x_ref, d_mean, d_std, T, ss, alpha, mode,
             Q, R, seeds=range(1000, 1020), N=None, z=None,
             disturb="nominal", x0=None):
    """20-seed (default) Monte-Carlo batch.

    Returns (mae_mean, mae_std, cost_mean, cost_std). Discounted cost is
    E[x'Qx+u'Ru]/(1-alpha), matching the paper definition.
    """
    maes, costs = [], []
    for seed in seeds:
        m, c = rollout(A, B, E, K, p, x_ref, d_mean, d_std, T, ss, seed,
                       Q, R, mae_mode=mode, N=N, z=z, disturb=disturb,
                       x0=x0)
        maes.append(m)
        costs.append(c / (1 - alpha))
    return (float(np.mean(maes)), float(np.std(maes)),
            float(np.mean(costs)), float(np.std(costs)))


# --------------------------------------------------------------------------- #
#  Model-based output-feedback closed loop
# --------------------------------------------------------------------------- #
def rollout_output_feedback(A, B, C, E, K, p, x_ref, d_mean, d_std,
                            T, ss, seed, Q, R, ell=3, mae_mode="full",
                            d_history="known", measurement_noise_std=0.0,
                            N=None, z=None, disturb="nominal"):
    """Closed-loop simulation using the output-window reconstructor.

    At time ``t >= ell`` the estimator uses only the last ``ell`` measured
    outputs, inputs, and the specified disturbance history.  The delayed state
    is recovered from the full output-window equation, then propagated through
    the complete input/disturbance history.  The controller applied to the
    plant is ``u_t = K (xhat_t - x_ref) + p``.

    Parameters
    ----------
    C : ndarray
        Output matrix.  ``O_ell`` must have full column rank.
    d_history : {"known", "mean"}
        ``known`` uses the realized disturbance history in the reconstruction
        and propagation.  This is the model-based setting in which disturbance
        samples are available (the paper's explicit reconstruction assumption).
        ``mean`` replaces the unknown realized history by ``d_mean`` and is a
        sensitivity check, not an exact reconstruction.
    measurement_noise_std : float
        Standard deviation of i.i.d. additive output noise.  The default zero
        recovers the noiseless algebraic identity in the paper.

    Returns
    -------
    dict
        Steady-window ``mae`` and ``stage_cost``, reconstruction diagnostics,
        the observability rank, and the number of warm-up / reconstructed steps.
        The stage cost is ``(x_t-x_ref)'Q(x_t-x_ref)+u_t'Ru_t`` with the same
        time index as the control input; discounting is applied by
        :func:`mc_batch_output_feedback`.
    """
    if d_history not in ("known", "mean"):
        raise ValueError("d_history must be 'known' or 'mean'")
    if ell < 1:
        raise ValueError("ell must be positive")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    nx = A.shape[0]
    nd = E.shape[1]
    ny = C.shape[0]
    O, D_ell, E_ell = output_window_maps(A, B, C, E, ell)
    O_pinv = np.linalg.pinv(O)
    rank_O = int(np.linalg.matrix_rank(O))
    if rank_O < nx:
        raise ValueError(f"O_ell is rank deficient: rank={rank_O}, nx={nx}")

    rng = np.random.default_rng(seed)
    x = x_ref.copy()
    y_hist, u_hist, d_hist = [], [], []
    mae_list, cost_list = [], []
    recon_errors = []
    warmup_steps = 0

    for t in range(T):
        # At t < ell there is no complete output window.  We use the known
        # initial state only during this finite warm-up, then switch permanently
        # to the output-window reconstructor for t >= ell.
        if t < ell:
            xhat = x.copy()
            warmup_steps += 1
        else:
            y_window = np.asarray(y_hist[t - ell:t], dtype=float).reshape(-1)
            u_window = np.asarray(u_hist[t - ell:t], dtype=float)
            if d_history == "known":
                d_window = np.asarray(d_hist[t - ell:t], dtype=float)
            else:
                d_window = np.tile(np.asarray(d_mean, dtype=float), (ell, 1))

            rhs = (y_window - D_ell @ u_window.reshape(-1)
                   - E_ell @ d_window.reshape(-1))
            x_delayed = O_pinv @ rhs
            xhat = propagate_reconstructed_state(
                A, B, E, x_delayed, u_window, d_window)

        e = x - x_ref
        u = K @ (xhat - x_ref) + p
        if disturb == "adversarial":
            if N is None or z is None:
                raise ValueError("adversarial rollout needs N and z")
            d = N @ x + z
        elif disturb == "nominal":
            d = d_mean + d_std * rng.standard_normal(nd)
        else:
            raise ValueError(disturb)

        if t >= ss:
            if mae_mode == "full":
                mae_list.append(float(np.linalg.norm(e)))
            elif mae_mode == "angle":
                mae_list.append(float(np.linalg.norm(e[[0, 2, 4]])))
            elif mae_mode == "freq":
                mae_list.append(float(np.linalg.norm(e[[0, 2]])))
            else:
                raise ValueError(mae_mode)
            cost_list.append(float(e @ Q @ e + u @ R @ u))
            recon_errors.append(float(np.linalg.norm(xhat - x)))

        # The output at t is generated from x_t and is appended for future
        # windows.  Output noise, when requested, affects the estimator only.
        y = C @ x
        if measurement_noise_std > 0.0:
            y = y + measurement_noise_std * rng.standard_normal(ny)
        y_hist.append(y)
        u_hist.append(u.copy())
        d_hist.append(np.asarray(d, dtype=float).copy())
        x = A @ x + B @ u + E @ d

    if not mae_list:
        raise ValueError("steady-state window is empty")
    return dict(
        mae=float(np.mean(mae_list)),
        stage_cost=float(np.mean(cost_list)),
        recon_rmse=float(np.sqrt(np.mean(np.square(recon_errors)))),
        recon_max=float(np.max(recon_errors)),
        rank_O=rank_O,
        ell=int(ell),
        warmup_steps=int(warmup_steps),
        reconstructed_steps=len(recon_errors),
        d_history=d_history,
    )


def mc_batch_output_feedback(A, B, C, E, K, p, x_ref, d_mean, d_std,
                             T, ss, alpha, mode, Q, R,
                             seeds=range(1000, 1020), ell=3,
                             d_history="known", measurement_noise_std=0.0,
                             N=None, z=None, disturb="nominal"):
    """Monte-Carlo batch for the model-based output-feedback loop."""
    runs = [
        rollout_output_feedback(
            A, B, C, E, K, p, x_ref, d_mean, d_std, T, ss, seed, Q, R,
            ell=ell, mae_mode=mode, d_history=d_history,
            measurement_noise_std=measurement_noise_std,
            N=N, z=z, disturb=disturb)
        for seed in seeds
    ]
    maes = np.asarray([r["mae"] for r in runs])
    costs = np.asarray([r["stage_cost"] / (1.0 - alpha) for r in runs])
    recons = np.asarray([r["recon_rmse"] for r in runs])
    return dict(
        mae_mean=float(maes.mean()), mae_std=float(maes.std()),
        cost_mean=float(costs.mean()), cost_std=float(costs.std()),
        recon_rmse_mean=float(recons.mean()), recon_rmse_std=float(recons.std()),
        recon_max=float(max(r["recon_max"] for r in runs)),
        rank_O=int(runs[0]["rank_O"]), ell=int(ell),
        warmup_steps=int(runs[0]["warmup_steps"]),
        reconstructed_steps=int(runs[0]["reconstructed_steps"]),
        d_history=d_history,
    )


# --------------------------------------------------------------------------- #
#  Unknown-disturbance-history output feedback via a steady-state KF observer
# --------------------------------------------------------------------------- #
def kalman_observer_gain(A, C, process_cov, measurement_cov):
    """Steady-state Kalman gain for the observable pair ``(A,C)``.

    The observer is used when the realized disturbance history is unavailable.
    It reconstructs the conditional mean of the state from the current output,
    the previous input, and the nominal disturbance mean.  This is a dynamic
    output-feedback estimator, not the exact static window map (which requires
    the realized disturbance history).
    """
    from scipy.linalg import solve_discrete_are
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    Qw = np.asarray(process_cov, dtype=float)
    Rv = np.asarray(measurement_cov, dtype=float)
    P = solve_discrete_are(A.T, C.T, Qw, Rv)
    S = C @ P @ C.T + Rv
    L = P @ C.T @ np.linalg.inv(S)
    Aobs = A - L @ C
    if float(np.max(np.abs(np.linalg.eigvals(Aobs)))) >= 1.0:
        raise ValueError("Kalman observer is not stable")
    return L, P, Aobs


def rollout_output_feedback_kf(A, B, C, E, K, p, x_ref, d_mean, d_std,
                               T, ss, seed, Q, R, mae_mode="full",
                               x0=None, process_floor=1e-10,
                               measurement_noise_std=1e-8):
    """Output-feedback rollout with no realized disturbance history.

    The estimator is a steady-state Kalman filter driven by ``y=Cx`` and the
    nominal disturbance mean.  The realized disturbance is still injected into
    the plant, but its sample path is not passed to the estimator.  This is the
    unknown-history counterpart of the exact output-window reconstructor.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    d_mean = np.asarray(d_mean, dtype=float).reshape(-1)
    d_std = np.asarray(d_std, dtype=float).reshape(-1)
    nx, ny = A.shape[0], C.shape[0]

    Qw = E @ np.diag(np.square(d_std)) @ E.T + process_floor * np.eye(nx)
    Rv = max(float(measurement_noise_std), 1e-12) ** 2 * np.eye(ny)
    L, _, Aobs = kalman_observer_gain(A, C, Qw, Rv)

    rng = np.random.default_rng(seed)
    x = x_ref.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    xhat = x_ref.copy()
    mae_list, cost_list, recon_errors = [], [], []
    max_state = 0.0
    diverged = None
    for t in range(T):
        e = x - x_ref
        u = K @ (xhat - x_ref) + p
        d = d_mean + d_std * rng.standard_normal(d_mean.size)
        y = C @ x + measurement_noise_std * rng.standard_normal(ny)
        max_state = max(max_state, float(np.max(np.abs(x))))
        if not np.all(np.isfinite(x)) or max_state > 1e8:
            diverged = t
            break
        if t >= ss:
            if mae_mode == "full":
                mae_list.append(float(np.linalg.norm(e)))
            elif mae_mode == "angle":
                mae_list.append(float(np.linalg.norm(e[[0, 2, 4]])))
            elif mae_mode == "freq":
                mae_list.append(float(np.linalg.norm(e[[0, 2]])))
            else:
                raise ValueError(mae_mode)
            cost_list.append(float(e @ Q @ e + u @ R @ u))
            recon_errors.append(float(np.linalg.norm(xhat - x)))
        x_next = A @ x + B @ u + E @ d
        xhat = A @ xhat + B @ u + E @ d_mean + L @ (y - C @ xhat)
        x = x_next

    if not mae_list:
        raise ValueError("steady-state window is empty")
    return dict(
        mae=float(np.mean(mae_list)),
        stage_cost=float(np.mean(cost_list)),
        recon_rmse=float(np.sqrt(np.mean(np.square(recon_errors)))),
        recon_max=float(np.max(recon_errors)),
        observer_rho=float(np.max(np.abs(np.linalg.eigvals(Aobs)))),
        observer_gain=L,
        max_state=max_state,
        diverged=diverged,
    )


def mc_batch_output_feedback_kf(A, B, C, E, K, p, x_ref, d_mean, d_std,
                                T, ss, alpha, mode, Q, R,
                                seeds=range(1000, 1020), x0=None,
                                process_floor=1e-10,
                                measurement_noise_std=1e-8):
    """20-seed Monte-Carlo batch for unknown-history output feedback."""
    runs = [rollout_output_feedback_kf(
        A, B, C, E, K, p, x_ref, d_mean, d_std, T, ss, seed, Q, R,
        mae_mode=mode, x0=x0, process_floor=process_floor,
        measurement_noise_std=measurement_noise_std) for seed in seeds]
    maes = np.asarray([r["mae"] for r in runs])
    costs = np.asarray([r["stage_cost"] / (1.0 - alpha) for r in runs])
    recons = np.asarray([r["recon_rmse"] for r in runs])
    return dict(
        mae_mean=float(maes.mean()), mae_std=float(maes.std()),
        cost_mean=float(costs.mean()), cost_std=float(costs.std()),
        recon_rmse_mean=float(recons.mean()), recon_rmse_std=float(recons.std()),
        recon_max=float(max(r["recon_max"] for r in runs)),
        observer_rho=float(runs[0]["observer_rho"]),
        max_state=float(max(r["max_state"] for r in runs)),
        n_diverged=int(sum(r["diverged"] is not None for r in runs)),
        estimator="steady_state_kalman_filter_unknown_disturbance_history",
    )
