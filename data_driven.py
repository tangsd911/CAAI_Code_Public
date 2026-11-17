#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identification-based data-driven SDPI building blocks (Alg. 4, tex L1263-1528).

Closed-loop data under a fixed policy (K^(i), p^(i), N^(i), z_bar^(i))::

    x_{k+1} = G^(i) x_k + B e_k + E xi_k,
    e_k     = u_k - K^(i) x_k - p^(i),
    xi_k    = omega_k - N^(i) x_k - z_bar^(i).

Identification regressors (T3 / T5, model-free control loop):

  * Closed-loop identification (eq. L1320-1330): regress X+ on [X; E_e]. The
    disturbance term E xi is treated as noise; the true disturbance D is NOT a
    regressor.
  * Open-loop identification (T5 real [X; U] route): regress X+ on [X; U],
    again treating E xi as noise.
  * Affine offset: a constant column 1 is added to recover F = B p + E z_bar.

Truth-usage contract (T5c): the true A, B, E are referenced ONLY inside
``collect_closed_loop`` (data generation) and inside the error-metric functions
(comparison against the solved model-based optimum). The identification,
Lyapunov evaluation and policy update below never read A, B, E.

Identifiability note: if the adversary law is d=Nx+z+xi and no exogenous
disturbance-channel experiment is available, the current disturbance is
feedback-correlated with the state and xi enters the transition error; E is
then confounded with A+EN even when the regressor matrix has full rank.  The
code therefore exposes an explicit identifiability certificate.  Full E/N
experiments use an independent disturbance probe q, with d=Nx+z+q+xi and q as
an excluded 2SLS instrument.  Without q the code returns a negative result and
does not feed a confounded estimate into SDPI.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2


# --------------------------------------------------------------------------- #
#  Data generation — the ONLY place the true A, B, E are used to simulate.
# --------------------------------------------------------------------------- #
def collect_closed_loop(spec, K, p, N, zbar, n_steps, seed,
                        expl_scale=0.1, warmup=100, dist_mode="adversarial",
                        state_obs_noise_std=0.0, input_obs_noise_std=0.0,
                        dist_obs_noise_std=0.0, target_obs_noise_std=0.0,
                        dist_probe_std=0.0):
    """Simulate the closed loop under policy (K,p,N,zbar) with exploration e.

    The true A, B, E, d_bar, d_std enter here only to generate trajectories.

    dist_mode:
      'adversarial' (Alg.4 closed-loop model, L1278-1282): the realized
          disturbance follows the affine adversary policy d = N x + zbar + xi,
          xi zero-mean independent noise; then x+ = G x + B e + E xi and the
          residual xi is uncorrelated with the regressors [x; e].
      'nominal': physical nominal Gaussian d = d_bar + d_std eps (kept only
          for the bias discussion; the realized closed loop is A+BK, so G is
          not identified under this mode).
    Returns stacked arrays (columns = samples).
    """
    rng = np.random.default_rng(seed)
    A, B, E = spec["A"], spec["B"], spec["E"]
    nx, nu, nd = A.shape[0], B.shape[1], E.shape[1]
    d_bar, d_std = spec["d_bar"], spec["d_std"]

    x = np.zeros(nx)
    X, U, Xp, Ee, Xi, Dtrue, Qprobe = [], [], [], [], [], [], []
    for k in range(n_steps + warmup):
        e = expl_scale * rng.standard_normal(nu)
        u = K @ x + p + e
        # Exogenous disturbance-channel excitation.  It is intentionally
        # independent of the state and of the adversary innovation xi.  With
        # q_k=0 the current disturbance is an affine feedback signal and E is
        # not separately identifiable from closed-loop data (see
        # ``identifiability_certificate``).
        q = (np.asarray(dist_probe_std, dtype=float)
             * rng.standard_normal(nd))
        if dist_mode == "adversarial":
            xi = d_std * rng.standard_normal(nd)
            d = N @ x + zbar + q + xi
        else:
            d = d_bar + q + d_std * rng.standard_normal(nd)
            xi = d - N @ x - zbar - q
        xp = A @ x + B @ u + E @ d
        if k >= warmup:
            # Measurements can be corrupted independently of the plant.  The
            # simulation itself always uses the true state and input.
            X.append(x + state_obs_noise_std * rng.standard_normal(nx))
            U.append(u + input_obs_noise_std * rng.standard_normal(nu))
            Xp.append(xp + target_obs_noise_std * rng.standard_normal(nx))
            Ee.append(e.copy())
            Xi.append(xi.copy())
            Dtrue.append(d + dist_obs_noise_std * rng.standard_normal(nd))
            Qprobe.append(q.copy())
        x = xp
    return dict(
        X=np.array(X).T, U=np.array(U).T, Xp=np.array(Xp).T,
        Ee=np.array(Ee).T, Xi=np.array(Xi).T, D=np.array(Dtrue).T,
        Q=np.array(Qprobe).T,
        nx=nx, nu=nu, nd=nd)


def collect_support_points(spec, n_steps, seed=0):
    """Collect an independent pilot sample used to estimate the support ``V``.

    The pilot is deliberately separate from the policy-iteration batches: it
    contains only physical disturbance observations around the ambient
    operating point.  It is used to estimate the ambiguity-set centre (and,
    for ``g>1``, the empirical support).  No controller or policy gains are
    read by this estimator.  The true generator is used only to create these
    data, as it must be in a simulation study.
    """
    rng = np.random.default_rng(seed)
    d_bar = np.asarray(spec["d_bar"], dtype=float)
    d_std = np.asarray(spec["d_std"], dtype=float)
    return d_bar[:, None] + d_std[:, None] * rng.standard_normal(
        (d_bar.size, int(n_steps)))


def estimate_support(samples, g=1, center=True):
    """Estimate a one-point or empirical disturbance support from pilot data.

    ``g=1`` uses the sample mean.  For ``g>1``, a deterministic quantile-based
    support is returned; this is a sampling rule, not a claim that the true
    distribution is known.  The returned matrix has shape ``(g, n_d)``.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.shape[0] == 1 and samples.shape[1] != 1:
        # Callers normally pass observations as columns.  Tolerate the
        # transposed form for the one-dimensional case.
        pass
    n_obs, nd = samples.shape
    g = int(g)
    if g < 1:
        raise ValueError("g must be a positive integer")
    if g == 1:
        return samples.mean(axis=0, keepdims=True)
    qs = (np.arange(g, dtype=float) + 0.5) / g
    V = np.empty((g, nd), dtype=float)
    for j in range(nd):
        V[:, j] = np.quantile(samples[:, j], qs)
    if center:
        # Keep the empirical mean exactly equal to the pilot mean.  Quantiles
        # can otherwise shift a finite support by O(N^-1/2), which would be
        # confounded with the identification error in the experiments.
        V += samples.mean(axis=0, keepdims=True) - V.mean(axis=0, keepdims=True)
    return V


# --------------------------------------------------------------------------- #
#  Least-squares identification routes (no true D in any regressor)
# --------------------------------------------------------------------------- #
def _lstsq(reg, target):
    """target = Theta @ reg, least squares; returns Theta."""
    Theta, *_ = np.linalg.lstsq(reg.T, target.T, rcond=None)
    return Theta.T


def identify_closed_loop(data):
    """Regress Xp on [X; E_e] -> [G_hat, B_hat] (eq. L1328).

    Also identifies the affine offset F_hat by adding a constant column.
    Returns estimates plus rank / lambda-PE diagnostics.
    """
    X, Xp, Ee = data["X"], data["Xp"], data["Ee"]
    nx, nu, N = data["nx"], data["nu"], X.shape[1]
    reg = np.vstack([X, Ee])
    Theta = _lstsq(reg, Xp)
    G_hat = Theta[:, :nx]
    B_hat = Theta[:, nx:nx + nu]
    # affine offset: regressors [X; Ee; 1]
    reg_a = np.vstack([X, Ee, np.ones((1, N))])
    Theta_a = _lstsq(reg_a, Xp)
    F_hat = Theta_a[:, -1]
    # PE diagnostics (paper L1337-1339)
    rank = int(np.linalg.matrix_rank(reg.T))
    Phi = reg_a @ reg_a.T / N
    lam_min = float(np.min(np.linalg.eigvalsh(Phi)))
    return dict(G_hat=G_hat, B_hat=B_hat, F_hat=F_hat,
                rank=int(rank), reg_dim=nx + nu, lam_min_Phi=lam_min)


def identify_open_loop(data):
    """Regress Xp on [X; U] -> [A_hat, B_hat] (T5 real [X; U] route; E xi noise)."""
    X, Xp, U = data["X"], data["Xp"], data["U"]
    nx, nu, N = data["nx"], data["nu"], X.shape[1]
    reg = np.vstack([X, U])
    Theta = _lstsq(reg, Xp)
    A_hat = Theta[:, :nx]
    B_hat = Theta[:, nx:nx + nu]
    reg_a = np.vstack([X, U, np.ones((1, N))])
    Theta_a = _lstsq(reg_a, Xp)
    F_hat = Theta_a[:, -1]
    rank = int(np.linalg.matrix_rank(reg.T))
    Phi = reg_a @ reg_a.T / N
    lam_min = float(np.min(np.linalg.eigvalsh(Phi)))
    return dict(A_hat=A_hat, B_hat=B_hat, F_hat=F_hat,
                rank=rank, reg_dim=nx + nu, lam_min_Phi=lam_min)


# --------------------------------------------------------------------------- #
#  Identified-model policy evaluation and improvement (data-driven loop)
# --------------------------------------------------------------------------- #
def lyapunov_policy_eval(G_hat, PhiP, alpha):
    """Solve P_hat - alpha G_hat' P_hat G_hat = PhiP (identified closed loop)."""
    A_l = np.sqrt(alpha) * G_hat.T  # scipy: P - A_l P A_l' = PhiP
    return solve_discrete_lyapunov(A_l, PhiP)




# --------------------------------------------------------------------------- #
#  Error metrics — the ONLY place truth matrices are compared against.
# --------------------------------------------------------------------------- #
def identification_errors(spec, opt, cl, ol, P_hat, K_hat, p_hat):
    """Relative / absolute identification and policy errors vs the model optimum.

    opt: baseline model-based solution dict. Truth A, B, E, P, K, p enter here
    only through the already-solved ``opt`` and ``spec`` for scoring.
    """
    A, B, E = spec["A"], spec["B"], spec["E"]
    G_star = opt["G"]
    return dict(
        eG=float(np.linalg.norm(cl["G_hat"] - G_star) / np.linalg.norm(G_star)),
        eB_cl=float(np.linalg.norm(cl["B_hat"] - B) / np.linalg.norm(B)),
        eA=float(np.linalg.norm(ol["A_hat"] - A) / np.linalg.norm(A)),
        eB_ol=float(np.linalg.norm(ol["B_hat"] - B) / np.linalg.norm(B)),
        eP=float(np.linalg.norm(P_hat - opt["P"]) / np.linalg.norm(opt["P"])),
        eK=float(np.linalg.norm(K_hat - opt["K"], "fro")),
        ep=float(np.linalg.norm(p_hat - opt["p"])),
    )


def loglog_slope(Ns, means):
    """Slope of log(mean error) vs log N; expected ~ -1/2 (Thm.6 O(N^-1/2))."""
    Ns = np.asarray(Ns, dtype=float)
    means = np.asarray(means, dtype=float)
    ok = means > 0
    if ok.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(Ns[ok]), np.log(means[ok]), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------- #
#  Full A,B,E identification (state/input/disturbance data) and IV/2SLS
# --------------------------------------------------------------------------- #
def _stack_full(data):
    X, U, D, Xp = data["X"], data["U"], data["D"], data["Xp"]
    return np.vstack([X, U, D, np.ones((1, X.shape[1]))]), Xp


def identify_full_ols(data):
    """Identify ``X+ = [A B E c][X;U;D;1]`` by ordinary least squares.

    The realized disturbance ``D`` is a measured regressor here.  This is the
    state/input/disturbance-data identification route used by the data-driven
    validation experiments; it is distinct from the reduced closed-loop-map
    regression in :func:`identify_closed_loop`.
    """
    X = data["X"]
    nx, nu, nd = data["nx"], data["nu"], data["nd"]
    W, Y = _stack_full(data)
    Theta = _lstsq(W, Y)
    A_hat = Theta[:, :nx]
    B_hat = Theta[:, nx:nx + nu]
    E_hat = Theta[:, nx + nu:nx + nu + nd]
    c_hat = Theta[:, -1]
    Phi = W @ W.T / W.shape[1]
    return dict(A_hat=A_hat, B_hat=B_hat, E_hat=E_hat, c_hat=c_hat,
                rank=int(np.linalg.matrix_rank(W.T)),
                reg_dim=W.shape[0],
                lam_min_Phi=float(np.min(np.linalg.eigvalsh(Phi))),
                cond_Phi=float(np.linalg.cond(Phi)),
                method="OLS", instrument_mode="self",
                rank_instruments=None, instrument_dim=None,
                first_stage_min_F=None, first_stage_min_eig=None,
                probe_present=False)


def _scale_rows(M):
    """Scale rows before least squares and return the inverse scaling."""
    M = np.asarray(M, dtype=float)
    s = np.std(M, axis=1)
    s[~np.isfinite(s) | (s < 1e-12)] = 1.0
    return M / s[:, None], 1.0 / s


def _first_stage_diagnostics(W, Z, reduced_rows, target_rows, n_total):
    """First-stage rank/strength diagnostics for excluded instruments.

    ``reduced_rows`` indexes the instruments retained in the null model.  The
    reported partial F statistics ask whether the excluded instruments add
    explanatory power for each potentially endogenous regressor in
    ``target_rows``.  This is the relevant diagnostic for the affine
    disturbance block, whose current value is correlated with its innovation.
    """
    excluded = np.setdiff1d(np.arange(Z.shape[0]), reduced_rows)
    if excluded.size == 0:
        return dict(first_stage_min_F=None, first_stage_F=None,
                    first_stage_min_eig=None, excluded_dim=0)
    Zf = Z.T
    Zr = Z[reduced_rows].T
    Fs, eigs = [], []
    for j in target_rows:
        y = W[j]
        bf, *_ = np.linalg.lstsq(Zf, y, rcond=None)
        br, *_ = np.linalg.lstsq(Zr, y, rcond=None)
        rf = y - Zf @ bf
        rr = y - Zr @ br
        rss_f = float(rf @ rf)
        rss_r = float(rr @ rr)
        q = excluded.size
        df = max(n_total - Z.shape[0], 1)
        F = ((rss_r - rss_f) / q) / max(rss_f / df, 1e-30)
        Fs.append(float(F))
        eigs.append(float(np.min(np.linalg.eigvalsh(Zf.T @ Zf / n_total))))
    return dict(first_stage_min_F=float(min(Fs)),
                first_stage_F=[float(x) for x in Fs],
                first_stage_min_eig=float(min(eigs)),
                excluded_dim=int(excluded.size))


def identify_full_iv(data, n_lags=1, instrument_mode="probe"):
    """Two-stage least squares for ``x+ = [A B E c][x;u;d;1]``.

    The structural disturbance ``d_t`` is endogenous whenever the adversary
    policy is affine in the state, because ``d_t=N x_t+z+q_t+xi_t`` and the
    innovation ``xi_t`` also appears in the state equation.  Lagging ``d`` is
    generally a weak or invalid repair under that feedback law.

    In ``instrument_mode='probe'`` the designed disturbance-channel probe
    ``q_t`` is used as an excluded instrument.  It is independent of ``xi_t``
    and directly excites the endogenous disturbance regressor.  This is the
    identifiable experimental design.  In ``instrument_mode='lagged'`` only
    lagged observed variables are used; the routine reports the first-stage
    strength and rank, but does not claim identification when they fail.

    All least-squares stages are row-scaled for conditioning, then transformed
    back to the original coefficient coordinates.  No true matrix enters the
    estimator.
    """
    X, U, D, Xp = data["X"], data["U"], data["D"], data["Xp"]
    nx, nu, nd = data["nx"], data["nu"], data["nd"]
    N = X.shape[1]
    if n_lags < 1 or N <= n_lags:
        raise ValueError("not enough samples for the requested instruments")
    W = np.vstack([X, U, D, np.ones((1, N))])
    start = int(n_lags)
    Wc = W[:, start:]
    Yc = Xp[:, start:]
    Q = data.get("Q")
    q_present = (Q is not None and Q.size and
                 float(np.std(Q)) > 1e-14)
    if instrument_mode == "probe" and not q_present:
        raise ValueError("probe instruments requested but no disturbance probe data are available")
    if instrument_mode not in ("probe", "lagged"):
        raise ValueError("instrument_mode must be 'probe' or 'lagged'")

    if instrument_mode == "probe":
        # Lagged x,u avoid using the current measurement error as its own
        # instrument; q instruments the current endogenous disturbance.  One
        # lag of q improves over-identification without using a lagged
        # endogenous disturbance.
        blocks = [X[:, start:], U[:, start:],
                  X[:, start - 1:N - 1], U[:, start - 1:N - 1],
                  Q[:, start:]]
        for lag in range(1, n_lags + 1):
            blocks.append(Q[:, start - lag:N - lag])
        blocks.append(np.ones((1, N - start)))
        Z = np.vstack(blocks)
        # Retain current and lagged x,u plus the intercept; the excluded block
        # is the disturbance probe q and its lag.  This isolates the strength
        # of the actual identifying instrument.
        reduced_rows = np.r_[np.arange(2 * (nx + nu)), Z.shape[0] - 1]
        target_rows = list(range(nx + nu, nx + nu + nd))
    else:
        blocks = []
        for lag in range(1, n_lags + 1):
            blocks.extend([X[:, start - lag:N - lag],
                           U[:, start - lag:N - lag],
                           D[:, start - lag:N - lag]])
        blocks.append(np.ones((1, N - start)))
        Z = np.vstack(blocks)
        reduced_rows = np.array([Z.shape[0] - 1])
        target_rows = list(range(nx + nu, nx + nu + nd))

    # Scaled two-stage least squares.  Scaling is a similarity transform of
    # the least-squares problem, not a change of the structural model.
    Ws, inv_sw = _scale_rows(Wc)
    Zs, _ = _scale_rows(Z)
    Pi = _lstsq(Zs, Ws)
    Wh = Pi @ Zs
    Theta_s = _lstsq(Wh, Yc)
    Theta = Theta_s * inv_sw[None, :]
    A_hat = Theta[:, :nx]
    B_hat = Theta[:, nx:nx + nu]
    E_hat = Theta[:, nx + nu:nx + nu + nd]
    c_hat = Theta[:, -1]
    rank_Z = int(np.linalg.matrix_rank(Z.T))
    rank_W = int(np.linalg.matrix_rank(W.T))
    Phi_W = Wc @ Wc.T / Wc.shape[1]
    Phi_Z = Z @ Z.T / Z.shape[1]
    fs = _first_stage_diagnostics(Wc, Zs, reduced_rows, target_rows,
                                  Wc.shape[1])
    resid = Yc - Theta @ Wc
    overid_df = max(Z.shape[0] - W.shape[0], 0)
    overid_J = None
    overid_p = None
    if overid_df > 0:
        Szz = Z @ Z.T / Z.shape[1]
        Szz_inv = np.linalg.pinv(Szz, rcond=1e-12)
        Js, ps = [], []
        for i in range(resid.shape[0]):
            g = Z @ resid[i] / Z.shape[1]
            J = float(Z.shape[1] * g @ Szz_inv @ g)
            Js.append(J)
            ps.append(float(chi2.sf(J, overid_df)))
        overid_J = [float(x) for x in Js]
        overid_p = [float(x) for x in ps]
    return dict(A_hat=A_hat, B_hat=B_hat, E_hat=E_hat, c_hat=c_hat,
                rank=rank_W, rank_instruments=rank_Z,
                reg_dim=W.shape[0], instrument_dim=Z.shape[0],
                lam_min_Phi=float(np.min(np.linalg.eigvalsh(Phi_W))),
                lam_min_instrument_cov=float(np.min(np.linalg.eigvalsh(Phi_Z))),
                cond_Phi=float(np.linalg.cond(Phi_W)),
                method="IV/2SLS", instrument_mode=instrument_mode,
                probe_present=bool(q_present),
                overid_df=int(overid_df), overid_J=overid_J,
                overid_p=overid_p,
                overid_p_min=(None if overid_p is None else float(min(overid_p))),
                **fs)


def identify_adversary_block(data, use_iv=False, n_lags=1):
    """Estimate ``d-q = N x + z + noise`` from measured disturbance data.

    The disturbance channel is observed in this validation setting.  Current
    state measurement error can be handled with lagged-state instruments;
    OLS is returned as a comparison.  The ``N`` estimate is not obtained from
    the state equation, so it is not confounded with ``E``.
    """
    X = data["X"]
    D = data["D"]
    Q = data.get("Q")
    target = D - Q if Q is not None else D
    N = X.shape[1]
    W = np.vstack([X, np.ones((1, N))])
    if not use_iv:
        Theta = _lstsq(W, target)
        Z = None
    else:
        start = int(n_lags)
        blocks = [X[:, start - lag:N - lag] for lag in range(1, n_lags + 1)]
        if Q is not None and Q.size and np.std(Q) > 1e-14:
            blocks.extend([Q[:, start - lag:N - lag]
                           for lag in range(1, n_lags + 1)])
        blocks.append(np.ones((1, N - start)))
        Z = np.vstack(blocks)
        Wc = W[:, start:]
        Tc = target[:, start:]
        Ws, inv_sw = _scale_rows(Wc)
        Zs, _ = _scale_rows(Z)
        Pi = _lstsq(Zs, Ws)
        Wh = Pi @ Zs
        Theta_s = _lstsq(Wh, Tc)
        Theta = Theta_s * inv_sw[None, :]
    N_hat = Theta[:, :X.shape[0]]
    z_hat = Theta[:, -1]
    Phi = W @ W.T / N
    out = dict(N_hat=N_hat, z_hat=z_hat, rank=int(np.linalg.matrix_rank(W.T)),
               reg_dim=W.shape[0], lam_min_Phi=float(np.min(np.linalg.eigvalsh(Phi))),
               method="IV/2SLS" if use_iv else "OLS")
    if Z is not None:
        fs = _first_stage_diagnostics(
            Wc, Zs, reduced_rows=np.array([Zs.shape[0] - 1]),
            target_rows=list(range(X.shape[0])), n_total=Wc.shape[1])
        out.update(rank_instruments=int(np.linalg.matrix_rank(Z.T)),
                   instrument_dim=Z.shape[0], **fs)
    return out


def identifiability_certificate(data, est):
    """Return an explicit statement of what the feedback data can identify."""
    Q = data.get("Q")
    probe_present = bool(Q is not None and Q.size and np.std(Q) > 1e-14)
    rank_ok = bool(est.get("rank") == est.get("reg_dim"))
    e_f = est.get("E_first_stage_min_F", est.get("first_stage_min_F"))
    n_f = est.get("N_first_stage_min_F", est.get("first_stage_min_F"))
    strength_ok = (e_f is not None and e_f >= 10.0)
    if not probe_present:
        e_status = "not_identifiable"
        reason = ("no exogenous disturbance-channel instrument: d=Nx+z+xi "
                  "is feedback-correlated with the state and xi enters the "
                  "transition error; E is confounded with A+EN")
    elif est.get("method") == "OLS":
        e_status = "not_identifiable_with_OLS"
        reason = ("d is feedback-correlated with the state and its innovation "
                  "enters the transition error; OLS is not an identification "
                  "design for E even when a probe is present")
    elif not rank_ok:
        e_status = "rank_deficient"
        reason = "closed-loop [x;u;d;1] regressor matrix is rank deficient"
    elif not strength_ok:
        e_status = "weak_instrument"
        reason = "excluded disturbance probe does not satisfy the first-stage strength threshold F>=10"
    else:
        e_status = "identifiable"
        reason = "full rank and strong exogenous disturbance-channel probe"
    n_status = ("identifiable" if rank_ok and (n_f is None or n_f >= 10.0)
                else "weak_instrument" if rank_ok else "rank_deficient")
    return dict(E_status=e_status, N_status=n_status,
                probe_present=probe_present, rank_ok=rank_ok,
                first_stage_min_F=e_f,
                strength_ok=strength_ok, reason=reason)


def full_identification_errors(spec, est, opt=None):
    """Relative errors for the full ``A,B,E`` estimate."""
    A, B, E = spec["A"], spec["B"], spec["E"]
    out = dict(
        eA=float(np.linalg.norm(est["A_hat"] - A) / np.linalg.norm(A)),
        eB=float(np.linalg.norm(est["B_hat"] - B) / np.linalg.norm(B)),
        eE=float(np.linalg.norm(est["E_hat"] - E) / np.linalg.norm(E)),
    )
    if "N_hat" in est:
        out["eN"] = float(np.linalg.norm(est["N_hat"] - opt["N"]) /
                          max(np.linalg.norm(opt["N"]), 1e-15))
        out["ez"] = float(np.linalg.norm(est["z_hat"] - opt["z"]))
    if opt is not None:
        out["policy_error_note"] = "computed separately after identified-model PI"
    return out
