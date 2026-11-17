#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The two paper scenarios, with parameters transcribed from the manuscript
(CAAI_Revised_Highlighted.tex, Section VII-A / VII-B) and from the verified
claims index 00_paper_claims.md.

Microgrid model version note (model v2, 2026-11-12 revision):
The microgrid is now the STANDARD symmetric two-area LFC swing model. The
earlier legacy implementation (Ac first row carried -T12/(2H1) with no tie-line
term in area 2, and the load / wind columns of Ec shared the same sign) was
found inconsistent with the standard swing equations; this is corrected here.
humanoid is untouched; its numbers are regression-pinned.

Sign convention (also printed by model_v2_matrices.py and in the README):
    state  x = [df1, dPm1, df2, dPm2]
    input  u = [dPc1, dPc2]
    dist   d = [dPN1, dPN2, dPW1, dPW2]
    dPN > 0  : load increase   (pulls frequency DOWN)
    dPW > 0  : wind generation  increase (pushes frequency UP)
    tie line dP_tie = T12 (df1 - df2).
"""

import numpy as np
from drc_core import euler_discretize


def build_humanoid():
    """3-joint humanoid leg, 6th-order MIMO.

    Parameters (tex L1548-1553):
      J = [0.5, 0.3, 0.2], B(damping) = [0.8, 0.5, 0.3], Ts = 0.01.
      State per joint: [angle, velocity]; input torque per joint;
      disturbance = external torque per joint.
    """
    J = np.array([0.5, 0.3, 0.2])
    Bp = np.array([0.8, 0.5, 0.3])
    Ts = 0.01
    Ac = np.zeros((6, 6))
    Bc = np.zeros((6, 3))
    Ec = np.zeros((6, 3))
    for i in range(3):
        idx = 2 * i
        Ac[idx, idx + 1] = 1.0
        Ac[idx + 1, idx + 1] = -Bp[i] / J[i]
        Bc[idx + 1, i] = 1.0 / J[i]
        Ec[idx + 1, i] = 1.0 / J[i]
    return euler_discretize(Ac, Bc, Ec, Ts)


def microgrid_continuous():
    """Continuous-time symmetric two-area LFC matrices (model v2).

    Returns (Ac, Bc, Ec, Ts) so that the matrix dump can print the continuous
    form as well as the forward-Euler discrete model used in simulation.
    """
    H = np.array([5.0, 4.0])
    D = np.array([0.8, 0.6])
    Tg = np.array([0.5, 0.4])
    T12 = 0.2
    Ts = 0.1
    Ac = np.array([
        [-(D[0] + T12) / (2 * H[0]), 1 / (2 * H[0]), +T12 / (2 * H[0]), 0.0],
        [-1 / Tg[0], -1 / Tg[0], 0.0, 0.0],
        [+T12 / (2 * H[1]), 0.0, -(D[1] + T12) / (2 * H[1]), 1 / (2 * H[1])],
        [0.0, 0.0, -1 / Tg[1], -1 / Tg[1]]])
    Bc = np.array([[0.0, 0.0], [1 / Tg[0], 0.0],
                   [0.0, 0.0], [0.0, 1 / Tg[1]]])
    Ec = np.array([
        [-1 / (2 * H[0]), 0.0, +1 / (2 * H[0]), 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1 / (2 * H[1]), 0.0, +1 / (2 * H[1])],
        [0.0, 0.0, 0.0, 0.0]])
    return Ac, Bc, Ec, Ts


def build_microgrid():
    """Two-area islanded microgrid, 4th-order MIMO (standard symmetric LFC).

    Parameters (tex L1659-1661, L1661):
      H = [5.0, 4.0], D = [0.8, 0.6], Tg = [0.5, 0.4], T12 = 0.2, Ts = 0.1.
    Forward-Euler discretization (unchanged method, Ts = 0.1).
    """
    Ac, Bc, Ec, Ts = microgrid_continuous()
    return euler_discretize(Ac, Bc, Ec, Ts)


# Output matrices y = C x (frequency / joint-angle measurements)
C_HUMANOID = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0]], dtype=float)
C_MICROGRID = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]], dtype=float)


def humanoid_spec():
    A, B, E = build_humanoid()
    return dict(
        name="humanoid",
        A=A, B=B, E=E, C=C_HUMANOID,
        Q=np.diag([200, 20, 200, 20, 200, 20]).astype(float),
        R=np.diag([0.05, 0.05, 0.05]).astype(float),
        d_bar=np.array([5.0, 3.0, 2.0]),
        d_std=np.array([1.0, 0.6, 0.4]),
        x_ref=np.zeros(6),
        alpha=0.95, lam=5.0,
        T=3000, ss=1500,
        mae_mode="full",
        seeds=list(range(1000, 1020)),
        x0=np.array([0.30, 0, 0.30, 0, 0.30, 0]),
    )


def microgrid_spec():
    A, B, E = build_microgrid()
    return dict(
        name="microgrid",
        A=A, B=B, E=E, C=C_MICROGRID,
        Q=np.diag([1000, 1, 1000, 1]).astype(float),
        R=np.diag([0.1, 0.1]).astype(float),
        d_bar=np.array([0.15, 0.10, 0.20, 0.15]),
        d_std=np.sqrt(np.array([0.005, 0.003, 0.010, 0.006])),
        x_ref=np.zeros(4),
        alpha=0.95, lam=5.0,
        T=40000, ss=30000,
        mae_mode="freq",
        seeds=list(range(1000, 1020)),
        x0=np.array([0.1, 0, 0.1, 0]),
    )


# --------------------------------------------------------------------------- #
# Current v3 manuscript reference values (numerically identical to the v2
# corrected tables at the time this package was generated). These are
# COMPARISON CONSTANTS ONLY; every measured value in the experiment tables and
# figures is recomputed from the solver/Monte-Carlo run. Legacy model values
# are never used as the primary paper outputs.
# --------------------------------------------------------------------------- #
PAPER = dict(
    humanoid=dict(
        drc_mae=(0.0455, 0.0009), lqr_mae=(0.291, 0.001), advantage=84.4,
        # Table 2: shift -> (LQR MAE, DRC MAE, advantage %)
        shifts=[(0.0, 0.2910, 0.0455, 84.4),
                (0.5, 0.4338, 0.1689, 61.1),
                (1.0, 0.5772, 0.3094, 46.4),
                (2.0, 0.8644, 0.5939, 31.3),
                (3.0, 1.1519, 0.8794, 23.7)],
        # theory scalars
        iters=420, lam_min_P=31.78, rho_G=0.985, p_star=[-4.53, -2.86, -1.96],
        rank_O3=6,
        # ablation rows: humanoid MAE / microgrid cost / microgrid MAE
        ablation=dict(
            lqr_pure=(0.291, 0.745, 0.0049),
            affine_lqr=(0.039, 0.745, 0.0049),
            drc_no_bias=(0.289, 0.591, 0.0036),
            drc_affine=(0.0455, 0.489, 0.0028),
        ),
        # lambda sensitivity: lambda -> (MAE, cost)
        lambda_humanoid={5: (0.0455, 41.16), 10: (0.0462, 41.38),
                         20: (0.0465, 41.50)},
    ),
    microgrid=dict(
        drc_cost=(0.489, 0.005), lqr_cost=(0.745, 0.009),
        drc_mae=(0.0028, 0.00003), lqr_mae=(0.0049, 0.00003),
        cost_advantage=34.5,
        # robustness table: nominal / shifted cost and MAE
        nom_cost=(0.489, 0.745), shf_cost=(0.914, 1.606),
        deg_cost=(87.1, 115.4),
        nom_mae=(0.0028, 0.0049), shf_mae=(0.0035, 0.0073),
        iters=76, lam_min_P=1.756, rho_G=0.802, p_star=[-0.345, -0.362],
        lambda_micro={5: (0.0028, 0.489), 10: (0.0027, 0.403),
                      20: (0.0028, 0.394)},
        lambda_low_note="lambda=1,2: no stable converged solution",
        mean_only_cost=(0.581, 1.320),  # DRC, LQR
    ),
)
