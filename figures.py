#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate the two 3x3 MIMO figures.

All bar heights and bar annotations come from Monte-Carlo numbers computed in
this run (core.mc_batch on the solved controllers); no result literal is used
for a bar height or annotation. Trajectories are single-seed rollouts from the
paper-specified x0.

Outputs go to output/figures/; the manuscript PDFs in the parent directory are
never touched.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drc_core as core
import models

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "output", "figures")
os.makedirs(FIG, exist_ok=True)

C_DRC, C_LQR = "#1f77b4", "#d9534f"


def _legend(ax, fontsize=8):
    """Add a legend only when the panel actually has labeled artists."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fontsize)


def _traj(A, B, E, K, p, mean, std, x0, steps, seed, nd):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    xs, us, ds = [], [], []
    for _ in range(steps):
        u = K @ x + p
        d = mean + std * rng.standard_normal(nd)
        xs.append(x.copy())
        us.append(u.copy())
        ds.append(d.copy())
        x = A @ x + B @ u + E @ d
    return np.array(xs), np.array(us), np.array(ds)


def _mc(spec, K, p, N=None, z=None, disturb="nominal", d_mean=None,
        d_std=None):
    d_mean = spec["d_bar"] if d_mean is None else d_mean
    d_std = spec["d_std"] if d_std is None else d_std
    return core.mc_batch(spec["A"], spec["B"], spec["E"], K, p, spec["x_ref"],
                         d_mean, d_std, spec["T"], spec["ss"], spec["alpha"],
                         spec["mae_mode"], spec["Q"], spec["R"],
                         seeds=spec["seeds"], N=N, z=z, disturb=disturb)


def make_humanoid_fig():
    spec = models.humanoid_spec()
    A, B, E = spec["A"], spec["B"], spec["E"]
    drc = core.solve_drc_pi(A, B, E, spec["Q"], spec["R"], spec["alpha"],
                            spec["lam"], spec["d_bar"])
    Klqr, _, _, _ = core.solve_discounted_lqr(A, B, spec["Q"], spec["R"],
                                              spec["alpha"])

    # real MC numbers for the performance bars (full-state norm MAE)
    d_mc = _mc(spec, drc["K"], drc["p"])
    l_mc = _mc(spec, -Klqr, np.zeros(3))
    drc_mae, lqr_mae = d_mc[0], l_mc[0]

    x0 = spec["x0"]
    t = np.arange(300) * 0.01
    xd, ud, dd = _traj(A, B, E, drc["K"], drc["p"], spec["d_bar"],
                       spec["d_std"], x0, 300, 7, 3)
    xl, ul, _ = _traj(A, B, E, -Klqr, np.zeros(3), spec["d_bar"],
                      spec["d_std"], x0, 300, 7, 3)

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.0))
    titles = ["Hip", "Knee", "Ankle"]
    for j in range(3):
        c = 2 * j
        ax[0, j].plot(t, xd[:, c], color=C_DRC,
                      label="DRC (model-based full state)")
        ax[0, j].plot(t, xl[:, c], "--", color=C_LQR, label="LQR")
        ax[0, j].axhline(0, color="gray", ls=":")
        ax[0, j].set_title(f"{titles[j]} Joint Angle Tracking")
        ax[0, j].set_ylabel("Joint angle (rad)")
        ax[0, j].set_ylim(-0.02, 0.33)
    for j in range(3):
        ax[1, j].plot(t, ud[:, j], color=C_DRC,
                      label="DRC (model-based full state)")
        ax[1, j].plot(t, ul[:, j], "--", color=C_LQR, label="LQR")
        ax[1, j].set_title(f"{titles[j]} Control Torque")
        ax[1, j].set_ylabel("Torque (N·m)")
    for j in range(2):
        ax[2, j].plot(t, dd[:, j], color="#2ca02c")
        ax[2, j].axhline(spec["d_bar"][j], color="red", ls="--",
                         label=f"Mean = {spec['d_bar'][j]:.1f}")
        ax[2, j].set_title(f"{titles[j]} External Disturbance")
        ax[2, j].set_ylabel("Ext. torque (N·m)")
        ax[2, j].set_xlabel("Time (s)")
        _legend(ax[2, j])

    # performance bars: measured MC MAE
    bars = ax[2, 2].bar(["DRC", "LQR"], [drc_mae, lqr_mae],
                        color=[C_DRC, C_LQR])
    ax[2, 2].set_title("Steady-State Full-State Error")
    ax[2, 2].set_ylabel(r"$\|x-x_{\rm ref}\|$")
    for b, v in zip(bars, [drc_mae, lqr_mae]):
        ax[2, 2].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=9)

    for r in range(3):
        for c in range(3):
            ax[r, c].grid(alpha=0.3)
            if c < 2 or r < 2:
                _legend(ax[r, c])
            if r == 0:
                ax[r, c].set_xlabel("Time (s)")
    fig.suptitle(r"Model-Based Full-State Humanoid MIMO Benchmark: DRC vs LQR",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG, "sim_humanoid_mimo.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "sim_humanoid_mimo.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    return dict(benchmark="model-based full state", drc_mae=drc_mae,
                lqr_mae=lqr_mae)


def make_microgrid_fig():
    spec = models.microgrid_spec()
    A, B, E = spec["A"], spec["B"], spec["E"]
    drc = core.solve_drc_pi(A, B, E, spec["Q"], spec["R"], spec["alpha"],
                            spec["lam"], spec["d_bar"])
    Klqr, _, _, _ = core.solve_discounted_lqr(A, B, spec["Q"], spec["R"],
                                              spec["alpha"])

    # real MC numbers: nominal cost / MAE and shifted MAE
    nd_mc = _mc(spec, drc["K"], drc["p"])
    nl_mc = _mc(spec, -Klqr, np.zeros(2))
    sd_mc = _mc(spec, drc["K"], drc["p"], d_mean=1.5 * spec["d_bar"],
                d_std=np.sqrt(2) * spec["d_std"])
    sl_mc = _mc(spec, -Klqr, np.zeros(2), d_mean=1.5 * spec["d_bar"],
                d_std=np.sqrt(2) * spec["d_std"])
    drc_cost, lqr_cost = nd_mc[2], nl_mc[2]
    drc_mae = [nd_mc[0], sd_mc[0]]
    lqr_mae = [nl_mc[0], sl_mc[0]]

    x0 = spec["x0"]
    t = np.arange(300) * 0.1
    xnd, und, dn = _traj(A, B, E, drc["K"], drc["p"], spec["d_bar"],
                         spec["d_std"], x0, 300, 7, 4)
    xnl, unl, _ = _traj(A, B, E, -Klqr, np.zeros(2), spec["d_bar"],
                        spec["d_std"], x0, 300, 7, 4)
    xsd, usd, _ = _traj(A, B, E, drc["K"], drc["p"], 1.5 * spec["d_bar"],
                        np.sqrt(2) * spec["d_std"], x0, 300, 7, 4)
    xsl, usl, _ = _traj(A, B, E, -Klqr, np.zeros(2), 1.5 * spec["d_bar"],
                        np.sqrt(2) * spec["d_std"], x0, 300, 7, 4)

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.0))
    ax[0, 0].plot(t, xnd[:, 0], color=C_DRC, label="DRC")
    ax[0, 0].plot(t, xnl[:, 0], "--", color=C_LQR, label="LQR")
    ax[0, 0].set_title("Area 1 Frequency Response (Nominal)")
    ax[0, 0].set_ylabel(r"$\Delta f_1$ (pu)")
    ax[0, 1].plot(t, xnd[:, 2], color=C_DRC)
    ax[0, 1].plot(t, xnl[:, 2], "--", color=C_LQR)
    ax[0, 1].set_title("Area 2 Frequency Response (Nominal)")
    ax[0, 1].set_ylabel(r"$\Delta f_2$ (pu)")

    bars = ax[0, 2].bar(["DRC", "LQR"], [drc_cost, lqr_cost],
                        color=[C_DRC, C_LQR])
    ax[0, 2].set_title("Discounted Cost (Nominal)")
    ax[0, 2].set_ylabel(r"$J_\gamma$")
    for b, v in zip(bars, [drc_cost, lqr_cost]):
        ax[0, 2].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=9)

    ax[1, 0].plot(t, und[:, 0], color=C_DRC)
    ax[1, 0].plot(t, unl[:, 0], "--", color=C_LQR)
    ax[1, 0].set_title("Area 1 Governor Control")
    ax[1, 0].set_ylabel(r"$\Delta P_{c1}$ (pu)")
    ax[1, 1].plot(t, und[:, 1], color=C_DRC)
    ax[1, 1].plot(t, unl[:, 1], "--", color=C_LQR)
    ax[1, 1].set_title("Area 2 Governor Control")
    ax[1, 1].set_ylabel(r"$\Delta P_{c2}$ (pu)")
    ax[1, 2].plot(t, dn[:, 0], color="#2ca02c", label="Load $N_1$")
    ax[1, 2].plot(t, dn[:, 1], color="#9467bd", label="Load $N_2$")
    ax[1, 2].plot(t, dn[:, 2], color="#17becf", label="Wind $w_1$")
    ax[1, 2].plot(t, dn[:, 3], color="#e377c2", label="Wind $w_2$")
    ax[1, 2].set_title("Disturbances (Nominal)")
    ax[1, 2].set_ylabel("d (pu)")
    _legend(ax[1, 2])

    ax[2, 0].plot(t, xsd[:, 0], color=C_DRC, label="DRC (shifted)")
    ax[2, 0].plot(t, xsl[:, 0], "--", color=C_LQR, label="LQR (shifted)")
    ax[2, 0].set_title(r"Area 1: 50% Mean + 100% Var. Shift")
    ax[2, 0].set_ylabel(r"$\Delta f_1$ (pu)")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].plot(t, xsd[:, 2], color=C_DRC)
    ax[2, 1].plot(t, xsl[:, 2], "--", color=C_LQR)
    ax[2, 1].set_title(r"Area 2: 50% Mean + 100% Var. Shift")
    ax[2, 1].set_ylabel(r"$\Delta f_2$ (pu)")
    ax[2, 1].set_xlabel("Time (s)")

    labels = ["Nominal", "Shifted"]
    xx = np.arange(2)
    w = 0.38
    ax[2, 2].bar(xx - w / 2, drc_mae, w, label="DRC", color=C_DRC)
    ax[2, 2].bar(xx + w / 2, lqr_mae, w, label="LQR", color=C_LQR)
    ax[2, 2].set_xticks(xx)
    ax[2, 2].set_xticklabels(labels)
    ax[2, 2].set_ylabel(r"Steady-state freq. MAE")
    ax[2, 2].set_title("Distributional Robustness")
    _legend(ax[2, 2])
    for i, (dv, lv) in enumerate(zip(drc_mae, lqr_mae)):
        ax[2, 2].text(i - w / 2, dv, f"{dv:.3f}", ha="center", va="bottom",
                      fontsize=8)
        ax[2, 2].text(i + w / 2, lv, f"{lv:.3f}", ha="center", va="bottom",
                      fontsize=8)

    for r in range(3):
        for c in range(3):
            ax[r, c].grid(alpha=0.3)
            if c < 2 and r < 2:
                _legend(ax[r, c])
    fig.suptitle(r"Model-Based Full-State Two-Area Microgrid Benchmark: DRC vs LQR",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG, "sim_microgrid_mimo.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "sim_microgrid_mimo.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    return dict(benchmark="model-based full state",
                drc_cost=drc_cost, lqr_cost=lqr_cost,
                drc_mae=drc_mae, lqr_mae=lqr_mae)


def make_all():
    return dict(humanoid=make_humanoid_fig(),
                microgrid=make_microgrid_fig())


if __name__ == "__main__":
    make_all()
