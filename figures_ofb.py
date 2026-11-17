#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate output-feedback figures with unknown disturbance history."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models
import drc_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "output", "figures")
os.makedirs(FIG, exist_ok=True)
C_DRC, C_LQR = "#1f77b4", "#d9534f"


def _kf_traj(spec, K, p, x0, steps, seed):
    A, B, C, E = spec["A"], spec["B"], spec["C"], spec["E"]
    nx = A.shape[0]
    Qw = E @ np.diag(spec["d_std"] ** 2) @ E.T + 1e-10 * np.eye(nx)
    Rv = 1e-16 * np.eye(C.shape[0])  # 1e-8 std
    L, _, _ = core.kalman_observer_gain(A, C, Qw, Rv)
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    xh = spec["x_ref"].copy()
    xs, us, ds = [], [], []
    for _ in range(steps):
        u = K @ (xh - spec["x_ref"]) + p
        d = spec["d_bar"] + spec["d_std"] * rng.standard_normal(len(spec["d_bar"]))
        y = C @ x
        xs.append(x.copy()); us.append(u.copy()); ds.append(d.copy())
        x = A @ x + B @ u + E @ d
        xh = A @ xh + B @ u + E @ spec["d_bar"] + L @ (y - C @ xh)
    return np.asarray(xs), np.asarray(us), np.asarray(ds)


def _mc_ofb(spec, K, p):
    r = core.mc_batch_output_feedback_kf(
        spec["A"], spec["B"], spec["C"], spec["E"], K, p, spec["x_ref"],
        spec["d_bar"], spec["d_std"], spec["T"], spec["ss"], spec["alpha"],
        spec["mae_mode"], spec["Q"], spec["R"], seeds=spec["seeds"],
        x0=spec["x0"])
    return r


def make_humanoid():
    spec = models.humanoid_spec()
    d = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                          spec["R"], spec["alpha"], spec["lam"], spec["d_bar"])
    Klqr, _, _, _ = core.solve_discounted_lqr(spec["A"], spec["B"], spec["Q"],
                                              spec["R"], spec["alpha"])
    xd, ud, dd = _kf_traj(spec, d["K"], d["p"], spec["x0"], 300, 7)
    xl, ul, _ = _kf_traj(spec, -Klqr, np.zeros(3), spec["x0"], 300, 7)
    md, ml = _mc_ofb(spec, d["K"], d["p"]), _mc_ofb(spec, -Klqr, np.zeros(3))
    t = np.arange(300) * 0.01
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.0))
    labels = ["Hip", "Knee", "Ankle"]
    for j in range(3):
        c = 2 * j
        ax[0, j].plot(t, xd[:, c], color=C_DRC, label="DRC (unknown-history OFB)")
        ax[0, j].plot(t, xl[:, c], "--", color=C_LQR, label="LQR (unknown-history OFB)")
        ax[0, j].set_title(f"{labels[j]} Joint Angle Tracking")
        ax[0, j].set_ylabel("Joint angle (rad)")
        ax[1, j].plot(t, ud[:, j], color=C_DRC, label="DRC")
        ax[1, j].plot(t, ul[:, j], "--", color=C_LQR, label="LQR")
        ax[1, j].set_title(f"{labels[j]} Control Torque")
        ax[1, j].set_ylabel("Torque (N·m)")
    for j in range(2):
        ax[2, j].plot(t, dd[:, j], color="#2ca02c")
        ax[2, j].axhline(spec["d_bar"][j], color="red", ls="--")
        ax[2, j].set_title(f"External Disturbance {j+1}")
        ax[2, j].set_xlabel("Time (s)")
    bars = ax[2, 2].bar(["DRC", "LQR"],
                        [md["mae_mean"], ml["mae_mean"]], color=[C_DRC, C_LQR])
    ax[2, 2].set_title("20-Seed Steady-State MAE")
    ax[2, 2].set_ylabel(r"MAE $\|x-x_{\rm ref}\|$ (rad)")
    for b, v in zip(bars, [md["mae_mean"], ml["mae_mean"]]):
        ax[2, 2].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=9)
    for r in range(3):
        for c in range(3):
            ax[r, c].grid(alpha=0.3)
            h, l = ax[r, c].get_legend_handles_labels()
            if h:
                ax[r, c].legend(fontsize=7)
    fig.suptitle("Unknown-History Output-Feedback Humanoid Benchmark (KF Estimator)",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG, "sim_humanoid_mimo_ofb.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "sim_humanoid_mimo_ofb.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    return dict(drc=md, lqr=ml)


def make_microgrid():
    spec = models.microgrid_spec()
    d = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                          spec["R"], spec["alpha"], spec["lam"], spec["d_bar"])
    Klqr, _, _, _ = core.solve_discounted_lqr(spec["A"], spec["B"], spec["Q"],
                                              spec["R"], spec["alpha"])
    xd, ud, dd = _kf_traj(spec, d["K"], d["p"], spec["x0"], 300, 7)
    xl, ul, _ = _kf_traj(spec, -Klqr, np.zeros(2), spec["x0"], 300, 7)
    md, ml = _mc_ofb(spec, d["K"], d["p"]), _mc_ofb(spec, -Klqr, np.zeros(2))
    t = np.arange(300) * 0.1
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.0))
    ax[0, 0].plot(t, xd[:, 0], color=C_DRC, label="DRC")
    ax[0, 0].plot(t, xl[:, 0], "--", color=C_LQR, label="LQR")
    ax[0, 0].set_title("Area 1 Frequency")
    ax[0, 0].set_ylabel(r"$\Delta f_1$ (pu)")
    ax[0, 1].plot(t, xd[:, 2], color=C_DRC)
    ax[0, 1].plot(t, xl[:, 2], "--", color=C_LQR)
    ax[0, 1].set_title("Area 2 Frequency")
    ax[0, 1].set_ylabel(r"$\Delta f_2$ (pu)")
    bars = ax[0, 2].bar(["DRC", "LQR"],
                        [md["cost_mean"], ml["cost_mean"]], color=[C_DRC, C_LQR])
    ax[0, 2].set_title("20-Seed Discounted Cost")
    ax[0, 2].set_ylabel(r"$J_\gamma$")
    for b, v in zip(bars, [md["cost_mean"], ml["cost_mean"]]):
        ax[0, 2].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=9)
    ax[1, 0].plot(t, ud[:, 0], color=C_DRC)
    ax[1, 0].plot(t, ul[:, 0], "--", color=C_LQR)
    ax[1, 0].set_title("Area 1 Governor Control")
    ax[1, 1].plot(t, ud[:, 1], color=C_DRC)
    ax[1, 1].plot(t, ul[:, 1], "--", color=C_LQR)
    ax[1, 1].set_title("Area 2 Governor Control")
    ax[1, 2].plot(t, dd[:, 0], label="Load 1")
    ax[1, 2].plot(t, dd[:, 2], label="Wind 1")
    ax[1, 2].set_title("Disturbance Samples")
    ax[1, 2].legend(fontsize=7)
    ax[2, 0].plot(t, xd[:, 0], color=C_DRC)
    ax[2, 0].plot(t, xl[:, 0], "--", color=C_LQR)
    ax[2, 0].set_title("Area 1 Frequency: OFB")
    ax[2, 0].set_ylabel(r"$\Delta f_1$ (pu)")
    ax[2, 1].plot(t, xd[:, 2], color=C_DRC)
    ax[2, 1].plot(t, xl[:, 2], "--", color=C_LQR)
    ax[2, 1].set_title("Area 2 Frequency: OFB")
    ax[2, 1].set_ylabel(r"$\Delta f_2$ (pu)")
    ax[2, 2].bar(["DRC", "LQR"], [md["mae_mean"], ml["mae_mean"]],
                 color=[C_DRC, C_LQR])
    ax[2, 2].set_title("20-Seed Frequency MAE")
    ax[2, 2].set_ylabel("MAE")
    for r in range(3):
        for c in range(3):
            ax[r, c].grid(alpha=0.3)
            h, l = ax[r, c].get_legend_handles_labels()
            if h:
                ax[r, c].legend(fontsize=7)
    fig.suptitle("Unknown-History Output-Feedback Microgrid Benchmark (KF Estimator)",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG, "sim_microgrid_mimo_ofb.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "sim_microgrid_mimo_ofb.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    return dict(drc=md, lqr=ml)


def make_all():
    return dict(humanoid=make_humanoid(), microgrid=make_microgrid())


if __name__ == "__main__":
    make_all()
