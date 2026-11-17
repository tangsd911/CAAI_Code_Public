#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace a deliberately non-stabilizing initial policy to a stabilizing SDPI policy.

The trace is a deterministic humanoid closed-loop illustration.  The first
``switch_step`` plant steps use the initial policy; the remaining steps use the
verified SDPI policy.  A second panel records the spectral radius at the initial
policy, the scaled initializer, and every logged SDPI iterate.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import drc_core as core
import models
import sdpi

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
TBL = OUT / "tables"
LOG = OUT / "logs"
FIG = OUT / "figures"
for directory in (OUT, TBL, LOG, FIG):
    directory.mkdir(parents=True, exist_ok=True)


def run(switch_step=20, steps=300):
    spec = models.humanoid_spec()
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    V = spec["d_bar"].reshape(1, -1)

    Klqr, _, _, _ = core.solve_discounted_lqr(A, B, Q, R, alpha)
    K0 = -3.4 * Klqr
    p0 = np.zeros(B.shape[1])
    N0 = np.zeros((E.shape[1], A.shape[0]))
    z0 = np.zeros(E.shape[1])

    bad_rho = sdpi.rho(A + B @ K0 + E @ N0)
    res = sdpi.solve_sdpi(
        A, B, E, Q, R, alpha, lam, V,
        K0=K0, p0=p0, N0=N0, z0=z0,
        max_phase2=30)
    init_info = res["initialization"]
    attempts = init_info.get("scaled_candidate_attempts", [])
    scaled_rho = float(attempts[-1]["rho_original"]) if attempts else float(res["rho_G"])

    x = spec["x0"].copy()
    trace_rows = []
    for k in range(steps):
        phase = "initial_unstable" if k < switch_step else "stabilized"
        if k < switch_step:
            K, p = K0, p0
        else:
            K, p = res["K"], res["p"]
        e = x - spec["x_ref"]
        u = K @ e + p
        d = spec["d_bar"]
        err = float(np.linalg.norm(e))
        trace_rows.append({
            "step": k,
            "phase": phase,
            "error_norm": err,
            "control_norm": float(np.linalg.norm(u)),
            "state_norm": float(np.linalg.norm(x)),
        })
        x = A @ x + B @ u + E @ d

    stage_rows = []
    stage_rows.append({"stage": 0, "phase": "initial", "rho_G": bad_rho,
                       "beta_scale": 1.0})
    stage_rows.append({"stage": 1, "phase": "scaled initializer", "rho_G": scaled_rho,
                       "beta_scale": float(init_info["beta_scale"])})
    for i, h in enumerate(res["history"]):
        stage_rows.append({"stage": 2 + i, "phase": h["phase"],
                           "rho_G": float(h["rho_G"]),
                           "beta_scale": float(h["beta_scale"])})

    with (TBL / "e23_initial_gain_transition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0]))
        w.writeheader(); w.writerows(trace_rows)
    with (TBL / "e23_initial_gain_transition_stages.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        w.writeheader(); w.writerows(stage_rows)
    with (TBL / "e23_initial_gain_transition.md").open("w", encoding="utf-8") as f:
        f.write("| metric | value |\n|---|---:|\n")
        f.write(f"| initial rho(G) | {bad_rho:.6f} |\n")
        f.write(f"| scaled initializer rho(G) | {scaled_rho:.6f} |\n")
        f.write(f"| final SDPI rho(G) | {res['rho_G']:.6f} |\n")
        f.write(f"| switch step | {switch_step} |\n")
        f.write(f"| peak error norm | {max(r['error_norm'] for r in trace_rows):.6f} |\n")
        f.write(f"| final error norm | {trace_rows[-1]['error_norm']:.6f} |\n")

    trace = np.asarray([r["error_norm"] for r in trace_rows])
    rho_values = np.asarray([r["rho_G"] for r in stage_rows])
    fig, ax = plt.subplots(2, 1, figsize=(7.1, 4.1), sharex=False)
    k = np.arange(steps)
    ax[0].plot(k, trace, color="#1f77b4", lw=1.7)
    ax[0].axvline(switch_step - 0.5, color="#d9534f", ls="--", lw=1.1,
                  label="switch to scaled/SDPI policy")
    ax[0].set_xlim(0, steps - 1)
    ax[0].set_ylabel(r"$\|x_k-x_{\rm ref}\|$")
    ax[0].set_xlabel("closed-loop step")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("Initialization-to-stabilization transition (humanoid)")
    ax[0].grid(alpha=0.25)

    ax[1].plot(np.arange(len(rho_values)), rho_values, "o-", color="#2ca02c", ms=4)
    ax[1].axhline(1.0, color="#d9534f", ls="--", lw=1.0)
    ax[1].axvline(1, color="gray", ls=":", lw=1.0)
    ax[1].set_yscale("linear")
    ax[1].set_xlabel("initialization / SDPI stage")
    ax[1].set_ylabel(r"$\rho(G)$")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "sim_initial_gain_transition.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "initial_rho": bad_rho,
        "scaled_initializer_rho": scaled_rho,
        "final_rho": float(res["rho_G"]),
        "phase1_steps": int(res["phase1_steps"]),
        "stop_reason": res["stop_reason"],
        "switch_step": int(switch_step),
        "peak_error_norm": float(np.max(trace)),
        "final_error_norm": float(trace[-1]),
        "stages": stage_rows,
    }
    with (LOG / "e23_initial_gain_transition.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
