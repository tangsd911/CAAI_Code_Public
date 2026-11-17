#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonzero constant-reference experiment.

This is a direct numerical check of the nominal affine-bias identity

    p = B^dagger ((I - A) x_ref - E d_bar),

used in the affine nominal LQR baseline.  The humanoid has a feasible
constant equilibrium for the selected reference because the required bias
vector lies in range(B).  The main experiments retain x_ref = 0 so that they
remain comparable to the original tables.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

import drc_core as core
import models

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
TBL = OUT / "tables"
LOG = OUT / "logs"
for directory in (OUT, TBL, LOG):
    directory.mkdir(parents=True, exist_ok=True)


def _steady_error_norm(spec, K, p, x_ref, x0):
    """Deterministic steady-window error for the constant reference."""
    A, B, E = spec["A"], spec["B"], spec["E"]
    x = np.asarray(x0, dtype=float).copy()
    errors = []
    for _ in range(spec["T"]):
        e = x - x_ref
        u = K @ e + p
        x = A @ x + B @ u + E @ spec["d_bar"]
        if _ >= spec["ss"]:
            errors.append(float(np.linalg.norm(e)))
    return float(np.mean(errors))


def run():
    spec = models.humanoid_spec()
    A, B, E = spec["A"], spec["B"], spec["E"]
    x_ref = np.array([0.20, 0.0, 0.30, 0.0, 0.20, 0.0])
    x0 = x_ref + spec["x0"]
    K, rho_lqr, _, _ = core.solve_discounted_lqr(
        A, B, spec["Q"], spec["R"], spec["alpha"])
    p_bias = np.linalg.pinv(B) @ ((np.eye(A.shape[0]) - A) @ x_ref - E @ spec["d_bar"])
    bias_residual = float(np.linalg.norm(B @ p_bias - ((np.eye(A.shape[0]) - A) @ x_ref - E @ spec["d_bar"])))

    rows = []
    for label, p in (("pure LQR", np.zeros(B.shape[1])), ("affine LQR bias", p_bias)):
        det_mae = _steady_error_norm(spec, -K, p, x_ref, x0)
        mc = core.mc_batch(
            A, B, E, -K, p, x_ref, spec["d_bar"], spec["d_std"],
            spec["T"], spec["ss"], spec["alpha"], spec["mae_mode"],
            spec["Q"], spec["R"], seeds=spec["seeds"], x0=x0)
        rows.append({
            "scenario": "humanoid",
            "reference": x_ref.tolist(),
            "controller": label,
            "seeds": len(spec["seeds"]),
            "deterministic steady error norm": det_mae,
            "MC MAE": float(mc[0]),
            "MC cost": float(mc[2]),
            "bias residual": bias_residual,
            "rho_G": float(rho_lqr),
        })

    with (TBL / "e22_nonzero_constant_reference.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (TBL / "e22_nonzero_constant_reference.md").open("w", encoding="utf-8") as f:
        headers = list(rows[0])
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
        f.write("\nThe affine bias is computed from the nominal model and d_bar. "
                "The deterministic error uses zero measurement disturbance and the MC error uses the nominal Gaussian disturbance law.\n")
    with (LOG / "e22_nonzero_constant_reference.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "p_bias": p_bias.tolist(),
                   "bias_residual": bias_residual}, f, indent=2)
    print(json.dumps({"bias_residual": bias_residual, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run()
