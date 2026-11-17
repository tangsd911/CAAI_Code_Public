#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically verify every theorem/lemma used in the CAAI manuscript.

This file is intentionally separate from the paper's simulation tables.  It
checks the mathematical statements on the two application models:

  * Assumption 1: stabilizability, observability, reconstruction rank;
  * Wasserstein dual / affine Bellman fixed point;
  * Theorem 1: reconstruction error decomposition and N^-1/2 scaling;
  * Theorem 2: positive definiteness of P;
  * Scaling lemma: scaled-to-original policy mapping;
  * Stability iff theorem: the X-LMI certificate;
  * Stabilizing PI theorem: stability of every iterate and fixed point;
  * Data-driven consistency theorem: E/N identification and O_p(N^-1/2)
    policy/value errors.

The report records both the exact residuals and the remaining assumptions.
It does not turn a finite sample into a proof of an infinite-horizon claim.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np

import algorithm4 as a4
import data_driven as dd
import data_driven_reconstruction as ddr
import drc_core as core
import models
import rigorous_rate_validation
import sdpi
import thm3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LOG = os.path.join(OUT, "logs")
os.makedirs(LOG, exist_ok=True)


def _check(name: str, condition: bool, value=None, tol=None) -> Dict:
    rec = {"name": name, "passed": bool(condition)}
    if value is not None:
        rec["value"] = float(value)
    if tol is not None:
        rec["tolerance"] = float(tol)
    if not condition:
        raise AssertionError(f"{name} failed: value={value}, tol={tol}")
    return rec


def _rank(M: np.ndarray, tol=1e-9) -> int:
    return int(np.linalg.matrix_rank(M, tol=tol))


def _controllability(A, B):
    blocks = [B]
    for i in range(1, A.shape[0]):
        blocks.append(np.linalg.matrix_power(A, i) @ B)
    return np.hstack(blocks)


def verify_assumptions(spec: Dict) -> List[Dict]:
    A, B, C = spec["A"], spec["B"], spec["C"]
    nx = A.shape[0]
    ctrb = _controllability(A, B)
    O, _, _ = core.output_window_maps(A, B, C, spec["E"], 3)
    return [
        _check("stabilizability rank", _rank(ctrb) == nx,
               _rank(ctrb), nx),
        _check("observability-window rank", _rank(O) == nx, _rank(O), nx),
    ]


def verify_wasserstein_bellman(spec: Dict) -> List[Dict]:
    """Check the affine worst-case Bellman residual and both FOCs."""
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    d = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    K, p, N, z, P, chi, psi = (d["K"], d["p"], d["N"], d["z"],
                               d["P"], d["chi"], d["psi"])
    G, F, vbar = d["G"], d["F"], spec["d_bar"]
    rng = np.random.default_rng(20261112)
    bellman, grad_u, grad_w = [], [], []
    for _ in range(32):
        x = 2.0 * rng.standard_normal(A.shape[0])
        u = K @ x + p
        w = N @ x + z
        xn = G @ x + F
        stage = (x @ Q @ x + u @ R @ u
                 - lam * (w - vbar) @ (w - vbar))
        V = x @ P @ x + 2 * chi @ x + psi
        Vn = xn @ P @ xn + 2 * chi @ xn + psi
        bellman.append(stage + alpha * Vn - V)
        gu = (2.0 * (R + alpha * B.T @ P @ B) @ u
              + 2.0 * alpha * B.T @ P @ (A @ x + E @ w)
              + 2.0 * alpha * B.T @ chi)
        gw = (-2.0 * lam * (w - vbar)
              + 2.0 * alpha * E.T @ P @ (A @ x + B @ u + E @ w)
              + 2.0 * alpha * E.T @ chi)
        grad_u.append(np.linalg.norm(gu))
        grad_w.append(np.linalg.norm(gw))
    return [
        _check("affine Bellman residual", max(abs(np.asarray(bellman))) < 1e-8,
               max(abs(np.asarray(bellman))), 1e-8),
        _check("input FOC residual", max(grad_u) < 2e-8, max(grad_u), 2e-8),
        _check("adversary FOC residual", max(grad_w) < 2e-8, max(grad_w), 2e-8),
        _check("closed-loop G Schur", d["rho_G"] < 1.0, d["rho_G"], 1.0),
    ]


def verify_positive_P(spec: Dict) -> List[Dict]:
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    d = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    K, N, P, G = d["K"], d["N"], d["P"], d["G"]
    Phi = Q + K.T @ R @ K - lam * N.T @ N
    lyap = P - alpha * G.T @ P @ G - Phi
    lam_min = float(np.min(np.linalg.eigvalsh(P)))
    phi_min = float(np.min(np.linalg.eigvalsh(Phi)))
    lam_crit = float(alpha * np.max(np.linalg.eigvalsh(E.T @ P @ E)))
    Hww = -lam * np.eye(E.shape[1]) + alpha * E.T @ P @ E
    return [
        _check("P symmetric", np.linalg.norm(P - P.T) < 1e-9,
               np.linalg.norm(P - P.T), 1e-9),
        _check("P Lyapunov residual", np.linalg.norm(lyap) < 2e-7,
               np.linalg.norm(lyap), 2e-7),
        _check("Phi_P positive", phi_min > 0.0, phi_min, 0.0),
        _check("P positive", lam_min > 0.0, lam_min, 0.0),
        _check("Hww negative", np.max(np.linalg.eigvalsh(Hww)) < 0.0,
               np.max(np.linalg.eigvalsh(Hww)), 0.0),
        _check("fixed lambda above Hww threshold", lam > lam_crit,
               lam - lam_crit, 0.0),
    ]


def verify_reconstruction_bound(spec: Dict) -> List[Dict]:
    """Check model-based exact reconstruction and data-driven N^-1/2 order."""
    A, B, E, C = spec["A"], spec["B"], spec["E"], spec["C"]
    ell = 3
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal(A.shape[0])
    us = rng.standard_normal((ell, B.shape[1]))
    ds = rng.standard_normal((ell, E.shape[1]))
    ys, xcur = [], x0.copy()
    for i in range(ell):
        ys.append(C @ xcur)
        xcur = A @ xcur + B @ us[i] + E @ ds[i]
    ywin = np.concatenate(ys)
    z = core.reconstruct_delayed_state(A, B, C, E, ywin, us.reshape(-1),
                                       ds.reshape(-1), ell)
    x_true = xcur
    xhat = core.propagate_reconstructed_state(A, B, E, z, us, ds)
    known_err = float(np.linalg.norm(xhat - x_true))

    scan = ddr.reconstruction_rate_scan(
        spec, Ns=(1000, 2000, 4000, 8000), seeds=(100, 101, 102), ell=3)
    sl = scan["slopes"]
    # The statistical identities are expected to decay at the parametric
    # rate; a deliberately broad numerical acceptance window avoids pretending
    # three seeds identify an asymptotic constant.
    order_ok = all(-0.80 < sl[k] < -0.20
                   for k in ("reconstruction", "input_identity",
                             "disturbance_identity"))
    finite_ok = all(np.isfinite(v) for row in scan["per_n"].values()
                    for v in row.values())
    return [
        _check("exact model-based reconstruction", known_err < 1e-9,
               known_err, 1e-9),
        _check("data-driven N^-1/2 trends", order_ok and finite_ok,
               min(sl[k] for k in ("reconstruction", "input_identity",
                                    "disturbance_identity"))),
    ] + [
        {"name": f"data-driven slope {k}", "passed": True, "value": float(v)}
        for k, v in sl.items()
    ]


def verify_scaling_lmi_sdpi(spec: Dict) -> List[Dict]:
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    Phi = Q + opt["K"].T @ R @ opt["K"] - lam * opt["N"].T @ opt["N"]
    cert = thm3.construct_feasible_X(opt["G"], opt["P"], Phi, alpha)
    lmi = thm3.sdp_feasibility(opt["G"], opt["P"], Phi, alpha)
    lyap = thm3.lyapunov_sdp(opt["G"])
    good = [
        _check("X LMI first block positive", cert["eig_C1"] > 0,
               cert["eig_C1"], 0.0),
        _check("X LMI second block positive", cert["eig_C2"] > 0,
               cert["eig_C2"], 0.0),
        _check("direct X-SDP feasible", bool(lmi["feasible"]),
               float(lmi["eig_C2"] or 0.0), 0.0),
        _check("pure Lyapunov SDP feasible", bool(lyap.get("feasible")),
               0.0 if lyap.get("feasible") else 1.0, 0.0),
    ]

    # An intentionally destabilizing policy for the scaling/initializer path.
    Klqr, _, _, _ = core.solve_discounted_lqr(A, B, Q, R, alpha)
    bad = dict(K=-5.0 * Klqr, p=np.zeros(B.shape[1]),
               N=np.zeros((E.shape[1], A.shape[0])), z=np.zeros(E.shape[1]))
    G_bad = A + B @ bad["K"] + E @ bad["N"]
    stable, init = sdpi._initial_stabilizing_policy(
        A, B, E, Q, R, alpha, lam, spec["d_bar"].reshape(1, -1), bad,
        max_rounds=8)
    rho_bad = sdpi.rho(G_bad)
    rho_stable = sdpi.rho(A + B @ stable["K"] + E @ stable["N"])

    res = sdpi.solve_sdpi(A, B, E, Q, R, alpha, lam,
                          spec["d_bar"].reshape(1, -1),
                          K0=bad["K"], p0=bad["p"], N0=bad["N"], z0=bad["z"],
                          max_phase2=20, eps=1e-7)
    iter_rhos = [float(h["rho_G"]) for h in res["history"]]
    return good + [
        _check("initial destabilizing policy", rho_bad >= 1.0, rho_bad, 1.0),
        _check("scaled initializer returns stable original policy",
               rho_stable < 1.0, rho_stable, 1.0),
        _check("SDPI every logged iterate stable", max(iter_rhos) < 1.0,
               max(iter_rhos), 1.0),
        _check("SDPI final fixed-point residual small",
               max(res["fixed_point"][k] for k in
                   ("residual_P", "residual_chi", "residual_psi")) < 1e-6,
               max(res["fixed_point"][k] for k in
                   ("residual_P", "residual_chi", "residual_psi")), 1e-6),
        {"name": "scaling initializer source", "passed": True,
         "value": init["source"]},
    ]


def verify_data_driven_consistency(spec: Dict) -> List[Dict]:
    """Check the consistency theorem's implementation and report rate diagnostics.

    The paper deliberately does not make a uniform finite-sample rate claim.
    The slope values are therefore diagnostics, not theorem-pass/fail checks.
    """
    scan = a4.alg4_slope_scan(
        spec, Ns=(4000, 8000, 16000), seeds=(0, 1, 2), rounds=1,
        use_iv=True, n_lags=1, dist_probe_scale=0.5, expl_scale=1.0)
    sl = scan["slopes"]
    keys = ("eA", "eB", "eE", "eP", "eK", "ep")
    finite = all(np.isfinite(sl[k]) for k in keys)
    checks = [_check("data-driven slope diagnostics finite", finite,
                     max(abs(sl[k]) for k in keys))]
    checks += [
        {"name": f"diagnostic slope {k}", "passed": True,
         "value": float(sl[k])}
        for k in keys
    ]
    checks.append({"name": "scope", "passed": True,
                   "value": "consistency theorem; rates reported as diagnostics only"})
    return checks


def verify_rigorous_rate(spec: Dict) -> List[Dict]:
    """Validate the local rate under the explicit PE/conditioning assumptions."""
    r = rigorous_rate_validation.run(
        spec, Ns=(8000, 16000, 32000, 64000, 128000, 256000),
        seeds=(0, 1, 2, 3, 4), target_noise=1e-5)
    s = r["slopes"]
    keys = ("eP", "eK", "ep")
    finite = all(np.isfinite(s[k]) for k in keys)
    near_half = all(-0.85 < s[k] < -0.15 for k in keys)
    return [
        _check("local-rate diagnostics finite", finite,
               max(abs(s[k]) for k in keys)),
        _check("local-rate diagnostics consistent with N^-1/2 regime", near_half,
               min(s[k] for k in keys)),
    ] + [
        {"name": f"local-rate slope {k}", "passed": True, "value": float(s[k])}
        for k in keys
    ]


def main():
    report = {"suite": "paper theorem verification", "models": {},
              "all_passed": True}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        rec = {}
        rec["assumptions"] = verify_assumptions(spec)
        rec["wasserstein_bellman"] = verify_wasserstein_bellman(spec)
        rec["positive_P"] = verify_positive_P(spec)
        rec["reconstruction_bound"] = verify_reconstruction_bound(spec)
        rec["scaling_lmi_sdpi"] = verify_scaling_lmi_sdpi(spec)
        rec["data_driven_consistency"] = verify_data_driven_consistency(spec)
        rec["rigorous_rate_validation"] = verify_rigorous_rate(spec)
        report["models"][name] = rec
        print(f"[PASS] {name}: theorem-level numerical checks")

    with open(os.path.join(LOG, "paper_theory_verification.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    lines = ["# Paper theorem verification", "",
             "All numerical checks below use the two application models in the"
             " manuscript.  They verify algebraic identities, stability"
             " certificates, and finite-sample rates; they do not turn a finite"
             " sample into a global infinite-horizon proof.", ""]
    for name, rec in report["models"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| Check | Result | Value |")
        lines.append("|---|---:|---:|")
        for group, checks in rec.items():
            for c in checks:
                val = c.get("value", "")
                lines.append(f"| {group}: {c['name']} | "
                             f"{'PASS' if c['passed'] else 'FAIL'} | {val} |")
        lines.append("")
    with open(os.path.join(OUT, "PAPER_THEORY_VERIFICATION.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("ALL PAPER THEOREM VERIFICATIONS PASSED")
    print(os.path.join(OUT, "PAPER_THEORY_VERIFICATION.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
