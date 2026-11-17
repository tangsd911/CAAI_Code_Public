#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empirical audit of the constants in the reconstruction error theorem.

The theorem states an upper bound with unspecified positive constants.  This
script does not pretend to prove those constants; it reports a reproducible
finite-sample envelope:

    err <= C1/sqrt(N) + ||T_d|| * ||d_window - vbar_window||.

The first term bounds the identification error e_{N,k}; the second term is
kept explicitly because the squared-loss minimax center is the empirical mean
and a Wasserstein radius does not translate that center.  Constants are
calibrated on independent training seeds and checked on held-out seeds.  The
reported ratio is finite-sample evidence, not a proof of a universal constant.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Sequence

import numpy as np

import data_driven_reconstruction as ddr
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")
LOG = os.path.join(OUT, "logs")
for _d in (OUT, TBL, LOG):
    os.makedirs(_d, exist_ok=True)


def _sp(M):
    return float(np.linalg.norm(M, 2))


def _one_point(spec, N, zeta, seed):
    train = ddr.simulate_windows(spec, N, 3, seed)
    fit = ddr.fit_reconstruction(train, spec)
    test = ddr.simulate_windows(spec, 300, 3, seed + 100000)
    support = spec["d_bar"].reshape(1, -1)
    recz = ddr.reconstruct(test, fit, zeta=float(zeta), support=support)

    # Equation (23) is a per-window inequality.  Evaluate its two terms on the
    # same held-out windows used by the error audit.
    vbar = np.asarray(support, dtype=float).mean(axis=0)
    vbar_window = np.tile(vbar, test.ell)
    xhat_eq = fit["M_u"] @ test.U + fit["M_y"] @ test.Y
    xref_eq = fit["T_inv"] @ test.X
    eN_windows = xhat_eq - xref_eq - fit["T_d"] @ test.D
    residual_windows = fit["T_d"] @ (test.D - vbar_window[:, None])
    eN_max = float(np.max(np.linalg.norm(eN_windows, axis=0)))
    residual_max = float(np.max(np.linalg.norm(residual_windows, axis=0)))
    err_total = float(np.max(recz["err_robust"]))
    return dict(N=int(N), zeta=float(zeta), err_stat=err_total,
                err_total=err_total, eN=eN_max,
                residual=residual_max, td_norm=_sp(fit["T_d"]))


def _fit_and_test(spec, train_seeds, test_seeds,
                  Ns=(1000, 2000, 4000, 8000),
                  zetas=(0.0, 0.01, 0.02, 0.05, 0.1)) -> Dict:
    calib, held = [], []
    for seed in train_seeds:
        for N in Ns:
            for z in zetas[1:]:
                calib.append(_one_point(spec, N, z, seed))
    for seed in test_seeds:
        for N in Ns:
            for z in zetas[1:]:
                held.append(_one_point(spec, N, z, seed))

    # Conservative calibration of the identification-error constant.  The
    # residual term in (23) is not calibrated: its coefficient is fixed to one
    # by the triangle inequality and it is evaluated explicitly below.
    C1 = max(r["eN"] * np.sqrt(r["N"]) for r in calib)

    rows = []
    for r in held:
        pred = C1 / np.sqrt(r["N"]) + r["residual"]
        ratio = r["err_total"] / max(pred, 1e-15)
        rows.append(dict(r, prediction=pred, ratio=ratio,
                         violation=bool(r["err_total"] > pred + 1e-12)))
    return dict(calib=calib, held=rows,
                constants=dict(C1=float(C1), C_residual=1.0),
                max_test_ratio=float(max(r["ratio"] for r in rows)),
                violation_fraction=float(np.mean([r["violation"] for r in rows])),
                max_test_overrun=float(max(r["err_total"] - r["prediction"]
                                           for r in rows)))


def _write_table(name, headers, rows, notes=None):
    with open(os.path.join(TBL, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    with open(os.path.join(TBL, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
        if notes:
            f.write("\n" + notes + "\n")


def main():
    report = {}
    rows = []
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        r = _fit_and_test(spec, (100, 101, 102), (200, 201, 202))
        report[name] = r
        c = r["constants"]
        rows.append([name, f"{c['C1']:.6e}",
                     f"{c['C_residual']:.6e}", f"{r['max_test_ratio']:.4f}",
                     f"{r['violation_fraction']:.4f}",
                     f"{r['max_test_overrun']:.6e}",
                     len(r["calib"]), len(r["held"])])
    _write_table(
        "od_e5d_reconstruction_bound_audit",
        ["Scenario", "empirical C1", "residual coefficient",
         "max test ratio", "violation fraction", "max test overrun",
         "calibration points", "held-out points"], rows,
        notes=("The audited inequality is err <= C1/sqrt(N) + "
               "||T_d||*||d_window-vbar_window||. The residual coefficient is "
               "one by the triangle inequality and is not tuned. C1 is "
               "calibrated on seeds 100-102 and tested on seeds 200-202; this "
               "is finite-sample evidence, not a proof of a universal constant."))
    with open(os.path.join(LOG, "od_e5d_reconstruction_bound_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print("reconstruction bound audit:")
    for name, r in report.items():
        print(name, r["constants"], "max_ratio",
              r["max_test_ratio"], "violations", r["violation_fraction"])
    return report


if __name__ == "__main__":
    main()
