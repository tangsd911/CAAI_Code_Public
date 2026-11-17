#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the generated CAAI simulation artifacts against their scope.

This is a structural and numeric sanity check, not a substitute for the
theoretical tests.  It catches missing tables, truncated Monte-Carlo runs,
unexpected NaN values, infeasible hard-ball rows, and accidentally unstable
primary experiments.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")


def rows(name):
    path = os.path.join(TBL, name + ".csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value):
    return float(str(value).replace("±", ",").split(",")[0])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    checks = []

    e1 = rows("od_e1_main_output_feedback")
    require(len(e1) == 8, f"od_e1 rows={len(e1)}")
    ofb = [r for r in e1 if "unknown-history" in r["Feedback"]]
    require(len(ofb) == 4, "od_e1 missing unknown-history OFB rows")
    require(all(int(r["Diverged seeds"]) == 0 for r in ofb), "OFB divergence")
    require(all(math.isfinite(num(r["MAE"])) and math.isfinite(num(r["Discounted cost"]))
                for r in ofb), "nonfinite OFB metric")
    checks.append("output feedback: 4/4 unknown-history rows finite, 0 divergences")

    e2 = rows("od_e2_output_feedback_shift")
    require(len(e2) == 16, f"od_e2 rows={len(e2)}")
    require(all(int(r["Diverged seeds"]) == 0 for r in e2), "shift divergence")
    checks.append("OFB shift: 16 rows, 0 divergences")

    e3 = rows("od_e3_output_feedback_ablation")
    require(len(e3) == 8, f"od_e3 rows={len(e3)}")
    require(all(int(r["Diverged seeds"]) == 0 for r in e3), "ablation divergence")
    checks.append("OFB ablation: 8 rows, 0 divergences")

    e4 = rows("od_e4_algorithm4_end_to_end")
    require(len(e4) == 4, f"od_e4 rows={len(e4)}")
    iv = [r for r in e4 if r["Method"] == "IV/2SLS"]
    require(len(iv) == 2, "missing IV rows")
    require(all(r["E identifiable"] == "5/5" for r in iv), "E not identifiable in IV")
    require(all(r["N identifiable"] == "5/5" for r in iv), "N not identifiable in IV")
    require(all(r["post-hoc stable"] == "5/5" for r in iv), "IV policy not post-hoc stable")
    checks.append("Algorithm 4: IV E/N identifiable and post-hoc stable in 5/5 seeds")

    e5 = rows("od_e5_algorithm4_rates")
    require(len(e5) == 4, f"od_e5 rows={len(e5)}")
    checks.append("Algorithm 4 rates: four OLS/IV rows present")

    e6 = rows("od_e6_hard_wasserstein_own_adversary")
    require(len(e6) == 24, f"od_e6 rows={len(e6)}")
    require(all(r["status"] == "feasible" for r in e6), "hard-ball infeasible row")
    require(all(num(r["W2 max"]) <= num(r["zeta"]) + 1e-10 for r in e6),
            "W2 max exceeds target")
    checks.append("hard W2 scan: 24/24 ball-feasible, W2_max <= zeta")

    e7 = rows("od_e7_g_w2_membership")
    e7d = rows("od_e7_g_w2_support_draws")
    require(len(e7) == 20, f"od_e7 summary rows={len(e7)}")
    require(len(e7d) == 200, f"od_e7 draw rows={len(e7d)}")
    require(all(r["status"] == "feasible" for r in e7), "g-summary infeasible")
    require(all(r["status"] == "feasible" for r in e7d), "g-draw infeasible")
    require(all(num(r["W2 max"]) <= num(r["zeta"]) + 1e-10 for r in e7d),
            "g-draw W2 exceeds target")
    checks.append("general-g W2: 200/200 support draws ball-feasible")

    e8 = rows("od_e8_nonlinear_contact_saturation")
    require(len(e8) == 12, f"od_e8 rows={len(e8)}")
    require(all(r["Seeds"] == "20" and r["Valid seeds"] == "20" for r in e8),
            "nonlinear seed count")
    require(all(r["Diverged seeds"] == "0" for r in e8), "nonlinear divergence")
    checks.append("nonlinear stress: 12 rows x 20 seeds, 0 divergence")

    e9 = rows("od_e9_tail_only")
    require(len(e9) == 20, f"od_e9 rows={len(e9)}")
    require(all(r["Seeds"] == "20" and r["Valid seeds"] == "20" for r in e9),
            "tail seed count")
    require(all(r["Diverged seeds"] == "0" for r in e9), "tail divergence")
    checks.append("tail stress: 20 rows x 20 seeds, 0 divergence")

    for name in ("od_e5c_one_step_consistency",
                 "od_e5d_reconstruction_bound_audit",
                 "od_e5e_one_step_condition_scan",
                 "od_e5f_one_step_orthogonal_consistency",
                 "od_e5g_rigorous_rate_validation",
                 "e18_covariance_only", "e19_heavy_tail",
                 "e20_g_sample_size", "e21_saturation"):
        rr = rows(name)
        require(bool(rr), f"empty {name}")
        checks.append(f"{name}: {len(rr)} rows")

    for fig in ("sim_humanoid_mimo_ofb.pdf", "sim_microgrid_mimo_ofb.pdf"):
        require(os.path.getsize(os.path.join(OUT, "figures", fig)) > 0,
                f"missing figure {fig}")
    checks.append("figures: both OFB PDFs present")

    e22 = rows("e22_nonzero_constant_reference")
    require(len(e22) == 2, f"e22 rows={len(e22)}")
    require(all(abs(float(r["bias residual"])) < 1e-10 for r in e22),
            "nonzero-reference bias residual")
    require(float(e22[1]["deterministic steady error norm"]) < 1e-6,
            "nonzero-reference deterministic error")
    checks.append("nonzero constant reference: bias residual zero and affine error suppressed")
    e23 = rows("e23_initial_gain_transition_stages")
    require(len(e23) >= 2, f"e23 stages={len(e23)}")
    require(float(e23[0]["rho_G"]) > 1.0, "initial gain is not unstable")
    require(all(float(r["rho_G"]) < 1.0 for r in e23[1:]), "SDPI transition unstable")
    require(os.path.getsize(os.path.join(OUT, "figures", "sim_initial_gain_transition.pdf")) > 0,
            "missing initial-gain transition figure")
    checks.append("initial-gain transition: unstable start, stabilizing iterates, figure present")

    report = dict(ok=True, checks=checks)
    with open(os.path.join(OUT, "OUTPUT_VALIDATION.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT, "OUTPUT_VALIDATION.md"), "w",
              encoding="utf-8") as f:
        f.write("# Simulation output validation\n\n")
        for c in checks:
            f.write(f"- PASS: {c}\n")
    print("ALL OUTPUT VALIDATIONS PASSED")
    for c in checks:
        print(" [OK]", c)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("OUTPUT VALIDATION FAILED:", exc, file=sys.stderr)
        raise
