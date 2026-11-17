#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct one-step test of the data-driven consistency theorem.

The manuscript's theorem is a *fixed-policy batch* statement: estimate the
model, evaluate the current policy once, and perform one improvement.  It is
not the multi-iteration SDPI trajectory used by the end-to-end Algorithm 4
table.  This script isolates the theorem's actual data interface.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Sequence

import numpy as np

import data_driven as dd
import drc_core as core
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TBL = os.path.join(OUT, "tables")
LOG = os.path.join(OUT, "logs")
for _d in (OUT, TBL, LOG):
    os.makedirs(_d, exist_ok=True)


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


def run_model(spec: Dict, Ns: Sequence[int] = (4000, 8000, 16000, 32000),
              seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict:
    A, B, E = spec["A"], spec["B"], spec["E"]
    Q, R, alpha, lam = spec["Q"], spec["R"], spec["alpha"], spec["lam"]
    opt = core.solve_drc_pi(A, B, E, Q, R, alpha, lam, spec["d_bar"])
    V = spec["d_bar"].reshape(1, -1)
    per_n = {}
    for N in Ns:
        values = []
        for seed in seeds:
            data = dd.collect_closed_loop(
                spec, opt["K"], opt["p"], opt["N"], opt["z"], N,
                seed=seed, expl_scale=1.0,
                state_obs_noise_std=1e-4, input_obs_noise_std=1e-5,
                dist_obs_noise_std=1e-5, target_obs_noise_std=1e-4,
                dist_probe_std=0.5 * spec["d_std"])
            est = dd.identify_full_iv(data, n_lags=1,
                                      instrument_mode="probe")
            # Fixed policy batch: evaluate the known current policy once.
            pe = core.policy_evaluate(
                est["A_hat"], est["B_hat"], est["E_hat"], Q, R, alpha, lam,
                V, opt["K"], opt["p"], opt["N"], opt["z"])
            # One improvement, not repeated SDPI iterations.
            imp = core.policy_improve(
                est["A_hat"], est["B_hat"], est["E_hat"], Q, R, alpha, lam,
                V, pe["P"], pe["chi"])
            values.append(dict(
                eA=float(np.linalg.norm(est["A_hat"] - A) / np.linalg.norm(A)),
                eB=float(np.linalg.norm(est["B_hat"] - B) / np.linalg.norm(B)),
                eE=float(np.linalg.norm(est["E_hat"] - E) / np.linalg.norm(E)),
                eP=float(np.linalg.norm(pe["P"] - opt["P"]) /
                         np.linalg.norm(opt["P"])),
                eK=float(np.linalg.norm(imp["K"] - opt["K"], "fro")),
                ep=float(np.linalg.norm(imp["p"] - opt["p"])),
            ))
        per_n[int(N)] = dict(
            mean={k: float(np.mean([v[k] for v in values])) for k in values[0]},
            std={k: float(np.std([v[k] for v in values])) for k in values[0]},
        )
    slopes = {}
    for key in ("eA", "eB", "eE", "eP", "eK", "ep"):
        slopes[key] = dd.loglog_slope(
            list(per_n), [per_n[N]["mean"][key] for N in per_n])
    return dict(per_n=per_n, slopes=slopes, opt=opt,
                Ns=list(Ns), seeds=list(seeds))


def main():
    rows, point_rows, report = [], [], {}
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        r = run_model(spec)
        report[name] = r
        s = r["slopes"]
        rows.append([name, "IV/2SLS", f"{s['eA']:.4f}",
                     f"{s['eB']:.4f}", f"{s['eE']:.4f}",
                     f"{s['eP']:.4f}", f"{s['eK']:.4f}",
                     f"{s['ep']:.4f}"])
        for N, v in r["per_n"].items():
            point_rows.append([name, N,
                               f"{v['mean']['eA']:.6e}",
                               f"{v['mean']['eB']:.6e}",
                               f"{v['mean']['eE']:.6e}",
                               f"{v['mean']['eP']:.6e}",
                               f"{v['mean']['eK']:.6e}",
                               f"{v['mean']['ep']:.6e}"])
    _write_table(
        "od_e5c_one_step_consistency",
        ["Scenario", "Method", "slope eA", "slope eB", "slope eE",
         "slope eP", "slope eK", "slope ep"], rows,
        notes=("Direct fixed-policy batch: one model identification, one exact "
               "policy evaluation, and one improvement. This isolates the "
               "manuscript's local consistency theorem. It must not be "
               "confused with the multi-iteration SDPI table od_e5."))
    _write_table(
        "od_e5c_one_step_consistency_points",
        ["Scenario", "N", "mean eA", "mean eB", "mean eE",
         "mean eP", "mean eK", "mean ep"], point_rows,
        notes="Point estimates underlying od_e5c_one_step_consistency.")
    with open(os.path.join(LOG, "od_e5c_one_step_consistency.json"), "w",
              encoding="utf-8") as f:
        json.dump({k: {"slopes": v["slopes"], "per_n": v["per_n"]}
                   for k, v in report.items()}, f, indent=2, default=float)
    print("one-step consistency slopes:")
    for name, r in report.items():
        print(name, r["slopes"])
    return report


if __name__ == "__main__":
    main()
