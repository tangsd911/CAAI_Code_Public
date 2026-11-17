#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression tests for Algorithm 4, IV, E/N and SDPI."""

import numpy as np

import algorithm4 as a4
import data_driven as dd
import drc_core as core
import models
import sdpi


def exact_iv_test(name, spec):
    s = dict(spec)
    s["d_std"] = np.zeros_like(spec["d_std"])
    opt = core.solve_drc_pi(s["A"], s["B"], s["E"], s["Q"], s["R"],
                            s["alpha"], s["lam"], s["d_bar"])
    data = dd.collect_closed_loop(
        s, opt["K"], opt["p"], opt["N"], opt["z"], 4000, seed=0,
        expl_scale=1.0, dist_probe_std=0.5 * s["d_std"] + 1e-2)
    est = dd.identify_full_iv(data, n_lags=1, instrument_mode="probe")
    adv = dd.identify_adversary_block(data, use_iv=True)
    err = dict(
        A=float(np.linalg.norm(est["A_hat"] - s["A"])),
        B=float(np.linalg.norm(est["B_hat"] - s["B"])),
        E=float(np.linalg.norm(est["E_hat"] - s["E"])),
        N=float(np.linalg.norm(adv["N_hat"] - opt["N"])),
        z=float(np.linalg.norm(adv["z_hat"] - opt["z"])))
    assert max(err.values()) < 1e-8, (name, err)
    print(f"[OK] exact IV {name}: {err}")


def identifiability_negative_test(name, spec):
    r = a4.identifiability_negative_result(spec, N=2000, seed=0)
    assert r["certificate"]["E_status"] != "identifiable", (name, r)
    assert r["rank"] == r["reg_dim"]
    print(f"[OK] no-probe negative result {name}: {r['certificate']['E_status']}")


def scaling_map_test(name, spec):
    nx, nu, nd = spec["A"].shape[0], spec["B"].shape[1], spec["E"].shape[1]
    pol = dict(K=10.0 * np.ones((nu, nx)), p=np.zeros(nu),
               N=np.zeros((nd, nx)), z=np.zeros(nd))
    V = spec["d_bar"].reshape(1, -1)
    mapped, info = sdpi._initial_stabilizing_policy(
        spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
        spec["alpha"], spec["lam"], V, pol)
    rho = sdpi.rho(spec["A"] + spec["B"] @ mapped["K"] +
                   spec["E"] @ mapped["N"])
    assert rho < 1.0
    assert info["mapping_identity_residual"] < 1e-12
    print(f"[OK] scaling map {name}: rho={rho:.6f}, source={info['source']}")


def schur_boundary_test(name, spec):
    opt = core.solve_drc_pi(spec["A"], spec["B"], spec["E"], spec["Q"],
                            spec["R"], spec["alpha"], spec["lam"],
                            spec["d_bar"])
    r = sdpi.solve_sdpi(
        spec["A"], spec["B"], spec["E"], spec["Q"], spec["R"],
        spec["alpha"], 0.01, spec["d_bar"].reshape(1, -1),
        K0=opt["K"], p0=opt["p"], N0=opt["N"], z0=opt["z"],
        max_phase2=5, n_value_samples=10)
    assert r["stop_reason"] == "schur_complement_boundary", (name, r["stop_reason"])
    assert r["final_schur"]["boundary"] is True
    print(f"[OK] Schur boundary stop {name}: {r['stop_reason']}")


def end_to_end_smoke_test(name, spec):
    r = a4.run_data_driven_algorithm4(
        spec, N=3000, seed=0, rounds=1, use_iv=True,
        dist_probe_scale=0.5, expl_scale=1.0)
    assert r["certificate"]["E_status"] == "identifiable"
    assert r["accepted"] and r["true_rho_G"] < 1.0
    assert r["sdpi"]["stop_reason"] in ("stationarity", "interpolation_step")
    assert max(r["sdpi"]["final_delta"].values()) < 1e-7
    print(f"[OK] Algorithm 4 smoke {name}: rho={r['true_rho_G']:.6f}")


if __name__ == "__main__":
    for name, spec in (("humanoid", models.humanoid_spec()),
                       ("microgrid", models.microgrid_spec())):
        exact_iv_test(name, spec)
        identifiability_negative_test(name, spec)
        scaling_map_test(name, spec)
        schur_boundary_test(name, spec)
        end_to_end_smoke_test(name, spec)
    print("ALL ALGORITHM-4 TESTS PASSED")
