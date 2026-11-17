#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every simulation requested in the CAAI revision audit.

Default mode is the complete, reproducible suite.  It deliberately separates
the numerical experiments into five groups: output feedback, Algorithm 4 /
SDPI, hard Wasserstein membership, nonlinear / tail stress tests, and the
legacy model-based regression suite.  Every group writes its own CSV/Markdown
table and log; a failure stops the suite and is preserved in
``output/FINAL_SUITE_SUMMARY.json``.

Usage:
    python3 run_complete_suite.py
    python3 run_complete_suite.py --quick   # debugging only, reduced seeds
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
import subprocess


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LOG = os.path.join(OUT, "logs")
_MPL = os.path.join(HERE, ".mplconfig")
os.makedirs(_MPL, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL)


def _jsonable(obj):
    """Convert common scientific-Python values to JSON-safe objects."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _compact(value):
    """Keep the summary small while retaining pass/fail information."""
    if isinstance(value, dict):
        keys = {"feasible_rate", "status", "accepted", "stop_reason",
                "converged", "n_seeds", "valid_seeds", "diverged_seeds",
                "wall_seconds", "passed", "result"}
        return {str(k): _jsonable(v) for k, v in value.items() if k in keys}
    if isinstance(value, (list, tuple)):
        return {"rows": len(value)}
    return _jsonable(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="run a reduced debugging scan; not for paper results")
    parser.add_argument("--skip-regression", action="store_true",
                        help="skip the existing model-based regression suite")
    args = parser.parse_args()

    for d in (OUT, LOG, os.path.join(OUT, "tables"),
              os.path.join(OUT, "figures")):
        os.makedirs(d, exist_ok=True)

    import agent_od_experiments as od
    import experiments_w2 as w2
    import experiments_nl as nl
    import final_alg4_experiments as a4
    import figures_ofb
    import constant_reference
    import initial_gain_transition
    import one_step_condition_scan
    import one_step_consistency
    import one_step_orthogonal_consistency
    import paper_theory_verification
    import reconstruction_bound_audit
    import rigorous_rate_validation
    import scan_hardcoded
    import validate_outputs

    def run_script(script):
        cp = subprocess.run([sys.executable, script], cwd=HERE,
                            check=True, text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
        print(cp.stdout)
        return {"script": script, "exit_code": 0,
                "output_tail": cp.stdout[-2000:]}

    # The two specialized modules intentionally expose the same function
    # names as the base orchestrator; using them explicitly prevents an old
    # implementation from silently replacing a newer experiment.
    steps = [
        ("output_feedback_main", od.exp_ofb_main),
        ("output_feedback_shift", od.exp_ofb_shift),
        ("output_feedback_ablation", od.exp_ofb_ablation),
        ("algorithm4_end_to_end", a4.run_end_to_end),
        ("algorithm4_rates", a4.run_rates),
        ("algorithm4_adversary_rates", a4.run_adversary_rates),
        ("algorithm4_sdpi_diagnostics", a4.run_sdpi_diagnostics),
        ("algorithm4_identifiability_negative", a4.run_identifiability_negative),
        ("hard_wasserstein_own_adversary", w2.exp_hard_wasserstein),
        ("g_w2_membership", w2.exp_g_w2_membership),
        ("nonlinear_contact_saturation", nl.exp_nonlinear_validation),
        ("tail_only_heavy_tail", nl.exp_tail_only_validation),
        ("covariance_heavy_tail_saturation", od.exp_robustness_suite),
        ("figures_output_feedback", figures_ofb.make_all),
        ("nonzero_constant_reference", constant_reference.run),
        ("initial_gain_transition", initial_gain_transition.run),
        ("paper_theory_verification", paper_theory_verification.main),
        ("one_step_consistency", one_step_consistency.main),
        ("one_step_condition_scan", one_step_condition_scan.main),
        ("one_step_orthogonal_consistency", one_step_orthogonal_consistency.main),
        ("rigorous_rate_validation", rigorous_rate_validation.main),
        ("reconstruction_bound_audit", reconstruction_bound_audit.main),
    ]
    if not args.skip_regression:
        steps.extend([
            ("theory_tests", lambda: run_script("theory_tests.py")),
            ("algorithm4_tests", lambda: run_script("algorithm4_tests.py")),
        ])

    # Quick mode is a fast installation/compile check only.  It must never be
    # used to produce paper tables or to claim that the simulation suite ran.
    if args.quick:
        steps = [
            ("theory_tests", lambda: run_script("theory_tests.py")),
            ("algorithm4_tests", lambda: run_script("algorithm4_tests.py")),
            ("hardcode_scan", lambda: scan_hardcoded.main()),
        ]

    summary = {
        "suite": "CAAI complete simulation audit",
        "quick": bool(args.quick),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {},
    }
    t0 = time.time()
    for name, fn in steps:
        print(f"\n########## {name} ##########", flush=True)
        buf = io.StringIO()
        step_start = time.time()
        try:
            # Each step writes its own detailed log.  Keep stdout in the
            # console and in a captured summary.
            with contextlib.redirect_stdout(buf):
                result = fn()
            elapsed = time.time() - step_start
            text = buf.getvalue()
            with open(os.path.join(LOG, name + ".log"), "w", encoding="utf-8") as f:
                f.write(text)
            summary["steps"][name] = {
                "status": "ok",
                "wall_seconds": elapsed,
                "stdout_tail": text[-2000:],
                "result": _compact(result),
            }
            print(f"[OK] {name} ({elapsed:.1f}s)", flush=True)
        except Exception as exc:
            elapsed = time.time() - step_start
            summary["steps"][name] = {
                "status": "failed",
                "wall_seconds": elapsed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": buf.getvalue()[-5000:],
            }
            print(f"[FAILED] {name}: {type(exc).__name__}: {exc}", file=sys.stderr,
                  flush=True)
            break

    if not args.skip_regression and not args.quick:
        for extra_name, extra_fn in (
                ("hardcode_scan", scan_hardcoded.main),
                ("output_validation", validate_outputs.main)):
            try:
                extra_fn()
                summary["steps"][extra_name] = {"status": "ok"}
            except Exception as exc:
                summary["steps"][extra_name] = {
                    "status": "failed", "error": str(exc),
                    "traceback": traceback.format_exc()}

    summary["wall_seconds"] = time.time() - t0
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    expected_steps = len(steps) + (0 if (args.skip_regression or args.quick) else 2)
    summary["all_ok"] = (all(v.get("status") == "ok"
                             for v in summary["steps"].values())
                         and len(summary["steps"]) == expected_steps)
    with open(os.path.join(OUT, "FINAL_SUITE_SUMMARY.json"), "w",
              encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, allow_nan=False)

    # Human-readable audit report.  It intentionally lists scope limitations
    # next to the generated tables so a passing numerical test is not confused
    # with a global theoretical proof.
    lines = [
        "# CAAI complete simulation audit",
        "",
        f"Generated: {summary['finished_at']}",
        f"Quick mode: `{summary['quick']}`",
        f"All steps OK: `{summary['all_ok']}`",
        "",
        "## Generated artifacts",
        "",
    ]
    for name, rec in summary["steps"].items():
        lines.append(f"- `{name}`: **{rec['status']}** ({rec.get('wall_seconds', 0):.1f}s)")
    lines += [
        "",
        "## Tables",
        "",
        "The authoritative tables are in `output/tables/`; JSON manifests and",
        "stdout logs are in `output/logs/`.  Key files:",
        "",
        "- `od_e1_main_output_feedback.md`",
        "- `od_e2_output_feedback_shift.md`",
        "- `od_e3_output_feedback_ablation.md`",
        "- `od_e4_algorithm4_end_to_end.md`",
        "- `od_e5_algorithm4_rates.md`",
        "- `od_e5b_adversary_block_rates.md`",
        "- `od_e9_sdpi_diagnostics.md`",
        "- `od_e10_identifiability_negative.md`",
        "- `od_e6_hard_wasserstein_own_adversary.md`",
        "- `od_e7_g_w2_membership.md`",
        "- `od_e8_nonlinear_contact_saturation.md`",
        "- `od_e9_tail_only.md`",
        "- `e18_covariance_only.md`, `e19_heavy_tail.md`, `e20_g_sample_size.md`, `e21_saturation.md`",
        "",
        "## Scope guards",
        "",
        "- The output-feedback rows using a Kalman filter are unknown-disturbance-history",
        "  simulations.  They are not claimed to reproduce full-state performance.",
        "- Algorithm 4 requires an independent disturbance-channel probe for separate",
        "  identification of E and N; without it the code returns an explicit",
        "  non-identifiability result.",
        "- Hard-Wasserstein results are finite-dimensional, sampled trajectory",
        "  certificates.  They are not a proof of infinite-horizon global Bellman",
        "  optimality.",
        "- Nonlinear/contact/tail experiments are reduced stress models, not",
        "  high-fidelity hardware validations.",
    ]
    with open(os.path.join(OUT, "FINAL_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"all_ok": summary["all_ok"],
                      "wall_seconds": summary["wall_seconds"],
                      "summary": os.path.join(OUT, "FINAL_SUITE_SUMMARY.json")},
                     indent=2))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
