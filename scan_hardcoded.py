#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-coded result-literal scanner.

Scans every .py in this package and reports decimal literals that appear in
result-emitting or plotting statements (print / bar / text / annotate / plot /
axis labels / titles / limits / legends).

Classification:
  A. Suspected RESULT literals: decimals that are not in a layout/formatting
     context. These should be empty: every reported number must come from a
     solver or Monte-Carlo call.
  B. Layout / formatting constants: figsize, fontsize, linewidth, alpha,
     limits, dpi, bar width, seeds, tolerances, format specs, etc. These are
     allowed.
  C. Reference constants in models.py: model parameters and paper-reported
     comparison values copied from 00_paper_claims.md. They are never fed back
     into a simulation.

Usage:  python3 scan_hardcoded.py
Writes output/hardcode_scan.txt and prints the summary to stdout.
"""

import os
import re
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# statements that emit a result or draw a figure element
STMT = re.compile(
    r"\b(print|\.bar\(|\.barh\(|\.text\(|\.annotate\(|\.plot\(|\.axhline\("
    r"|set_title|set_ylabel|set_xlabel|set_xticks|set_yticks|set_ylim|"
    r"set_xlim|\.legend\(|figsize|subplots\(|\.grid\(|suptitle|"
    r"tight_layout|savefig)")
NUM = re.compile(r"(?<![\w:])([-+]?\d*\.\d+)")
FMT_SPEC = re.compile(r":\.\d+f?")

# layout / formatting contexts whose decimals are not result constants
LAYOUT_TOKENS = (
    "figsize", "fontsize", "linewidth", "alpha=", "ylim", "xlim", "dpi",
    "width", "w=", "seed", "steps", "tol", "max_iter", "bbox", "rect",
    "pad", "markersize", "grid", "ls=", "color", "legend", "suptitle",
    "tight_layout", "savefig", "arange", "mean", "std", "sqrt", "diag",
    "eye(", "zeros", "pinv", "eig", "svd", "matrix_rank", "matrix_power",
    "separator", "format", "round", "np.", "label", "title", "ylabel",
    "xlabel", "font", "rotation", "ncol", "loc", "marker", "linestyle",
)


def classify(line):
    for tok in LAYOUT_TOKENS:
        if tok in line:
            return "layout"
    return "result"


def main():
    py_files = sorted(glob.glob(os.path.join(HERE, "*.py")))
    suspected, layout = [], []

    for path in py_files:
        rel = os.path.basename(path)
        if rel == "scan_hardcoded.py":
            continue
        with open(path, encoding="utf-8") as f:
            for ln, raw in enumerate(f, 1):
                # strip inline / full-line comments before matching
                code = raw.split("#", 1)[0].rstrip()
                if not code:
                    continue
                if not STMT.search(code):
                    continue
                cleaned = FMT_SPEC.sub("", code)
                for m in NUM.finditer(cleaned):
                    lit = m.group(1)
                    entry = (rel, ln, lit, code.strip())
                    if classify(code) == "layout":
                        layout.append(entry)
                    else:
                        suspected.append(entry)

    # models.py reference constants (parameters + paper comparison values)
    reference = []
    with open(os.path.join(HERE, "models.py"), encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            if re.search(r"[-+]?\d*\.\d+", raw):
                reference.append((ln, raw.strip()))

    out = []
    out.append("Hard-coded result-literal scan")
    out.append("=" * 40)
    out.append(f"Python files scanned: {len(py_files)}")
    out.append("")
    out.append(f"A. Suspected result literals in print/bar/text/plot: "
               f"{len(suspected)}")
    for rel, ln, lit, line in suspected:
        out.append(f"  {rel}:{ln}  {lit}  ::  {line}")
    if not suspected:
        out.append("  (none)")
    out.append("")
    out.append(f"B. Layout / formatting constants (allowed): {len(layout)}")
    for rel, ln, lit, line in layout:
        out.append(f"  {rel}:{ln}  {lit}")
    out.append("")
    out.append("C. models.py constants (model parameters + paper comparison "
               "values; never fed to a simulation):")
    for ln, line in reference:
        out.append(f"  models.py:{ln}  {line}")

    report = "\n".join(out)
    with open(os.path.join(OUT, "hardcode_scan.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nResult-literal count (section A): {len(suspected)}")
    print(f"Layout-constant count (section B): {len(layout)}")
    return len(suspected), len(layout)


if __name__ == "__main__":
    main()
