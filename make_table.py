"""
make_table.py — LaTeX results table for ECE 757A paper.

Each cell shows Return / PE (path efficiency, italic). Final active_level used
for level column. Bold = best return per column.

Output: figures/results_table.tex
"""

from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TB_ROOT = Path("runs/gymnasium_env")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SEED   = 9
LAST_N = 3

METHODS = ["allo-8", "allo-16", "critgn", "lev-sel"]
METHOD_LABELS = {
    "allo-8":  r"\textsc{Allo-8}",
    "allo-16": r"\textsc{Allo-16}",
    "critgn":  r"\textsc{CritGN}",
    "lev-sel": r"\textsc{LevSel}",
}

# Columns: (env, budget_str, display_label)
COLUMNS = [
    ("4lv",      "b1M", r"4-level"),
    ("5lv",      "b1M", r"5-level"),
    ("skip",     "b1M", r"Skip"),
    ("skip-dyn", "b1M", r"Skip-dyn"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tb_dir(exp_name: str) -> Path | None:
    matches = sorted(TB_ROOT.glob(f"*__{exp_name}__*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def final_scalar(tb_dir: Path, tag: str) -> float | None:
    ea = EventAccumulator(str(tb_dir), size_guidance={tag: 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    vals = [e.value for e in events[-LAST_N:]]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Collect: ret[method][ci], pe[method][ci], lv[method][ci]
# ---------------------------------------------------------------------------
ret_data: dict[str, list] = {m: [] for m in METHODS}
pe_data:  dict[str, list] = {m: [] for m in METHODS}
lv_data:  dict[str, list] = {m: [] for m in METHODS}

for method in METHODS:
    for env, budget, _ in COLUMNS:
        exp    = f"{method}-{env}-{budget}-s{SEED}"
        tb_dir = find_tb_dir(exp)
        if tb_dir is None:
            print(f"  [warn] missing: {exp}")
            ret_data[method].append(None)
            pe_data[method].append(None)
            lv_data[method].append(None)
        else:
            ret_data[method].append(final_scalar(tb_dir, "charts/rollout_mean_return"))
            pe_data[method].append(final_scalar(tb_dir, "charts/path_efficiency"))
            lv_data[method].append(final_scalar(tb_dir, "curriculum/active_level"))

# ---------------------------------------------------------------------------
# Best return per column for bolding
# ---------------------------------------------------------------------------
best_ret = []
for ci in range(len(COLUMNS)):
    vals = [ret_data[m][ci] for m in METHODS if ret_data[m][ci] is not None]
    best_ret.append(max(vals) if vals else None)

# ---------------------------------------------------------------------------
# Build LaTeX
# ---------------------------------------------------------------------------
col_spec = "l" + "r" * len(COLUMNS)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Mean episodic return and path efficiency (PE, \textit{italic}) "
             r"per method and environment at 1\,M training steps (seed 9, mean of last 3 "
             r"checkpoints). Bold return = best per column.}")
lines.append(r"\label{tab:results}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")

# header
lines.append("Method & " + " & ".join(d for _, _, d in COLUMNS) + r" \\")
lines.append(r"\midrule")

# data rows
for method in METHODS:
    row_cells = [METHOD_LABELS[method]]
    for ci in range(len(COLUMNS)):
        ret = ret_data[method][ci]
        pe  = pe_data[method][ci]

        if ret is None:
            row_cells.append("---")
            continue

        ret_str = f"{ret:.3f}"
        if best_ret[ci] is not None and abs(ret - best_ret[ci]) < 1e-4:
            ret_str = rf"\textbf{{{ret_str}}}"

        pe_str = rf"\textit{{{pe:.3f}}}" if pe is not None else ""

        cell = rf"\shortstack{{{ret_str}\\{pe_str}}}"
        row_cells.append(cell)

    lines.append(" & ".join(row_cells) + r" \\[2pt]")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines) + "\n"

out_path = OUT_DIR / "results_table.tex"
out_path.write_text(tex)
print(f"[make_table] saved → {out_path}")
print(tex)
