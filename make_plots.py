"""
make_plots.py — Learning curves for ECE 757A paper.

Reads TensorBoard event files and produces a 2×4 figure:
  top row    — path efficiency vs global_step (one subplot per env)
  bottom row — mean episodic return vs global_step (one subplot per env)
One line per method. IEEE column-width styling.

Output: figures/learning_curves.pdf  +  figures/learning_curves.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TB_ROOT  = Path("runs/gymnasium_env")
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SEED    = 9
BUDGET  = "b1M"

TAGS = {
    "pe":     "charts/path_efficiency",
    "return": "charts/rollout_mean_return",
}

METHODS = ["allo-8", "allo-16", "critgn", "lev-sel"]
ENVS    = ["4lv", "5lv", "skip", "skip-dyn"]
ENV_LABELS = {
    "4lv":      "4-level (static)",
    "5lv":      "5-level (+dynamic)",
    "skip":     "Skip (0→1→3)",
    "skip-dyn": "Skip-dyn (1→2→4)",
}

COLORS = {
    "allo-8":  "#1f77b4",
    "allo-16": "#17becf",
    "critgn":  "#ff7f0e",
    "lev-sel": "#d62728",
}
LINESTYLES = {"allo-8": "-", "allo-16": "--", "critgn": "-.", "lev-sel": "-"}
METHOD_LABELS = {
    "allo-8":  "Allo-8",
    "allo-16": "Allo-16",
    "critgn":  "CritGN",
    "lev-sel": "LevSel (ours)",
}

MAX_PTS = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tb_dir(exp_name: str) -> Path | None:
    matches = sorted(TB_ROOT.glob(f"*__{exp_name}__*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_scalar(tb_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(str(tb_dir), size_guidance={tag: 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=np.float32)
    vals   = np.array([e.value for e in events], dtype=np.float32)
    if len(steps) > MAX_PTS:
        idx = np.linspace(0, len(steps) - 1, MAX_PTS, dtype=int)
        steps, vals = steps[idx], vals[idx]
    return steps, vals


def smooth(vals, w=5):
    if len(vals) <= w:
        return vals
    kernel = np.ones(w) / w
    sm = np.convolve(vals, kernel, mode="same")
    sm[:w//2]  = vals[:w//2]
    sm[-w//2:] = vals[-w//2:]
    return sm


# ---------------------------------------------------------------------------
# IEEE rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":         8,
    "axes.linewidth":    0.5,
    "lines.linewidth":   1.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize":   7,
    "legend.framealpha": 0.8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
})

ROW_LABELS = {"pe": "Path Efficiency", "return": "Mean Return"}
ROW_YLIMS  = {"pe": (-0.05, 1.05), "return": None}

fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0), sharex=False)

for row_i, metric_key in enumerate(["pe", "return"]):
    tag = TAGS[metric_key]
    for col_i, env in enumerate(ENVS):
        ax = axes[row_i, col_i]
        for method in METHODS:
            exp = f"{method}-{env}-{BUDGET}-s{SEED}"
            tb_dir = find_tb_dir(exp)
            if tb_dir is None:
                print(f"  [warn] no TB dir for {exp}")
                continue
            steps, vals = load_scalar(tb_dir, tag)
            if len(steps) == 0:
                print(f"  [warn] no '{tag}' tag in {tb_dir.name}")
                continue
            ax.plot(steps / 1_000, smooth(vals),
                    color=COLORS[method], linestyle=LINESTYLES[method],
                    label=METHOD_LABELS[method], alpha=0.9)

        if row_i == 0:
            ax.set_title(ENV_LABELS[env])
        if col_i == 0:
            ax.set_ylabel(ROW_LABELS[metric_key])
        if ROW_YLIMS[metric_key]:
            ax.set_ylim(*ROW_YLIMS[metric_key])
        ax.set_xlabel("Steps (k)")
        ax.grid(True, linewidth=0.3, alpha=0.5)

# shared legend below the figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=7)

fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(OUT_DIR / "learning_curves.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "learning_curves.png", dpi=300, bbox_inches="tight")
print(f"[make_plots] saved → {OUT_DIR}/learning_curves.pdf  +  .png")
