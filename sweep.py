"""
sweep.py — run multiple ppo.py configs in parallel.
Edit RUNS at the top, then: uv run python sweep.py
Logs go to runs/logs/<exp_name>.log
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

MAX_PARALLEL = 1  # adjust based on GPU/CPU headroom

# ---------------------------------------------------------------------------
# Large ablation sweep: 13 methods × 3 budgets × 2 seeds = 78 runs
# Goal: stress-test all methods under compute constraints to reveal ordering.
# At 500k everything saturates; 100k/200k/300k expose genuine differences.
#
# Key tension: heuristic signals use rolling window W=10 per level.
# iterations per budget: 100k=24, 200k=48, 300k=73 (batch=4096)
# Need 3 expansions (L0→L3). Min iterations for heuristic: ~3×15=45.
# → At 100k, W=10 heuristics structurally can't reach L3.
# → heur-ev-w5 (W=5) tests whether halving the window rescues tight budgets.
# ---------------------------------------------------------------------------

METHODS = {
    # --- ev_abs: point-in-time EV threshold (no window, fires when critic is good enough) ---
    # Calibration: EV at expansion was 0.007/0.037/0.048; threshold just above forces real critic learning
    "heur-ev-abs-005":  dict(curriculum_strategy="heuristic", heuristic_signal="ev_abs", ev_abs_eps=0.05),
    "heur-ev-abs-008":  dict(curriculum_strategy="heuristic", heuristic_signal="ev_abs", ev_abs_eps=0.08),
    "heur-ev-abs-010":  dict(curriculum_strategy="heuristic", heuristic_signal="ev_abs", ev_abs_eps=0.10),

    # --- crit_gnorm_abs: critic grad norm threshold (no window, drops at convergence) ---
    # Calibration: drops to 0.024 at L0 convergence, 0.029 at L2 convergence; spikes to 0.052-0.078 at new levels
    "heur-cg-abs-035":  dict(curriculum_strategy="heuristic", heuristic_signal="crit_gnorm_abs", crit_gnorm_abs_eps=0.035),
    "heur-cg-abs-050":  dict(curriculum_strategy="heuristic", heuristic_signal="crit_gnorm_abs", crit_gnorm_abs_eps=0.050),
    "heur-cg-abs-060":  dict(curriculum_strategy="heuristic", heuristic_signal="crit_gnorm_abs", crit_gnorm_abs_eps=0.060),

    # --- adv_std: more calibrated thresholds (stable range 0.28–0.36, θ=0.3 was on the edge) ---
    "heur-advstd-030":  dict(curriculum_strategy="heuristic", heuristic_signal="adv_std", adv_std_eps=0.30),
    "heur-advstd-035":  dict(curriculum_strategy="heuristic", heuristic_signal="adv_std", adv_std_eps=0.35),
    "heur-advstd-040":  dict(curriculum_strategy="heuristic", heuristic_signal="adv_std", adv_std_eps=0.40),

    # --- critic_gnorm slope with W=5 (already works at W=10/500k; test tighter budget) ---
    "heur-critgn-w5":   dict(curriculum_strategy="heuristic", heuristic_signal="critic_gnorm", heuristic_eps=0.001, signal_window=5),
}

BUDGETS = [100_000, 200_000, 300_000]
SEEDS   = [1, 2]

RUNS = [
    {**method_cfg,
     "total_timesteps": budget,
     "seed": seed,
     "exp_name": f"{name}-b{budget // 1000}k-s{seed}"}
    for name, method_cfg in METHODS.items()
    for budget in BUDGETS
    for seed in SEEDS
]


def run_one(cfg: dict) -> tuple:
    exp_name = cfg["exp_name"]
    log_dir = Path("runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_name}.log"

    cmd = ["uv", "run", "python", "ppo.py"]
    for k, v in cfg.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd += [flag]
        else:
            cmd += [flag, str(v)]

    print(f"[sweep] starting: {exp_name}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=f)

    status = "done" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[sweep] {status}: {exp_name}  (log: {log_path})")
    return exp_name, proc.returncode


def summarise(exp_name: str):
    """Print last-3 key metrics for a completed run."""
    run_root = Path("runs/gymnasium_env")
    matches = sorted(run_root.glob(f"*__{exp_name}__*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        print(f"  [no TB dir found for {exp_name}]")
        return
    path = matches[-1]
    ea = event_accumulator.EventAccumulator(str(path), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    def last(tag, n=3):
        if tag not in tags:
            return "N/A"
        return [round(s.value, 3) for s in ea.Scalars(tag)[-n:]]

    level_vals = [s.value for s in ea.Scalars("curriculum/level")] if "curriculum/level" in tags else []
    max_level = int(max(level_vals)) if level_vals else "N/A"
    print(f"  {exp_name}: level={max_level}  success={last('charts/success_rate')}  "
          f"return={last('charts/rollout_mean_return')}  entropy={last('internal_signals/entropy_mean')}")


if __name__ == "__main__":
    print(f"[sweep] {len(RUNS)} total runs, MAX_PARALLEL={MAX_PARALLEL}")
    t0 = time.time()
    completed = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, cfg): cfg["exp_name"] for cfg in RUNS}
        for f in as_completed(futures):
            name, rc = f.result()
            completed.append((name, rc))

    elapsed = time.time() - t0
    print(f"\n[sweep] all {len(RUNS)} runs finished in {elapsed / 60:.1f} min\n")
    print("=== Summary ===")
    for name, rc in sorted(completed):
        if rc == 0:
            summarise(name)
        else:
            print(f"  {name}: FAILED")
