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
# Ablation sweep: env-sequence × method × budget
#
# METHODS  — 3 curriculum strategies
# ENVS     — 3 level sequences (original 4-level, 5-level + dynamic walls, skip 11×11)
# BUDGETS  — 300k + 400k for env variation; 200k added for budget ablation on 4-level only
# SEEDS    — 3 seeds per config
#
# Total: (3 envs × 3 methods × 2 budgets × 3 seeds) + (3 methods × 1 extra budget × 3 seeds)
#      = 54 + 9 = 63 runs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase A — domain randomization data collection (run FIRST, before Phase C)
# Small network (hidden=32) for cheap data; transfer-to-full-net is part of the experiment.
# Set RUNS = DOMRAND_RUNS to collect data, then train offline_gate_v2.pt via offline_gate_v2.py.
# ---------------------------------------------------------------------------
DOMRAND_RUNS = [
    {"curriculum_strategy": "domain_rand",
     "start_level": 3,            # all 4 levels accessible from step 0
     "max_level": 3,
     "hidden_state_size": 8,     # half-size network (transfer experiment)
     "total_timesteps": 300_000,
     "domain_rand_log_path": f"runs/domrand_data/dr-s{seed}.npy",
     "seed": seed,
     "exp_name": f"dr-small-s{seed}"}
    for seed in range(1, 21)      # 20 runs → ~4000-5000 labeled rows
]

# ---------------------------------------------------------------------------
# Phase B/C methods (main ablation — add level-sel after offline_gate_v2.pt is trained)
# ---------------------------------------------------------------------------
METHODS = {
    # Periodic baseline: expand every 8 rollouts regardless of agent state
    # "allo":    dict(curriculum_strategy="allopoietic", expand_every_n=8),
    # Best heuristic from Phase 2: fire when critic grad-norm slope flattens
    # "critgn":  dict(curriculum_strategy="heuristic", heuristic_signal="critic_gnorm",
    #                 heuristic_eps=0.001, signal_window=5),
    # Offline gate v1: binary expand/wait, distilled from critgn-w5
    # "offgate": dict(curriculum_strategy="offline_gate",
    #                 offline_gate_path="offline_gate_critgn.pt"),
    # Offline gate v2: N-class level selector trained on domain-rand data
    # Uncomment after running: uv run python offline_gate_v2.py --data-dir runs/domrand_data
    "lev-sel": dict(curriculum_strategy="level_selector",
                    level_selector_path="offline_gate_v2.pt"),
}

ENVS = {
    # Original 4-level curriculum: 5×5 → 7×7 → 11×11 → 13×13 (no walls, static)
    "4lv":  dict(level_sequence="0,1,2,3"),
    # 5-level: adds 9×9 with 1 dynamic wall — qualitative shift (stochastic obstacles)
    "5lv":  dict(level_sequence="0,1,2,3,4"),
    # Skip 11×11: forces a large difficulty jump from 7×7 directly to 13×13
    "skip": dict(level_sequence="0,1,3"),
}

SEEDS = [1, 2, 3]

# Part 1 — environment × method × budget (300k + 400k)
# Tests: does level conditioning and the offline gate generalise across env configs and budgets?
RUNS = [
    {**method_cfg, **env_cfg,
     "total_timesteps": budget,
     "seed": seed,
     "exp_name": f"{mname}-{ename}-b{budget // 1000}k-s{seed}"}
    for mname, method_cfg in METHODS.items()
    for ename, env_cfg in ENVS.items()
    for budget in [300_000, 400_000]
    for seed in SEEDS
]

# Part 2 — budget ablation on original 4-level only (200k; 300k+400k already in Part 1)
# Tests: does budget pressure differentially hurt heuristic vs learned methods?
RUNS += [
    {**method_cfg, "level_sequence": "0,1,2,3",
     "total_timesteps": 200_000,
     "seed": seed,
     "exp_name": f"{mname}-4lv-b200k-s{seed}"}
    for mname, method_cfg in METHODS.items()
    for seed in SEEDS
]

# RUNS = DOMRAND_RUNS


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
    print(f"[sweep] Methods: {list(METHODS)}")
    print(f"[sweep] Envs:    {list(ENVS)}")
    print(f"[sweep] (DOMRAND_RUNS has {len(DOMRAND_RUNS)} runs — set RUNS=DOMRAND_RUNS to collect Phase A data)")
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
