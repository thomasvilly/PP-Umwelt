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

METHODS = {
    "allo-8":    dict(curriculum_strategy="allopoietic", expand_every_n=8),
    "allo-16": dict(curriculum_strategy="allopoietic", expand_every_n=16),
    "allo-32": dict(curriculum_strategy="allopoietic", expand_every_n=32),
    "spdl-07": dict(curriculum_strategy="spdl"),
    "critgn":  dict(curriculum_strategy="heuristic", heuristic_signal="critic_gnorm",
                    heuristic_eps=0.001, signal_window=5),
    "offgate": dict(curriculum_strategy="offline_gate",
                    offline_gate_path="offline_gate_critgn.pt"),
    "lev-sel": dict(curriculum_strategy="level_selector",
                    level_selector_path="offline_gate_v2.pt"),
    "domrand": dict(curriculum_strategy="domain_rand", start_level=3, max_level=3),
}

ENVS = {
    # Original 4-level curriculum: 5×5 → 7×7 → 11×11 → 13×13 (no walls, static)
    "4lv":  dict(level_sequence="0,1,2,3"),
    # 5-level: adds 9×9 with 1 dynamic wall — qualitative shift (stochastic obstacles)
    "5lv":  dict(level_sequence="0,1,2,3,4"),
    # Skip 11×11: forces a large difficulty jump from 7×7 directly to 13×13
    "skip": dict(level_sequence="0,1,3"),
    # 5 -> 11 -> 9x9 dynamic
    "skip-dyn": dict(level_sequence="1,2,4"),
}

SEEDS = [1, 9, 5]

RUNS = [
    {**method_cfg, **env_cfg,
     "total_timesteps": budget,
     "seed": seed,
     "exp_name": f"{mname}-{ename}-b{budget // 1000}k-s{seed}"}
    for mname, method_cfg in METHODS.items()
    for ename, env_cfg in ENVS.items()
    for budget in [200_000, 300_000, 400_000]
    for seed in SEEDS
]

# RUNS = DOMRAND_RUNS # to generate the domain randomization data to train offline gate

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
