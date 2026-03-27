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

MAX_PARALLEL = 3  # adjust based on GPU/CPU headroom

RUNS = [
    dict(exp_name="allopoietic-n10", curriculum_strategy="allopoietic", expand_every_n=10, total_timesteps=200000),
    dict(exp_name="allopoietic-n25", curriculum_strategy="allopoietic", expand_every_n=25, total_timesteps=200000),
    dict(exp_name="spdl-07",         curriculum_strategy="spdl", spdl_reward_threshold=0.7, total_timesteps=200000),
]


def run_one(cfg: dict) -> tuple:
    exp_name = cfg["exp_name"]
    log_dir = Path("runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_name}.log"

    cmd = ["uv", "run", "python", "ppo.py"]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]

    print(f"[sweep] starting: {exp_name}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=f)

    status = "done" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[sweep] {status}: {exp_name}  (log: {log_path})")
    return exp_name, proc.returncode


def summarise(exp_name: str):
    """Print last-5 key metrics for a completed run."""
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
    for name, rc in completed:
        if rc == 0:
            summarise(name)
        else:
            print(f"  {name}: FAILED")
