"""
make_gifs.py — Convert MP4 recordings to GIFs for the ECE 757A paper.

Finds all MP4 files under videos/ whose directory name starts with gif-*,
takes the last episode (last N frames), and saves as a looping GIF.

Usage:
    uv run python make_gifs.py [--fps 15] [--max-frames 300] [--out-dir figures]

Prereqs:
    uv add imageio "imageio[ffmpeg]"
"""

import argparse
import re
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import imageio

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fps",            type=int, default=10,
                    help="GIF playback framerate")
parser.add_argument("--last-episodes",  type=int, default=3,
                    help="Number of final episodes (MP4 files) to include per GIF")
parser.add_argument("--video-dir",      type=str, default="videos",
                    help="Root directory containing per-run video subdirs")
parser.add_argument("--out-dir",        type=str, default="figures",
                    help="Output directory for GIFs")
parser.add_argument("--pattern",        type=str, default="gif-",
                    help="Only process run dirs whose exp_name starts with this prefix "
                         "(default: 'gif-'; use 'b1M' for the 1M sweep, '' for all)")
args = parser.parse_args()

video_root = Path(args.video_dir)
out_dir    = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Find MP4s under gif-* run dirs
# ---------------------------------------------------------------------------
mp4_files: list[tuple[str, Path]] = []  # (exp_name, run_dir)

# videos/ may have a gymnasium_env/ subdirectory — search recursively
for run_dir in sorted(video_root.rglob("*")):
    if not run_dir.is_dir():
        continue
    if not any(run_dir.glob("*.mp4")):
        continue
    name = run_dir.name
    m = re.search(r"__([^_][^_]*?)__", name)
    exp_name = m.group(1) if m else name
    if args.pattern and not exp_name.startswith(args.pattern):
        continue
    # if multiple dirs with same exp_name, keep latest by mtime
    existing = next((i for i, (e, _) in enumerate(mp4_files) if e == exp_name), None)
    if existing is not None:
        if run_dir.stat().st_mtime > mp4_files[existing][1].stat().st_mtime:
            mp4_files[existing] = (exp_name, run_dir)
    else:
        mp4_files.append((exp_name, run_dir))

if not mp4_files:
    print(f"[make_gifs] no gif-* MP4 files found under {video_root}/")
    print("  Run: set RUNS = GIF_RUNS in sweep.py, then: uv run python sweep.py")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Convert each MP4 to GIF
# ---------------------------------------------------------------------------
for exp_name, run_dir in mp4_files:
    # Post-training eval episodes are named final-episode-N.mp4; sort and take last N
    all_mp4s = sorted(run_dir.glob("final-episode-*.mp4"),
                      key=lambda p: int(re.search(r"episode-(\d+)", p.stem).group(1))
                                    if re.search(r"episode-(\d+)", p.stem) else 0)
    if not all_mp4s:
        # fallback: any MP4, sorted by step number
        all_mp4s = sorted(run_dir.glob("*.mp4"),
                          key=lambda p: int(re.search(r"\d+", p.stem).group())
                                        if re.search(r"\d+", p.stem) else 0)
    episode_mp4s = all_mp4s[-args.last_episodes:]
    if not episode_mp4s:
        print(f"  [warn] no MP4s in {run_dir}")
        continue
    print(f"[make_gifs] {exp_name}: using last {len(episode_mp4s)} episode(s) of {len(all_mp4s)} ...")

    all_frames = []
    for mp4_path in episode_mp4s:
        try:
            frames = iio.imread(str(mp4_path), plugin="pyav")
        except Exception:
            reader = imageio.get_reader(str(mp4_path))
            frames = np.stack([f for f in reader])
            reader.close()
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
        all_frames.append(frames)

    frames = np.concatenate(all_frames, axis=0)

    out_path = out_dir / f"{exp_name}.gif"
    iio.imwrite(
        str(out_path),
        frames,
        plugin="pillow",
        loop=0,
        fps=args.fps,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"  → {out_path}  ({len(frames)} frames from {len(episode_mp4s)} episodes, {size_kb:.0f} KB)")

print(f"[make_gifs] done — {len(mp4_files)} GIF(s) written to {out_dir}/")
