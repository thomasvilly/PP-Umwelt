"""
offline_gate_v2.py — Train a transition outcome regressor from domain randomization data.

The model predicts the expected outcome (EV delta or return) of training at a given level
given the current signal state. At inference, the model is queried once per candidate level;
the level with the highest predicted outcome is selected. This is fully agnostic of the
number of levels and the specific level sequence at deployment time.

Input:  [sig_vec_z (7 or 3), ENV_DESC[level] (4)] = 11-dim (or 7-dim GRPO-safe)
Output: scalar predicted outcome (raw, no activation)
Loss:   MSE

Column layout of .npy files produced by ppo.py --domain-rand-log-path:
  0:    iteration
  1:    active_level
  2-8:  sig_vec raw   [EV, EV_slope, adv_std, crit_gnorm_slope, crit_gnorm_mean, entropy, actor_gnorm_slope]
  9-15: sig_vec_z     (Welford z-score, per-run)
  16:   ev_this
  17:   ev_prev
  18:   mean_return
  19:   mean_success

Usage:
  uv run python offline_gate_v2.py --data-dir runs/domrand_data --out offline_gate_v2.pt
  uv run python offline_gate_v2.py --data-dir runs/domrand_data --out offline_gate_v2_return.pt --outcome return
  uv run python offline_gate_v2.py --data-dir runs/domrand_data --grpo-safe --out offline_gate_v2_grpo.pt
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Column indices
# ---------------------------------------------------------------------------
LEVEL_IDX   = 1
SIG_Z       = slice(9, 16)    # 7 Welford z-scored signal dims
EV_THIS     = 16
EV_PREV     = 17
RETURN_IDX  = 18

_ENV_DESC: dict = {
    0: [25/169,   0,    0, 25/169],   # 5×5, no walls, static
    1: [49/169,   0,    0, 49/169],   # 7×7, no walls, static
    2: [121/169,  0,    0, 121/169],  # 11×11, no walls, static
    3: [169/169,  0,    0, 169/169],  # 13×13, no walls, static
    4: [81/169,   1/81, 1, 81/169],   # 9×9, 1 dynamic wall, stochastic
}

_GRPO_IDX = [2, 5, 6]   # adv_std, entropy_mean, actor_gnorm_slope


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_domrand_data(data_dir: Path) -> np.ndarray:
    """Load and concatenate all .npy files in data_dir. Returns (N, 20) float32 array."""
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    data = np.concatenate([np.load(p) for p in files], axis=0)
    print(f"[v2] loaded {len(files)} files → {data.shape[0]} rows")
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature / label construction
# ---------------------------------------------------------------------------

def build_regression_features(
    rows: np.ndarray,
    outcome: str = "ev_delta",
    grpo_safe: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs for regression directly from domain-rand rows.
    X: [sig_z (7 or 3), ENV_DESC[active_level] (4)]  — shape (N, 11) or (N, 7)
    y: outcome scalar per row — shape (N,)
    All rows are valid (no masking needed).
    """
    if grpo_safe:
        sig = rows[:, [SIG_Z.start + i for i in _GRPO_IDX]]   # (N, 3)
    else:
        sig = rows[:, SIG_Z]                                    # (N, 7)

    env_desc = np.array(
        [_ENV_DESC[int(rows[i, LEVEL_IDX])] for i in range(len(rows))],
        dtype=np.float32,
    )
    X = np.concatenate([sig, env_desc], axis=1).astype(np.float32)

    if outcome == "ev_delta":
        y = (rows[:, EV_THIS] - rows[:, EV_PREV]).astype(np.float32)
    elif outcome == "return":
        y = rows[:, RETURN_IDX].astype(np.float32)
    else:
        raise ValueError(f"Unknown outcome: {outcome!r}. Choose 'ev_delta' or 'return'.")

    print(f"[v2] features: X={X.shape}, y={y.shape}, outcome={outcome!r}")
    print(f"[v2] y stats: mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(in_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 500,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[nn.Sequential, torch.Tensor, torch.Tensor]:
    """
    Train MSE regression model.
    Returns (model, mu, sigma) where mu/sigma are per-feature z-score params
    fit on the training split — stored alongside weights for frozen deployment.
    """
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]

    idx = rng.permutation(len(X))
    val_n   = max(1, int(len(idx) * val_frac))
    train_i = idx[val_n:]
    val_i   = idx[:val_n]

    X_tr, y_tr = X[train_i], y[train_i]
    X_va, y_va = X[val_i],   y[val_i]

    # Z-score normalization fit on training set
    mu    = X_tr.mean(axis=0).astype(np.float32)
    sigma = X_tr.std(axis=0).astype(np.float32)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)

    X_tr_t = torch.tensor((X_tr - mu) / sigma)
    y_tr_t = torch.tensor(y_tr).unsqueeze(1)
    X_va_t = torch.tensor((X_va - mu) / sigma)
    y_va_t = torch.tensor(y_va).unsqueeze(1)

    model = build_model(in_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    best_val_mse = float("inf")
    best_state   = None
    print(f"[v2] training: {len(X_tr)} train, {len(X_va)} val, in_dim={in_dim}")
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = mse(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_mse = mse(model(X_va_t), y_va_t).item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep:>4}  train_mse={loss.item():.4f}  val_mse={val_mse:.4f}  best={best_val_mse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, torch.tensor(mu), torch.tensor(sigma)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="runs/domrand_data",  help="dir of .npy files")
    parser.add_argument("--out",          default="offline_gate_v2.pt", help="output path")
    parser.add_argument("--outcome",      default="ev_delta", choices=["ev_delta", "return"])
    parser.add_argument("--grpo-safe",    action="store_true",           help="3-dim GRPO input only")
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--val-frac",     type=float, default=0.2)
    parser.add_argument("--extract-only", action="store_true",           help="load data, print stats, exit")
    args = parser.parse_args()

    rows = load_domrand_data(Path(args.data_dir))

    if args.extract_only:
        X, y = build_regression_features(rows, outcome=args.outcome, grpo_safe=args.grpo_safe)
        print("[v2] --extract-only: done.")
        return

    X, y = build_regression_features(rows, outcome=args.outcome, grpo_safe=args.grpo_safe)
    in_dim = X.shape[1]

    model, mu, sigma = train(X, y, epochs=args.epochs, lr=args.lr, val_frac=args.val_frac)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mu":         mu,
        "sigma":      sigma,
        "in_dim":     in_dim,
        "outcome":    args.outcome,
        "grpo_safe":  args.grpo_safe,
    }, out_path)
    print(f"[v2] saved → {out_path}  (in_dim={in_dim}, outcome={args.outcome!r})")


if __name__ == "__main__":
    main()
