# Homeostatic Umwelt Expansion — Project Spec
**ECE 757A | Thomas Villeneuve | Due: March 30, 2026**

---

## Research Question

Can internal physiological signals of a neural network (gradient norms, critic quality, advantage
statistics) gate curriculum expansion more effectively than external performance metrics, and does
this constitute a form of autopoietic self-organisation?

**Phase 2 finding:** Yes — a single well-chosen internal signal (critic gradient norm slope W=5,
or explained variance threshold) matches or beats allopoietic at 300k steps. Key tension:
signals closest to the critic work best but are architecture-specific; `advantage_std` is the
only signal transferable to critic-free RL (GRPO/reinforce).

## Core Hypothesis

An agent whose curriculum expansion is gated by a learned forward model of its own learning
dynamics will:
1. Expand its umwelt more stably than reward-gated or externally-scheduled baselines
2. Recover more gracefully from environmental perturbations
3. Show measurably different internal dynamics at expansion boundaries vs. baselines

---

## Architecture

### Base Agent
- **Algorithm:** PPO (CleanRL `ppo.py` scaffold, modified directly)
- **Network:** MLP actor-critic (separate actor and critic heads, shared obs input)
- **Environment:** Custom 2D grid world, Gymnasium-compatible

### Grid World
- **Observation:** 3 channels × 13×13 padded, flattened to 508 dims
  (507 spatial + 1 BFS boolean hint: 1 if direct path to goal is wall-blocked)
- **Actions:** 4 (up, down, left, right)
- **Reward:** +1 goal, −0.01/step, Manhattan distance shaping × 0.10
- **Termination:** goal reached or 4×size² steps
- **Curriculum levels:** 5×5 → 7×7 → 11×11 → 13×13 (no walls); 13×13 w/ walls (stretch)

### Internal Signal Vector (gate input, 7-dim)
Computed once per rollout. W=5 default (empirically validated).

| # | Signal | Type | GRPO-safe? |
|---|--------|------|-----------|
| 1 | `explained_variance` | point-in-time | No (critic) |
| 2 | `expl_var_slope` | rolling W | No (critic) |
| 3 | `advantage_std` | point-in-time | **Yes** |
| 4 | `critic_gnorm_slope` | rolling W | No (critic) |
| 5 | `critic_gnorm_mean` | point-in-time | No (critic) |
| 6 | `entropy_mean` | point-in-time | **Yes** |
| 7 | `actor_gnorm_slope` | rolling W | **Yes** |

Removed from original spec: `param_delta_actor/critic` (6 attempts, all failed), combined
`grad_norm_slope` (noisier than critic gnorm alone), `critic_loss_var`, `grad_norm_var`.

### D3 Meta-Controller (Phase 3)
Rather than a binary gate, D3 selects which curriculum environment to train in next.
All embeddings (state and environment) live in the same 16-dim L2-normalised space.
Cosine similarity is the only scoring mechanism — no fixed thresholds anywhere.

- **State encoder:** `Linear(W×sig_dim, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 16)`; L2-normalised output
- **Environment embeddings:** `nn.Parameter(4, 16)`, initialised from analytic structural features (normalised grid cells, wall count, max steps) projected via fixed random matrix — no prior runs needed
- **Label:** EV improvement — `label[t] = 1 if mean(EV[t+1:t+K]) > EV[t]`, K=5 (same as Phase 2)
- **Training signal:** contrastive loss with soft cross-entropy target (one-hot for label=1, uniform-over-others for label=0) + diversity regulariser (λ=0.01) + selection entropy regulariser (λ=0.01)
- **Replay buffer:** 200 (z_detached, env_idx, label) triples for cross-environment gradient augmentation
- **Single optimizer:** Adam lr=1e-3 for both encoder and env embeddings; τ=0.07 contrastive temperature
- **Two phases:** ε-greedy warmup (ε: 1→0 over 30 rollouts) → pure learned selection
- **Backward movement:** emergent — no special logic; state drifts toward compatible env naturally
- **All levels unlocked from start**
- **Two signal modes run simultaneously:**
  - `d3-full`: 7-dim (all validated signals including critic-dependent)
  - `d3-grpo`: 3-dim (adv_std, entropy_mean, actor_gnorm_slope — GRPO-transferable only)
  - Comparing these directly quantifies the cost of generalisability in the learned setting

---

## Baselines

| Baseline | Expansion trigger | Best result |
|---|---|---|
| Allo n=5 | Every 5 rollouts | ✅ Reaches L3; over-expands early |
| Allo n=8 | Every 8 rollouts | ✅ 300k: 95% success, 1.31 return |
| Allo n=10 | Every 10 rollouts | ✅ Standard baseline |
| SPDL (0.7) | Return > 0.7 | ✅ 500k; unstable at tight budgets |
| Domain rand | Uniform over unlocked levels | Tested; generally unstable |
| Heuristic: EV slope | `expl_var_slope` < 0.001 | ✅ 500k: L3, 100% success |
| Heuristic: critgn-w5 | `critic_gnorm_slope` < 0.001, W=5 | ✅ **300k: 98% success, 1.39 return** |
| Heuristic: adv_std | `advantage_std` < 0.30 | ✅ 300k: 95% success (best GRPO-safe) |
| Heuristic: ev_abs | `explained_variance` > 0.05 | ✅ 300k: 96% success |
| **D3-full** (7-dim signals) | cosine compatibility softmax | **Phase 3** |
| **D3-grpo** (3-dim GRPO-safe) | cosine compatibility softmax | **Phase 3** |

All share identical PPO internals and grid world; only the expansion trigger differs.

---

## Empirical Signal Findings (Phase 2)

### Calibration data (from allo-n8-b300k-s1)

Values at each expansion event — use to calibrate Phase 3 gate training thresholds:

| Signal | At L0→L1 | At L1→L2 | At L2→L3 | Converged L3 | Trend |
|--------|----------|----------|----------|--------------|-------|
| `explained_variance` | 0.007 | 0.037 | 0.048 | 0.06–0.35 | ↑ |
| `critic_gnorm_mean` | 0.024 | 0.052 | 0.029 | 0.024–0.060 | varies |
| `advantage_std` | 0.301 | 0.347 | 0.359 | 0.28–0.36 | stable |
| `entropy_mean` | 1.384 | 1.373 | 1.369 | 1.24–1.37 | ↓ |
| `grad_norm_mean` | 0.077 | 0.095 | 0.104 | 0.16–0.35 | ↑ with level |
| `value_std` | 0.100 | 0.153 | 0.159 | 0.16–0.25 | ↑ with level |

**Note:** `grad_norm_mean` and `value_std` both *increase* with curriculum level — cannot be
used as simple low-threshold gates.

### Signal taxonomy

**Tier 1 — Critic-quality (most predictive, critic-dependent):**
EV (abs + slope), `critic_gnorm_slope`, `critic_gnorm_abs`

**Tier 2 — Gradient signal quality (critic-free, GRPO-transferable):**
`advantage_std` (best; θ=0.30), `actor_gnorm_slope` (secondary)

**Tier 3 — PPO-specific (discard for generality):**
`approx_kl`, `clipfrac`

**Definitively abandoned:** `param_delta_*` — 6 attempts across 2 orders of magnitude, all
failed. Weight change magnitude spans too wide a range for reliable thresholding.

### Key tension (paper contribution)
The more internal a signal (farther from external reward), the more it tends to be
architecture-specific. The best signals (EV, critic gnorm) require a critic and do not
transfer to GRPO. The only GRPO-safe signal that reliably works (`adv_std`) is noisier and
threshold-sensitive. This internality ↔ generalisability trade-off is a core empirical finding.

### Budget sensitivity
- **100k (24 iters):** Only allo n=5/8 reliably reaches L3. W=10 heuristics structurally
  blocked (need ~45 iters minimum for 3 expansions with window fill).
- **200k (48 iters):** Mixed. Best heuristics reach L3 but return often negative (too fast).
- **300k (73 iters):** `critgn-w5` beats allo-n8 on return (1.39 vs 1.31). ✓
- **500k:** All methods saturate at 97–100% success. No discrimination.

---

## Evaluation Metrics

- **Sample efficiency:** cumulative reward vs. rollout count per curriculum level
- **Expansion timing:** rollout index at which each level is unlocked
- **Post-expansion stability:** critic loss variance in K rollouts following each expansion
- **Perturbation recovery:** (stretch) inject environmental change, measure rollouts to restabilize

---

## Timeline

### Part 1 — COMPLETE
- [x] CleanRL PPO scaffold modified
- [x] Grid world (numpy + Gymnasium, BFS solvability, WASD playable)
- [x] PPO learns all levels via sequential curriculum (97-100% success at L3)
- [x] Internal signal logging (grad norm, critic loss, entropy, EV per rollout)
- [x] Allopoietic baseline (every N rollouts)
- [x] SPDL-style baseline (reward threshold)
- [x] sweep.py for parallel/serial runs

### Part 2 — COMPLETE
- [x] Rolling slopes, per-component gnorms (actor + critic), adv_std
- [x] 11 heuristic signal variants implemented and tested
- [x] Bug fix: point-in-time signals decoupled from `window_full` gate
- [x] 78-run ablation (13 methods × 3 budgets × 2 seeds)
- [x] Signal taxonomy, calibration data, GRPO transferability analysis
- [x] `param_delta` definitively abandoned (6 attempts)
- [ ] Plots: expansion timing, post-expansion stability, sample efficiency

### Part 3 — IN PROGRESS (D3 joint contrastive meta-controller)
- [x] `d3.py`: D3Controller, contrastive loss, replay buffer, ε-greedy warmup
- [x] Env embeddings initialised from analytic structural features (no prior runs needed)
- [x] K=5 EV improvement labels; window cache for grad recomputation
- [x] ppo.py: `d3_contrastive` strategy, D3 args, sig_vec assembly, env selection hook
- [x] sweep.py: Phase 3 comparison METHODS (d3-full, d3-grpo, critgn-w5, allo-n8)
- [ ] Primary sweep: d3-full vs d3-grpo vs critgn-w5 vs allo-n8, 3 seeds, 300k
- [ ] Secondary ablations: warmup_rollouts, tau, temperature, selection mode
- [ ] Final plots and paper writeup

---

## Open Questions

1. Shared vs. separate trunk → **Resolved:** separate heads, shared obs input.
2. K for prediction horizon → **K=5 default** (ablate 3/5/10).
3. Mid-rollout expansion → **Resolved:** rollout boundaries only.
4. `terminated` vs `truncated` → **Resolved:** `truncated` bootstraps value, not terminal.
5. W (slope window) → **Resolved: W=5** (empirically best at 300k; structurally required at tight budgets).
6. ε calibration → **Resolved:** calibration table in Empirical Signal Findings above.
7. Domain rand scope → Sample from *unlocked* levels only.
8. Does the learned gate generalise across budget sizes (100k vs 300k)?
9. Should gate input be restricted to GRPO-safe signals to strengthen the generalisation claim?

---

## Key References
- Klink et al. (2020) — Self-Paced Deep RL (SPDL baseline)
- Portelas et al. (2020) — Teacher algorithms for curriculum learning
- Dennis et al. (2020) — Unsupervised Environment Design
- Oudeyer et al. (2007) — Intrinsic motivation / learning progress
- Maturana & Varela (1980) — Autopoiesis and Cognition
- CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

---

## Future Directions (post-submission scope)

*Prerequisite: internal signals show some advantage over external baselines (even heuristically),
justifying investment in richer representation learning for the gate.*

### D1 — Contrastive gate with online hindsight labels

Train a small encoder on (signal_window, label) pairs using the same K-rollout delay labels as
Part 3. Contrastive loss pulls together signal windows that precede good transitions and pushes
apart those that precede bad ones. At inference, gate fires when the current window's embedding
is closer to the "good transition" cluster than the "bad" cluster.

*Why:* the contrastive embedding may generalise across curriculum levels better than a scalar
predictor, learning a representation of "readiness" rather than a direct prediction of EV.

Key decisions: window size (ablate 5/10/20 rollouts), encoder architecture (1D conv vs flat MLP),
negative pair construction, momentum encoder (EMA).

---

### D2 — JEPA-style predictive gate (fully self-supervised)

Train a context encoder and predictor to predict the *representation* of a future signal window
from the current one, using a momentum-updated target encoder. Prediction error becomes the gate:
- Error low, slope flat → dynamics stabilised → advance
- Error high, slope falling → still learning → wait
- Error high, slope rising → chaotic → hold

*Why:* fully self-supervised, no hindsight labels, no reward. The gate asks "have my internals
settled into a predictable regime?" — the cleanest version of the homeostatic framing.

---

### D3 — Calibration phase + signature navigation (non-linear curriculum)

Two-phase system:
- **Phase 1 (calibration):** encoder observes signal dynamics at each level. Computes mean
  embedding µᵢ and covariance Σᵢ per level — the "internal signature" at initialisation.
- **Phase 2 (training):** gate monitors `d(z_t, µ_current)`. When this falls below threshold,
  advance to level with smallest embedding distance that is still harder than current. If current
  state is closer to an easier level's signature → retreat.

*Why:* calibration gives a reference independent of how much the actor/critic has learned.
Distance to initialisation signature measures learning progress, not performance.

---

### Ordering
D1 before D2 (reuses Part 3 label infrastructure). D2 before D3 (encoder architecture carries
over). D1 vs D2 is the cleanest ablation: same signals, supervised contrastive vs self-supervised.
