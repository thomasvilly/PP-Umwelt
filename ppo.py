# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

from collections import deque

import gymnasium as gym
import gymnasium_env  # registers gymnasium_env/GridWorld-v0
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

"""
CleanRL ppo implementation
"""

# 4-dim analytic environment descriptor per curriculum level.
# [normalised_grid_cells, normalised_wall_count, stochasticity_flag, normalised_horizon]
# Used by lc_gate and offline_gate strategies; no prior runs needed.
_ENV_DESC: dict = {
    0: [25/169,   0,    0, 25/169],   # 5×5, no walls, static
    1: [49/169,   0,    0, 49/169],   # 7×7, no walls, static
    2: [121/169,  0,    0, 121/169],  # 11×11, no walls, static
    3: [169/169,  0,    0, 169/169],  # 13×13, no walls, static
    4: [81/169,   1/81, 1, 81/169],   # 9×9, 1 dynamic wall, stochastic
}


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "gymnasium_env/GridWorld-v0"
    """the id of the environment (used for run naming)"""
    curriculum_strategy: str = "allopoietic"
    """curriculum expansion strategy: allopoietic | spdl | domain_rand | heuristic | lc_gate | offline_gate | level_selector"""
    expand_every_n: int = 50
    """allopoietic: expand every N iterations"""
    spdl_reward_threshold: float = 0.7
    """spdl: expand when mean episodic return over last rollout exceeds this"""
    signal_window: int = 10
    """W: rolling window size for slope signals and param-delta (rollouts)"""
    heuristic_eps: float = 0.0
    """heuristic gate flat-band threshold for slope signals; 0 = disabled."""
    heuristic_signal: str = "both"
    """which signal(s) to use: both | gnorm | expl_var | or | param_delta | adv_std | entropy | kl | ev_abs | crit_gnorm_abs"""
    param_delta_eps: float = 0.0
    """threshold for param_delta heuristic (L2 weight change); 0 = disabled."""
    adv_std_eps: float = 0.0
    """threshold for adv_std heuristic: expand when advantage std drops below this; 0 = disabled."""
    entropy_eps: float = 0.0
    """threshold for entropy heuristic: expand when policy entropy drops below this; 0 = disabled."""
    kl_eps: float = 0.0
    """threshold for approx_kl heuristic: expand when per-rollout KL drops below this; 0 = disabled."""
    clipfrac_eps: float = 0.0
    """threshold for clipfrac heuristic: expand when clip fraction drops below this; 0 = disabled."""
    ev_abs_eps: float = 0.0
    """threshold for point-in-time EV gate: expand when explained_variance exceeds this; 0 = disabled."""
    crit_gnorm_abs_eps: float = 0.0
    """threshold for critic grad norm gate: expand when critic_grad_norm_mean drops below this; 0 = disabled."""
    actor_only_param_delta: bool = False
    """if True, param_delta gate uses actor delta only (not requiring critic to also stabilise)."""

    # Level-conditioned gate (curriculum_strategy="lc_gate")
    # Online BCE gate: input = 7-dim sig_vec + 4-dim env descriptor = 11-dim (or 7-dim GRPO)
    lc_gate_lr:         float = 1e-3  # Adam lr
    lc_gate_k:          int   = 5     # K-rollout EV look-ahead for labels
    lc_gate_thr:        float = 0.5   # p(expand) threshold
    lc_gate_grpo_safe:  bool  = False # if True, use 3-dim GRPO signals → 7-dim input

    # Offline meta-learned gate (curriculum_strategy="offline_gate")
    # Frozen gate trained offline on Phase 2 sweep data (see offline_gate.py)
    offline_gate_path:      str   = "offline_gate.pt"  # path to saved weights
    offline_gate_thr:       float = 0.5                # p(expand) threshold
    offline_gate_grpo_safe: bool  = False              # if True, 3-dim GRPO signals → 7-dim input

    # Domain randomization data logging — used to generate training data for offline gate v2
    domain_rand_log_path: str = ""
    """if non-empty, save per-rollout (sig_vec, level, EV, return) as .npy at end of run"""

    # Level selector (curriculum_strategy="level_selector")
    # Outcome regressor trained on domain-rand data (see offline_gate_v2.py).
    # Queried once per candidate level; switches to level with highest predicted outcome.
    level_selector_path:      str  = ""     # path to offline_gate_v2.pt
    level_selector_grpo_safe: bool = False  # if True, use 3-dim GRPO signals → 7-dim input

    level_sequence: str = ""
    """comma-separated curriculum level sequence, e.g. '0,1,2,3' or '0,1,3' or '0,1,2,3,4'.
    Overrides start_level/max_level when set."""
    max_level: int = 3
    """maximum curriculum level (0-3 for static walls, 4 adds dynamic objects); ignored when level_sequence is set"""
    start_level: int = 0
    """curriculum level to begin training at; ignored when level_sequence is set"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    hidden_state_size: int = 64
    """size of the hidden states in actor & critic nets"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(level, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array", level=level)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make("gymnasium_env/GridWorld-v0", level=level)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_state_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_state_size, args.hidden_state_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_state_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_state_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_state_size, args.hidden_state_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_state_size, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Parse curriculum level sequence
    if args.level_sequence:
        _level_seq = [int(x.strip()) for x in args.level_sequence.split(",") if x.strip()]
    else:
        _level_seq = list(range(args.start_level, args.max_level + 1))
    _level_seq_idx = 0
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    current_level = _level_seq[0]
    envs = gym.vector.SyncVectorEnv(
        [make_env(current_level, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Level-conditioned gate: online BCE MLP, 11-dim (or 7-dim GRPO) input
    lc_gate = lc_gate_opt = lc_bce = None
    lc_buf: list = []
    lc_ev_hist: list = []
    lc_warmup_done = False
    lc_gate_prob = 0.0
    if args.curriculum_strategy == "lc_gate":
        _lc_in = (3 if args.lc_gate_grpo_safe else 7) + 4
        lc_gate = nn.Sequential(
            nn.Linear(_lc_in, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        ).to(device)
        lc_gate_opt = torch.optim.Adam(lc_gate.parameters(), lr=args.lc_gate_lr)
        lc_bce = nn.BCELoss()

    # Offline gate: frozen MLP loaded from disk (weights + z-score normalisation params)
    offline_gate = None
    offline_gate_mu = offline_gate_sigma = None
    offline_gate_prob = 0.0
    if args.curriculum_strategy == "offline_gate":
        import os as _os
        _og_in = (3 if args.offline_gate_grpo_safe else 7) + 4
        offline_gate = nn.Sequential(
            nn.Linear(_og_in, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        ).to(device)
        if _os.path.exists(args.offline_gate_path):
            _ckpt = torch.load(args.offline_gate_path, map_location=device)
            if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
                offline_gate.load_state_dict(_ckpt["state_dict"])
                offline_gate_mu    = _ckpt["mu"].to(device)
                offline_gate_sigma = _ckpt["sigma"].to(device)
            else:
                offline_gate.load_state_dict(_ckpt)  # legacy format (no normalisation)
        else:
            print(f"[offline_gate] WARNING: {args.offline_gate_path} not found — gate will be random")
        offline_gate.eval()

    # Level selector: frozen outcome regressor, trained on domain-rand data (offline_gate_v2.py)
    # Queries model once per candidate level; selects level with highest predicted outcome.
    level_selector = None; level_selector_mu = level_selector_sigma = None
    if args.curriculum_strategy == "level_selector":
        import os as _os4
        _ls_in = (3 if args.level_selector_grpo_safe else 7) + 4
        level_selector = nn.Sequential(
            nn.Linear(_ls_in, 16), nn.ReLU(), nn.Linear(16, 1)
        ).to(device)
        if _os4.path.exists(args.level_selector_path):
            _ls_ckpt = torch.load(args.level_selector_path, map_location=device)
            level_selector.load_state_dict(_ls_ckpt["state_dict"])
            if "mu" in _ls_ckpt:
                level_selector_mu    = _ls_ckpt["mu"].to(device)
                level_selector_sigma = _ls_ckpt["sigma"].to(device)
        else:
            print(f"[level_selector] WARNING: {args.level_selector_path} not found — scores will be random")
        level_selector.eval()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Curriculum and signal tracking state
    steps_since_expansion = 0
    active_level = current_level  # for domain_rand: level envs are currently set to
    lr_reset_iteration = 1  # iteration from which current LR annealing started
    rolling_critic_buf = deque(maxlen=10)
    rollout_returns = []
    rollout_successes = []
    rollout_path_efficiencies = []

    # Rolling window histories — all share window size W (args.signal_window)
    # signals: grad_norm, expl_var, actor_gnorm, critic_gnorm, param_delta
    W = args.signal_window
    grad_norm_history, expl_var_history, actor_gnorm_history, critic_gnorm_history, param_delta_history = (
        deque(maxlen=W) for _ in range(5)
    )
    actor_params_snap  = torch.cat([p.detach().cpu().flatten() for p in agent.actor.parameters()])
    critic_params_snap = torch.cat([p.detach().cpu().flatten() for p in agent.critic.parameters()])

    def linslope(buf):
        """Linear regression slope over a rolling deque. Returns 0 if fewer than 2 samples."""
        if len(buf) < 2:
            return 0.0
        y = np.array(buf, dtype=np.float64)
        return float(np.polyfit(np.arange(len(y), dtype=np.float64), y, 1)[0])

    # Running z-score state (Welford online algorithm) — shared by domain_rand_log and level_selector
    sig_vec   = np.zeros(7, dtype=np.float32)   # safe default before first rollout completes
    sig_vec_z = np.zeros(7, dtype=np.float32)
    _ev_prev  = float("nan")
    _rs_n    = 0
    _rs_mean = np.zeros(7, dtype=np.float64)
    _rs_M2   = np.zeros(7, dtype=np.float64)
    _dr_rows: list = []

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - lr_reset_iteration) / args.num_iterations
            frac = max(frac, 0.0)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_returns.clear()
        rollout_successes.clear()
        rollout_path_efficiencies.clear()

        # Domain randomisation: resample active level each rollout from unlocked levels
        if args.curriculum_strategy == "domain_rand" and current_level > 0:
            new_active = int(np.random.randint(0, current_level + 1))
            if new_active != active_level:
                active_level = new_active
                envs.close()
                envs = gym.vector.SyncVectorEnv(
                    [make_env(active_level, i, args.capture_video, run_name) for i in range(args.num_envs)]
                )
                next_obs, _ = envs.reset(seed=args.seed + iteration)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)

        # Level selector: query regressor for each candidate level, pick highest predicted outcome.
        # Fires every rollout; can advance, stay, or retreat. Warmup: wait until _rs_n >= 5.
        if args.curriculum_strategy == "level_selector" and level_selector is not None and _rs_n >= 5:
            _lsel_sig_np = sig_vec_z[[2, 5, 6]] if args.level_selector_grpo_safe else sig_vec_z
            _lsel_sig = torch.tensor(_lsel_sig_np, dtype=torch.float32, device=device)
            _lsel_scores: dict = {}
            with torch.no_grad():
                for _lv in _level_seq:
                    _lsel_env = torch.tensor(_ENV_DESC[_lv], dtype=torch.float32, device=device)
                    _lsel_in  = torch.cat([_lsel_sig, _lsel_env]).unsqueeze(0)
                    if level_selector_mu is not None:
                        _lsel_in = (_lsel_in - level_selector_mu) / level_selector_sigma
                    _lsel_scores[_lv] = level_selector(_lsel_in).item()
            _lsel_level = max(_lsel_scores, key=_lsel_scores.get)
            for _lv, _sc in _lsel_scores.items():
                writer.add_scalar(f"level_selector/score_{_lv}", _sc, global_step)
            writer.add_scalar("level_selector/selected", _lsel_level, global_step)
            if _lsel_level != active_level:
                active_level = _lsel_level
                current_level = max(current_level, active_level)
                envs.close()
                envs = gym.vector.SyncVectorEnv(
                    [make_env(active_level, i, args.capture_video, run_name) for i in range(args.num_envs)]
                )
                next_obs, _ = envs.reset(seed=args.seed + iteration)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # gymnasium 1.x: episode stats are in infos["episode"] with mask infos["_episode"]
            if "episode" in infos:
                ep_mask = infos.get("_episode", np.ones(args.num_envs, dtype=bool))
                for i in range(args.num_envs):
                    if ep_mask[i]:
                        ep_return = float(infos["episode"]["r"][i])
                        ep_length = int(infos["episode"]["l"][i])
                        print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        rollout_returns.append(ep_return)
                        # return > 0 means goal was reached (step penalty alone can't yield > 0)
                        success = ep_return > 0
                        rollout_successes.append(success)
                        opt_path = int(infos["optimal_path_length"][i]) if "optimal_path_length" in infos else -1
                        if success and opt_path > 0:
                            rollout_path_efficiencies.append(opt_path / ep_length)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        iter_grad_norms, iter_actor_gnorms, iter_critic_gnorms = [], [], []
        iter_critic_losses, iter_entropies = [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # Compute separate actor/critic norms before joint clipping
                actor_gnorm  = nn.utils.clip_grad_norm_(agent.actor.parameters(),  float('inf')).item()
                critic_gnorm = nn.utils.clip_grad_norm_(agent.critic.parameters(), float('inf')).item()
                grad_norm    = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm).item()
                optimizer.step()
                iter_grad_norms.append(grad_norm)
                iter_actor_gnorms.append(actor_gnorm)
                iter_critic_gnorms.append(critic_gnorm)
                iter_critic_losses.append(v_loss.item())
                iter_entropies.append(entropy_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # --- Internal signal vector (computed once per iteration) ---
        gnorm_mean  = float(np.mean(iter_grad_norms))
        gnorm_var   = float(np.var(iter_grad_norms))
        closs_mean  = float(np.mean(iter_critic_losses))
        closs_var   = float(np.var(iter_critic_losses))
        ent_mean    = float(np.mean(iter_entropies))
        rolling_critic_buf.append(closs_mean)
        rolling_closs_mean = float(np.mean(rolling_critic_buf))

        # Rollout-level value and advantage stats
        value_mean = values.mean().item()
        value_std  = values.std().item()
        adv_mean   = b_advantages.mean().item()
        adv_std    = b_advantages.std().item()

        # Rolling window signals — append then compute slope for each tracked signal
        grad_norm_history.append(gnorm_mean)
        expl_var_history.append(explained_var)
        actor_gnorm_history.append(float(np.mean(iter_actor_gnorms)))
        critic_gnorm_history.append(float(np.mean(iter_critic_gnorms)))
        gnorm_slope        = linslope(grad_norm_history)
        expl_var_slope     = linslope(expl_var_history)
        actor_gnorm_slope  = linslope(actor_gnorm_history)
        critic_gnorm_slope = linslope(critic_gnorm_history)
        # param_delta: L2 weight change since last rollout (actor and critic separately)
        actor_params_now   = torch.cat([p.detach().cpu().flatten() for p in agent.actor.parameters()])
        critic_params_now  = torch.cat([p.detach().cpu().flatten() for p in agent.critic.parameters()])
        param_delta_actor  = (actor_params_now  - actor_params_snap).norm().item()
        param_delta_critic = (critic_params_now - critic_params_snap).norm().item()
        actor_params_snap, critic_params_snap = actor_params_now, critic_params_now
        param_delta_history.append(param_delta_actor)
        param_delta_slope  = linslope(param_delta_history)

        # 7-dim signal vector for D3 (assembled every rollout; negligible cost)
        # Order: [EV, EV_slope, adv_std, crit_gnorm_slope, crit_gnorm_mean, entropy, actor_gnorm_slope]
        sig_vec = np.array([
            explained_var, expl_var_slope, adv_std,
            critic_gnorm_slope, float(np.mean(iter_critic_gnorms)),
            ent_mean, actor_gnorm_slope,
        ], dtype=np.float32)

        # Welford running z-score (per-run online normalization, used by level_selector + domain_rand_log)
        _rs_n += 1
        _rs_d       = sig_vec.astype(np.float64) - _rs_mean
        _rs_mean   += _rs_d / _rs_n
        _rs_M2     += _rs_d * (sig_vec.astype(np.float64) - _rs_mean)
        _rs_std     = np.sqrt(np.maximum(_rs_M2 / max(_rs_n - 1, 1), 1e-12)).astype(np.float32)
        sig_vec_z   = ((sig_vec - _rs_mean.astype(np.float32)) / _rs_std).astype(np.float32)

        # Domain randomization data logging
        if args.domain_rand_log_path:
            _mean_ret = float(np.mean(rollout_returns))   if rollout_returns   else 0.0
            _mean_suc = float(np.mean(rollout_successes)) if rollout_successes else 0.0
            _dr_rows.append(np.concatenate([
                [iteration, active_level],
                sig_vec,                                       # raw  cols 2-8
                sig_vec_z,                                     # z    cols 9-15
                [explained_var, _ev_prev if not np.isnan(_ev_prev) else explained_var,
                 _mean_ret, _mean_suc],                        # cols 16-19
            ]).astype(np.float32))
        _ev_prev = explained_var

        # --- Level-conditioned gate: per-rollout online update + inference ---
        if lc_gate is not None:
            _grpo_idx = [2, 5, 6]  # adv_std, entropy_mean, actor_gnorm_slope
            _sig = sig_vec[_grpo_idx] if args.lc_gate_grpo_safe else sig_vec
            _env_desc = torch.tensor(_ENV_DESC[current_level], dtype=torch.float32, device=device)
            _gate_in  = torch.cat([torch.tensor(_sig, dtype=torch.float32, device=device), _env_desc])
            lc_buf.append((_gate_in.detach(), explained_var))
            lc_ev_hist.append(explained_var)
            if len(lc_buf) > args.lc_gate_k:
                _old_in, _old_ev = lc_buf.pop(0)
                _label = float(np.mean(lc_ev_hist[-args.lc_gate_k:]) > _old_ev)
                lc_ev_hist.pop(0)
                lc_gate_opt.zero_grad()
                _pred = lc_gate(_old_in.unsqueeze(0))
                _loss = lc_bce(_pred, torch.tensor([[_label]], dtype=torch.float32, device=device))
                _loss.backward()
                lc_gate_opt.step()
                lc_warmup_done = True
                writer.add_scalar("lc_gate/loss",  _loss.item(), global_step)
                writer.add_scalar("lc_gate/label", _label,       global_step)
            with torch.no_grad():
                lc_gate_prob = lc_gate(_gate_in.unsqueeze(0)).item() if lc_warmup_done else 0.0
            writer.add_scalar("lc_gate/prob", lc_gate_prob, global_step)

        # --- Offline gate: frozen inference only ---
        if offline_gate is not None:
            _grpo_idx = [2, 5, 6]
            _sig = sig_vec[_grpo_idx] if args.offline_gate_grpo_safe else sig_vec
            _env_desc = torch.tensor(_ENV_DESC[current_level], dtype=torch.float32, device=device)
            _gate_in  = torch.cat([torch.tensor(_sig, dtype=torch.float32, device=device), _env_desc])
            with torch.no_grad():
                if offline_gate_mu is not None:
                    _gate_in = (_gate_in - offline_gate_mu) / offline_gate_sigma
                offline_gate_prob = offline_gate(_gate_in.unsqueeze(0)).item()
            writer.add_scalar("offline_gate/prob", offline_gate_prob, global_step)

        # Episode-level stats accumulated during this rollout
        mean_return   = float(np.mean(rollout_returns))          if rollout_returns          else float('nan')
        success_rate  = float(np.mean(rollout_successes))        if rollout_successes        else float('nan')
        path_eff_mean = float(np.mean(rollout_path_efficiencies)) if rollout_path_efficiencies else float('nan')

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Internal physiological signals
        writer.add_scalar("internal_signals/grad_norm_mean",         gnorm_mean,        global_step)
        writer.add_scalar("internal_signals/grad_norm_var",          gnorm_var,         global_step)
        writer.add_scalar("internal_signals/critic_loss_mean",       closs_mean,        global_step)
        writer.add_scalar("internal_signals/critic_loss_var",        closs_var,         global_step)
        writer.add_scalar("internal_signals/entropy_mean",           ent_mean,          global_step)
        writer.add_scalar("internal_signals/rolling_critic_loss",    rolling_closs_mean, global_step)
        writer.add_scalar("internal_signals/actor_grad_norm_mean",   float(np.mean(iter_actor_gnorms)),  global_step)
        writer.add_scalar("internal_signals/critic_grad_norm_mean",  float(np.mean(iter_critic_gnorms)), global_step)
        writer.add_scalar("internal_signals/value_mean",             value_mean,        global_step)
        writer.add_scalar("internal_signals/value_std",              value_std,         global_step)
        writer.add_scalar("internal_signals/advantage_mean",         adv_mean,          global_step)
        writer.add_scalar("internal_signals/advantage_std",          adv_std,           global_step)
        writer.add_scalar("internal_signals/explained_variance",     explained_var,     global_step)
        writer.add_scalar("internal_signals/grad_norm_slope",        gnorm_slope,       global_step)
        writer.add_scalar("internal_signals/expl_var_slope",         expl_var_slope,    global_step)
        writer.add_scalar("internal_signals/param_delta_actor",      param_delta_actor,  global_step)
        writer.add_scalar("internal_signals/param_delta_critic",     param_delta_critic, global_step)
        writer.add_scalar("internal_signals/param_delta_slope",      param_delta_slope,  global_step)
        writer.add_scalar("internal_signals/actor_gnorm_slope",      actor_gnorm_slope,  global_step)
        writer.add_scalar("internal_signals/critic_gnorm_slope",     critic_gnorm_slope, global_step)

        # Episodic stats (only when episodes completed this rollout)
        if not np.isnan(mean_return):
            writer.add_scalar("charts/rollout_mean_return", mean_return,   global_step)
        if not np.isnan(success_rate):
            writer.add_scalar("charts/success_rate",        success_rate,  global_step)
        if not np.isnan(path_eff_mean):
            writer.add_scalar("charts/path_efficiency",     path_eff_mean, global_step)

        # Curriculum tracking
        writer.add_scalar("curriculum/level",                current_level,         global_step)
        writer.add_scalar("curriculum/active_level",         active_level,          global_step)
        writer.add_scalar("curriculum/steps_since_expansion", steps_since_expansion, global_step)
        steps_since_expansion += 1

        # --- Curriculum expansion ---
        if _level_seq_idx < len(_level_seq) - 1:
            should_expand = False
            if args.curriculum_strategy == "allopoietic":
                should_expand = (iteration % args.expand_every_n == 0)
            elif args.curriculum_strategy == "spdl":
                should_expand = (not np.isnan(mean_return) and mean_return > args.spdl_reward_threshold)
            elif args.curriculum_strategy == "domain_rand":
                should_expand = (iteration % args.expand_every_n == 0)
            elif args.curriculum_strategy == "heuristic":
                # Two gate families:
                #   slope-based  (_s): fire when |slope| < heuristic_eps, requires window full
                #   point-in-time: fire when value crosses per-signal threshold arg (no window needed)
                window_full = len(grad_norm_history) == W
                eps = args.heuristic_eps
                sig = args.heuristic_signal
                _s  = lambda slope: window_full and eps > 0 and abs(slope) < eps
                _lt = lambda val, thr: thr > 0 and val < thr   # expand when value drops below thr
                _gt = lambda val, thr: thr > 0 and val > thr   # expand when value rises above thr
                if   sig == "both":              should_expand = _s(gnorm_slope) and _s(expl_var_slope)
                elif sig == "or":                should_expand = _s(gnorm_slope)  or _s(expl_var_slope)
                elif sig == "gnorm":             should_expand = _s(gnorm_slope)
                elif sig == "expl_var":          should_expand = _s(expl_var_slope)
                elif sig == "actor_gnorm":       should_expand = _s(actor_gnorm_slope)
                elif sig == "critic_gnorm":      should_expand = _s(critic_gnorm_slope)
                elif sig == "param_delta_slope": should_expand = _s(param_delta_slope)
                elif sig == "param_delta":       should_expand = (
                    _lt(param_delta_actor, args.param_delta_eps) if args.actor_only_param_delta
                    else _lt(param_delta_actor, args.param_delta_eps) and _lt(param_delta_critic, args.param_delta_eps))
                elif sig == "adv_std":           should_expand = _lt(adv_std, args.adv_std_eps)
                elif sig == "entropy":           should_expand = _lt(ent_mean, args.entropy_eps)
                elif sig == "kl":                should_expand = _lt(approx_kl.item(), args.kl_eps)
                elif sig == "clipfrac":          should_expand = _lt(float(np.mean(clipfracs)), args.clipfrac_eps)
                elif sig == "ev_abs":            should_expand = _gt(explained_var, args.ev_abs_eps)
                elif sig == "crit_gnorm_abs":    should_expand = _lt(float(np.mean(iter_critic_gnorms)), args.crit_gnorm_abs_eps)
            elif args.curriculum_strategy == "homeostatic":
                pass  # placeholder — learned gate not yet implemented
            elif args.curriculum_strategy == "lc_gate":
                should_expand = lc_warmup_done and lc_gate_prob > args.lc_gate_thr
            elif args.curriculum_strategy == "offline_gate":
                # Mirror the window_full guard from the training label (steps_since_expansion >= W_MIN=5)
                should_expand = steps_since_expansion >= 5 and offline_gate_prob > args.offline_gate_thr

            if should_expand:
                _level_seq_idx += 1
                current_level = _level_seq[_level_seq_idx]
                active_level = current_level
                steps_since_expansion = 0
                lr_reset_iteration = iteration
                grad_norm_history.clear()   # reset slopes at level boundary
                expl_var_history.clear()
                actor_gnorm_history.clear()
                critic_gnorm_history.clear()
                param_delta_history.clear()
                if lc_gate is not None:     # reset circular buffer at level boundary
                    lc_buf.clear(); lc_ev_hist.clear(); lc_warmup_done = False
                envs.close()
                envs = gym.vector.SyncVectorEnv(
                    [make_env(current_level, i, args.capture_video, run_name) for i in range(args.num_envs)]
                )
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                print(f"*** Curriculum expanded to level {current_level} at iteration {iteration} ***")
                writer.add_scalar("curriculum/level", current_level, global_step)

    envs.close()
    writer.close()

    if args.domain_rand_log_path and _dr_rows:
        import os as _os5
        _os5.makedirs(_os5.path.dirname(_os5.path.abspath(args.domain_rand_log_path)), exist_ok=True)
        np.save(args.domain_rand_log_path, np.stack(_dr_rows))
        print(f"[domrand_log] {len(_dr_rows)} rows → {args.domain_rand_log_path}")