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
    """curriculum expansion strategy: allopoietic | spdl | homeostatic"""
    expand_every_n: int = 50
    """allopoietic: expand every N iterations (ignored if expand_every_n_episodes > 0)"""
    spdl_reward_threshold: float = 0.7
    """spdl: expand when mean episodic return over last rollout exceeds this"""
    max_level: int = 3
    """maximum curriculum level (0-3 for static walls, 4 adds dynamic objects)"""
    start_level: int = 0
    """curriculum level to begin training at"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
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
    current_level = args.start_level
    envs = gym.vector.SyncVectorEnv(
        [make_env(current_level, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    lr_reset_iteration = 1  # iteration from which current LR annealing started
    rolling_critic_buf = deque(maxlen=10)
    rollout_returns = []
    rollout_successes = []
    rollout_path_efficiencies = []

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - lr_reset_iteration) / args.num_iterations
            frac = max(frac, 0.0)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_returns.clear()
        rollout_successes.clear()
        rollout_path_efficiencies.clear()

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

        # Episodic stats (only when episodes completed this rollout)
        if not np.isnan(mean_return):
            writer.add_scalar("charts/rollout_mean_return", mean_return,   global_step)
        if not np.isnan(success_rate):
            writer.add_scalar("charts/success_rate",        success_rate,  global_step)
        if not np.isnan(path_eff_mean):
            writer.add_scalar("charts/path_efficiency",     path_eff_mean, global_step)

        # Curriculum tracking
        writer.add_scalar("curriculum/level",                current_level,         global_step)
        writer.add_scalar("curriculum/steps_since_expansion", steps_since_expansion, global_step)
        steps_since_expansion += 1

        # --- Curriculum expansion ---
        if current_level < args.max_level:
            should_expand = False
            if args.curriculum_strategy == "allopoietic":
                should_expand = (iteration % args.expand_every_n == 0)
            elif args.curriculum_strategy == "spdl":
                should_expand = (not np.isnan(mean_return) and mean_return > args.spdl_reward_threshold)
            elif args.curriculum_strategy == "homeostatic":
                pass  # placeholder — gate not yet implemented

            if should_expand:
                current_level += 1
                steps_since_expansion = 0
                lr_reset_iteration = iteration
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