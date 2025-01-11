import copy
import os
import random
from typing import Optional

import math
import torch
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict, field
import tyro
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from torch.distributions.categorical import Categorical
from open_spiel.python.algorithms import exploitability, expected_game_score
from tqdm.contrib.logging import logging_redirect_tqdm
import json
from torch.optim.optimizer import Optimizer

from envs import VectorEnv, PolicyInterface, GameInterface
import pickle
from training_env import PokerVectorEnv
from optimizer import AnchorAdamW
from agents.model import SimpleNet


def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@dataclass
class PPO_Args:
    game_name: str = "poker25"
    """name of the game"""
    alg_name: str = "appo"
    """logging name of the policy"""
    device: str = "cuda"
    """training device"""

    clip_coef: float = 0.02
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = np.inf
    """the target KL divergence threshold"""
    residual_coef: float = 0.01
    """penalty on invalid actions"""
    """ppo parameters"""

    adam_lr: float = 2e-4
    adam_beta1: float = 0.
    adam_beta2: float = 0.999
    adam_eps: float = 1e-5
    """adam parameters"""

    slingshot_coef: float = 0.
    """coefficient of the slingshot kl"""
    adam_weight_decay: float = 1.
    """adam adaptively weight decay"""
    mixing_anchor: bool = False
    """slow anchor movement"""
    averaging_anchor: bool = False
    """more stable anchor"""
    anchor_resets: int = 100
    """the number of anchor resets"""

    norm_adv: bool = True
    """Toggles advantages normalization"""
    norm_adv_beta: float = 0.

    adam_lr_min_ratio: float = 0.01
    """Toggle learning rate annealing for policy and value networks"""
    # ent_coef_min_ratio: float = 1.
    # """Toggle entropy coefficient rate annealing for policy and value networks"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    """ppo settings"""

    lbd: float = 0.99
    """reward function decay"""
    gae_lambda: float = 0.95
    """advantage function decay"""
    use_simple_advantage: bool = False
    """estimate advantage function in the simple way"""

    feature_dim: int = 256
    """NN feature dimension"""

    num_envs: int = 40
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    start_epoch: int = 0
    """continue training from previous"""
    total_steps: int = 2 * 10 ** 7
    """total number of steps"""

    num_minibatch: int = 4
    """the number of mini-batches"""
    log_freq = 500
    """logging frequency"""
    seed: int = 114514
    """seed for everything"""

    minibatch_size: Optional[int] = None
    """the mini-batch size (computed in runtime)"""
    batch_size: Optional[int] = None
    """the number of iterations (computed in runtime)"""
    total_iterations: Optional[int] = None
    """the number of iterations (computed in runtime)"""
    anchor_iterations: Optional[int] = None
    """anchor reset frequency (computed in runtime)"""

    def update_computations(self):
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatch
        self.total_iterations = self.total_steps // self.batch_size
        self.anchor_iterations = self.total_iterations // self.anchor_resets


def elementwise_kl(a, b, eps = 1e-9):
    # Add epsilon to avoid log(0)
    a_safe = a + eps
    b_safe = b + eps

    a_log = torch.log(a_safe)
    b_log = torch.log(b_safe)

    # KL(a[i] || b[i]) = sum over d: a[i, d] * (log(a[i, d]) - log(b[i, d]))
    kl_values = (a_safe * (a_log - b_log)).sum(dim = 1)
    return kl_values


def find_difference(args, baseline_args):
    changes = []
    args_dict = asdict(args)
    baseline_dict = asdict(baseline_args)

    for key, value in args_dict.items():
        if value != baseline_dict.get(key, None) and baseline_dict.get(key, None) is not None:
            changes.append(f"{key}={value}")

    return "-".join(changes) if changes else "baseline"


class PPO:
    def __init__(self, args: PPO_Args, envs: PokerVectorEnv, agent: SimpleNet, logbasedir: str = "runs/"):
        self.args = args
        self.device = args.device
        self.policy_name = find_difference(args, PPO_Args())

        self.agent = agent

        self.optimizer = AnchorAdamW(
            agent.parameters(),
            betas = (args.adam_beta1, args.adam_beta2),
            eps = args.adam_eps,
            lr = args.adam_lr,
            mixing_anchor = args.mixing_anchor,
            averaging_anchor = args.averaging_anchor,
            weight_decay = args.adam_weight_decay
        )

        self.envs = envs
        self.next_obs, self.current_player, self.legal_action = envs.reset()

        self.next_done = torch.zeros(envs.num_envs, device = self.device)
        self.next_reset = torch.zeros(envs.num_envs, device = self.device)

        self.total_iterations = 0
        self.total_steps = 0
        self.gradient_steps = 0

        self.ave_adv_mean = 0.
        self.ave_adv_std = 0.

        logdir = f"{logbasedir}/{self.policy_name}/"
        self.writer = SummaryWriter(logdir)
        self.writer.add_text(f"summary_{args.game_name}", f"<pre>{json.dumps(asdict(args), indent = 4)}</pre>",
                             global_step = 0)

        self.anchor = copy.deepcopy(self.agent)

    def iteration(self):
        args, envs = self.args, self.envs
        game_name = args.game_name
        total_steps = self.total_steps
        # frac = 1 - self.total_iterations / args.total_iterations
        frac = 1 - np.sin(self.total_iterations / args.total_iterations * np.pi / 2)
        lrnow = (frac + (1 - frac) * args.adam_lr_min_ratio) * args.adam_lr
        clip_coef_now = (frac + (1 - frac) * args.adam_lr_min_ratio) * args.clip_coef
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lrnow
        self.writer.add_scalar(f"charts_{game_name}/learning_rate", lrnow, total_steps)
        # ent_coef_now = (frac_cos + (1 - frac_cos) * args.ent_coef_min_ratio) * args.ent_coef
        # self.writer.add_scalar(f"charts_{game_name}/ent_coef", ent_coef_now, total_steps)

        if self.total_iterations % args.anchor_iterations == 0:
            self.anchor = copy.deepcopy(self.agent)
            self.optimizer.reset_anchor()
        self.total_iterations += 1

        ec_obs = torch.zeros([args.num_steps, envs.num_envs, envs.obs_encoded_size], device = self.device)
        actions = torch.zeros([args.num_steps, envs.num_envs], device = self.device)
        current_players = torch.zeros(args.num_steps, envs.num_envs, dtype = torch.int32, device = self.device)
        legal_actions_mask = torch.zeros(args.num_steps, envs.num_envs, envs.num_actions, device = self.device)
        logprobs = torch.zeros(args.num_steps, envs.num_envs, device = self.device)
        rewards = torch.zeros(args.num_steps, envs.num_envs, envs.num_players, device = self.device)
        dones = torch.zeros(args.num_steps, envs.num_envs, device = self.device)
        resets = torch.zeros(args.num_steps, envs.num_envs, device = self.device)
        values = torch.zeros(args.num_steps, envs.num_envs, device = self.device)

        with (torch.no_grad()):
            for step in range(args.num_steps):
                ec_obs[step] = envs.encode_obs(self.next_obs)
                dones[step] = self.next_done
                resets[step] = self.next_reset
                current_players[step] = self.current_player
                legal_actions_mask[step] = self.legal_action

                action, logprob, _, value, action_probs, _ = self.agent.get_logits_value(self.current_player, self.next_obs, self.legal_action)

                actions[step] = action
                logprobs[step] = logprob
                values[step] = value

                self.next_obs, reward, self.next_done, self.next_reset, self.current_player, self.legal_action = self.envs.step(action_probs, action)
                rewards[step] = reward

                self.total_steps += args.num_envs

            advantages = torch.zeros(args.num_steps, envs.num_envs, device = self.device)
            returns = torch.zeros(args.num_steps, envs.num_envs, device = self.device)
            lastgaelam = torch.zeros(envs.num_envs, envs.num_players, device = self.device)
            next_value = torch.zeros(envs.num_envs, envs.num_players, device = self.device)
            accumulated_rewards = torch.zeros(envs.num_envs, envs.num_players, device = self.device)

            _, _, _, next_value_player, _, _ = self.agent.get_logits_value(self.current_player, self.next_obs, self.legal_action)
            for player in range(envs.num_players):
                ids = (self.current_player == player)
                if torch.any(ids):
                    next_value[ids, player] = next_value_player[ids]
                    next_value[ids, 1 - player] = -next_value_player[ids]

            next_value *= (1. - self.next_reset).unsqueeze(-1).expand(-1, envs.num_players)
            for t in reversed(range(args.num_steps)):
                accumulated_rewards += rewards[t]
                for player in range(envs.num_players):
                    ids = (current_players[t] == player)
                    if torch.any(ids):
                        delta = next_value[ids, player] + accumulated_rewards[ids, player] - values[t, ids]
                        advantages[t, ids] = lastgaelam[ids, player] = delta + args.gae_lambda * lastgaelam[ids, player]
                        accumulated_rewards[ids, player] = 0
                        returns[t, ids] = advantages[t, ids] + values[t, ids]
                        next_value[ids, player] = values[t, ids]

                next_value *= (1. - resets[t]).unsqueeze(-1).expand(-1, envs.num_players)
                lastgaelam *= (1. - dones[t]).unsqueeze(-1).expand(-1, envs.num_players)
                accumulated_rewards *= (1. - resets[t]).unsqueeze(-1).expand(-1, envs.num_players) * args.lbd

        b_ec_obs = ec_obs.reshape([-1, envs.obs_encoded_size])
        b_logprobs = logprobs.reshape(-1)
        b_legal_actions = legal_actions_mask.reshape(-1, envs.num_actions)
        b_actions = actions.reshape(-1)
        b_current_player = current_players.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                self.gradient_steps += 1

                end = start + args.minibatch_size
                mb_inds = b_inds[start: end]

                b_obs = envs.decode_obs(b_ec_obs[mb_inds])

                _, newlogprob, entropy, newvalue, _, residual = self.agent.get_logits_value(
                    b_current_player[mb_inds], b_obs, b_legal_actions[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef_now).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    self.ave_adv_mean = self.ave_adv_mean * args.norm_adv_beta + mb_advantages.mean() * (
                            1 - args.norm_adv_beta)
                    self.ave_adv_std = self.ave_adv_std * args.norm_adv_beta + mb_advantages.std() * (
                            1 - args.norm_adv_beta)

                    ave_adv_mean_true = self.ave_adv_mean / (1 - args.norm_adv_beta ** self.gradient_steps)
                    ave_adv_std_true = self.ave_adv_std / (1 - args.norm_adv_beta ** self.gradient_steps)
                    mb_advantages = (mb_advantages - ave_adv_mean_true) / (ave_adv_std_true + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef_now, 1 + clip_coef_now)
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

                residual_loss = torch.mean(residual)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.residual_coef * residual_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                self.optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.writer.add_scalar(f"charts_{game_name}/value_loss", v_loss.item(), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/policy_loss", pg_loss.item(), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/entropy", entropy_loss.item(), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/old_approx_kl", old_approx_kl.item(), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/approx_kl", approx_kl.item(), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/clipfrac", np.mean(clipfracs), total_steps)
        self.writer.add_scalar(f"charts_{game_name}/explained_variance", explained_var, total_steps)
        self.writer.add_scalar(f"charts_{game_name}/residual", residual_loss.item(), total_steps)




def init_logger(console_handler = False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    filename = f"logs/logfile_{timestamp}.log"

    os.makedirs(os.path.dirname(filename), exist_ok = True)
    file_handler = logging.FileHandler(filename)
    file_format = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    if console_handler:
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def main(args, logger = None):
    if logger is None:
        logger = init_logger()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    alg_name = args.alg_name
    logger.info(f"Algorithm: {alg_name}")
    logger.info("Start!")
    game_name = args.game_name
    logger.info(f"Game: {game_name}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}")
    logger.info(f"Total iterations: {args.total_iterations}")

    envs = PokerVectorEnv(args.num_envs, args.device)
    logger.debug(f"Observation Encoded shape: {envs.obs_encoded_size}")
    logger.debug(f"Observation belief shape: {envs.obs_belief_shape}")
    logger.debug(f"Number of actions: {envs.num_actions}")
    logger.debug(f"Parameters: {json.dumps(asdict(args), indent = 4)}")
    model_dir = f"models/"
    os.makedirs(model_dir, exist_ok = True)
    result_dir = f"results/"
    os.makedirs(result_dir, exist_ok = True)
    # game_env = GameEnv()
    # game_env.load_open_spiel_game(pyspiel_env)
    agent = SimpleNet()
    agent = agent.to(device)
    ppo = PPO(args, envs, agent)
    policy_name = ppo.policy_name
    logger.info(f"Policy name: {policy_name}")

    for iteration in tqdm(range(args.total_iterations)):
        if iteration % args.log_freq == 0:
            agent = agent.to("cpu")
            torch.save(agent.state_dict(), f"{model_dir}/{policy_name}_{iteration}.pth")
            agent = agent.to(device)

        ppo.iteration()

    agent = agent.to("cpu")
    torch.save(agent.state_dict(), f"{model_dir}/{policy_name}.pth")


if __name__ == "__main__":
    args = tyro.cli(PPO_Args)
    args.update_computations()

    with logging_redirect_tqdm():
        main(args)
