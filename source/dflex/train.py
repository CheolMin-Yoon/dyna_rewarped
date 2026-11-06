"""Stock Rewarped task와 world_model_skills 모델·BPTT·SHAC 조립"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_skills.modules import RunningNormalization
from world_model_skills import BPTT, SHAC, CriticModel, PolicyModel

from source.dflex import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["humanoid", "ant", "allegro_hand"], default="humanoid")
    parser.add_argument("--algorithm", choices=["bptt", "shac"], default="bptt")
    parser.add_argument("--compile-losses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--activation", default="elu")
    parser.add_argument(
        "--norm-position",
        choices=["none", "last", "all", "last_after_activation", "all_after_activation"],
        default="none",
    )
    parser.add_argument("--orthogonal-gain", type=float)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalization-eps", type=float, default=1e-5)
    parser.add_argument("--normalization-initial-count", type=float, default=0.0)
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--critic-iterations", type=int, default=16)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--target-critic-alpha", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--optimizer-kwargs", type=json.loads)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--schedule-iterations", type=int)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setup-seed", type=int)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.episode_length is None:
        args.episode_length = 600 if args.task == "allegro_hand" else 1000
    if (
        min(
            args.num_envs,
            args.iterations,
            args.horizon,
            args.episode_length,
            args.save_interval,
            args.log_interval,
            *args.hidden_dims,
        )
        < 1
    ):
        parser.error("개수·길이·간격은 양수여야 함")
    if not 0 <= args.gamma <= 1 or not 0 < args.lr < float("inf") or not 0 < args.max_grad_norm < float("inf"):
        parser.error("gamma는 [0, 1], lr·max-grad-norm은 유한한 양수여야 함")
    if not 0 < args.init_std < float("inf"):
        parser.error("init-std는 유한한 양수여야 함")
    if args.optimizer_kwargs is not None and not isinstance(args.optimizer_kwargs, dict):
        parser.error("optimizer-kwargs는 JSON object여야 함")
    if args.schedule_iterations is not None and args.schedule_iterations < 1:
        parser.error("schedule-iterations는 양수여야 함")
    if not 0 <= args.min_lr < float("inf"):
        parser.error("min-lr은 유한한 음이 아닌 값이어야 함")
    if not 0 < args.normalization_eps < float("inf") or not 0 <= args.normalization_initial_count < float("inf"):
        parser.error("normalization-eps는 양수, initial-count는 음이 아닌 유한한 값이어야 함")
    if args.algorithm == "shac":
        if min(args.critic_iterations, args.num_mini_batches, *args.critic_hidden_dims) < 1:
            parser.error("critic 개수·크기는 양수여야 함")
        if args.num_mini_batches > args.num_envs * args.horizon:
            parser.error("num-mini-batches는 rollout 표본 수 이하여야 함")
        if not 0 <= args.lam <= 1 or not 0 <= args.target_critic_alpha <= 1:
            parser.error("lam·target-critic-alpha는 [0, 1]이어야 함")
        if not 0 < args.critic_lr < float("inf"):
            parser.error("critic-lr은 유한한 양수여야 함")

    from rewarped.envs.dflex.ant import Ant
    from rewarped.envs.dflex.humanoid import Humanoid

    setup_seed = args.seed if args.setup_seed is None else args.setup_seed
    torch.manual_seed(setup_seed)
    np.random.seed(setup_seed)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    output_dir = args.output_dir or Path("logs") / args.task / args.algorithm / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["output_dir"] = str(output_dir)
    config["torch_version"] = str(torch.__version__)
    config["precision"] = "fp32_tf32"
    config["model_compile_mode"] = "default"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env_config = {
        "episode_length": args.episode_length,
        "randomize": args.randomize,
        "seed": args.seed,
        "no_env_offset": True,
    }
    if args.task == "allegro_hand":
        from rewarped.envs.isaacgymenvs.allegro_hand import AllegroHand

        env_class = AllegroHand
    else:
        env_class = {"ant": Ant, "humanoid": Humanoid}[args.task]
    env = env_class(num_envs=args.num_envs, device=args.device, render=False, no_grad=False, **env_config)
    policy_config = {
        "obs_dim": env.num_observations,
        "action_dim": env.num_actions,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "init_std": args.init_std,
        "norm_position": args.norm_position,
        "orthogonal_gain": args.orthogonal_gain,
    }
    policy = PolicyModel(**policy_config).to(args.device)
    normalizer_config = {
        "input_dim": env.num_observations,
        "eps": args.normalization_eps,
        "initial_count": args.normalization_initial_count,
    }
    obs_normalizer = RunningNormalization(**normalizer_config).to(args.device) if args.normalize_obs else None
    if args.algorithm == "bptt":
        learner = BPTT(
            policy,
            gamma=args.gamma,
            learning_rate=args.lr,
            max_grad_norm=args.max_grad_norm,
            optimizer=args.optimizer,
            optimizer_kwargs=args.optimizer_kwargs,
        )
        optimizers = [(learner.optimizer, args.lr)]
    else:
        critic_config = {
            "obs_dim": env.num_observations,
            "hidden_dims": args.critic_hidden_dims,
            "activation": args.activation,
            "norm_position": args.norm_position,
            "orthogonal_gain": args.orthogonal_gain,
        }
        critic = CriticModel(**critic_config).to(args.device)
        learner = SHAC(
            policy,
            critic,
            gamma=args.gamma,
            lam=args.lam,
            policy_learning_rate=args.lr,
            critic_learning_rate=args.critic_lr,
            max_grad_norm=args.max_grad_norm,
            critic_iterations=args.critic_iterations,
            num_mini_batches=args.num_mini_batches,
            target_critic_alpha=args.target_critic_alpha,
            optimizer=args.optimizer,
            optimizer_kwargs=args.optimizer_kwargs,
        )
        optimizers = [(learner.policy_optimizer, args.lr), (learner.critic_optimizer, args.critic_lr)]
        critic.forward = torch.compile(critic.forward)
        learner.target_critic.forward = torch.compile(learner.target_critic.forward)
    policy.forward = torch.compile(policy.forward)
    if args.compile_losses:
        methods = ("loss",) if args.algorithm == "bptt" else ("loss", "critic_targets", "critic_loss")
        for name in methods:
            setattr(learner, name, torch.compile(getattr(learner, name)))

    def save_checkpoint(iteration):
        """재생·분석용 정책과 학습 설정 저장"""
        checkpoint = {
            "task": args.task,
            "algorithm": args.algorithm,
            "iteration": iteration,
            "policy_type": "PolicyModel",
            "policy_config": policy_config,
            "policy": policy.state_dict(),
            "loss_convention": "reset_discount_mean_over_horizon_and_envs",
            "env_config": env_config,
            "train_config": config,
            "obs_normalizer_config": normalizer_config if obs_normalizer is not None else None,
            "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer is not None else None,
        }
        if args.algorithm == "bptt":
            checkpoint["optimizer"] = learner.optimizer.state_dict()
        else:
            checkpoint.update(
                critic_config=critic_config,
                critic=critic.state_dict(),
                target_critic=learner.target_critic.state_dict(),
                policy_optimizer=learner.policy_optimizer.state_dict(),
                critic_optimizer=learner.critic_optimizer.state_dict(),
            )
        torch.save(checkpoint, output_dir / f"checkpoint_{iteration:06d}.pt")

    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
    env.reset()
    save_checkpoint(0)
    episode_returns = torch.zeros(args.num_envs, device=args.device)
    episode_lengths = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "iteration",
                "env_steps",
                "loss",
                "completed_episodes",
                "episode_return",
                "episode_length",
                "critic_loss",
                "policy_lr",
                "critic_lr",
            ]
        )
        for iteration in range(1, args.iterations + 1):
            if args.lr_schedule == "linear":
                progress = min((iteration - 1) / (args.schedule_iterations or args.iterations), 1.0)
                for optimizer, initial_lr in optimizers:
                    for group in optimizer.param_groups:
                        group["lr"] = initial_lr + (args.min_lr - initial_lr) * progress
            batch = rollout(
                policy, env, args.horizon, with_critic=args.algorithm == "shac", obs_normalizer=obs_normalizer
            )
            try:
                if args.algorithm == "bptt":
                    rewards, resets = batch
                    loss = learner.update(*batch)
                    critic_loss = None
                else:
                    rewards, _, resets, _, _ = batch
                    metrics = learner.update(*batch)
                    loss, critic_loss = metrics["policy_loss"], metrics["critic_loss"].item()
            except (FloatingPointError, RuntimeError):
                torch.save(
                    {"iteration": iteration, "batch": tuple(value.detach().cpu() for value in batch)},
                    output_dir / "failed_rollout.pt",
                )
                raise
            completed_returns, completed_lengths = [], []
            for reward, done in zip(rewards.detach(), resets.bool(), strict=True):
                episode_returns += reward
                episode_lengths += 1
                completed_returns.extend(episode_returns[done].tolist())
                completed_lengths.extend(episode_lengths[done].tolist())
                episode_returns[done] = 0
                episode_lengths[done] = 0
            mean_return = float(np.mean(completed_returns)) if completed_returns else float("nan")
            mean_length = float(np.mean(completed_lengths)) if completed_lengths else float("nan")
            writer.writerow(
                [
                    iteration,
                    iteration * args.horizon * args.num_envs,
                    loss.item(),
                    len(completed_returns),
                    mean_return,
                    mean_length,
                    critic_loss,
                    optimizers[0][0].param_groups[0]["lr"],
                    optimizers[1][0].param_groups[0]["lr"] if args.algorithm == "shac" else None,
                ]
            )
            stream.flush()
            if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.iterations:
                print(
                    f"iteration {iteration} | loss {loss.item():.6f} | episode return {mean_return:.3f}"
                    f" | completed {len(completed_returns)}",
                    flush=True,
                )
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                save_checkpoint(iteration)
    print(f"saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
