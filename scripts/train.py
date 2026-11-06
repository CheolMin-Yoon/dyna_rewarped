"""Stock Rewarped Ant·MLP·BPTT 조립과 학습 기록"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from world_model_skills import BPTT

from source.ant import make_policy, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
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

    from rewarped.envs.dflex.ant import Ant

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir or Path("logs/ant/bptt") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["output_dir"] = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env_config = {"episode_length": args.episode_length, "randomize": args.randomize, "seed": args.seed}
    env = Ant(num_envs=args.num_envs, device=args.device, render=False, no_grad=False, **env_config)
    policy_config = {
        "obs_dim": env.num_observations,
        "action_dim": env.num_actions,
        "hidden_dims": args.hidden_dims,
    }
    policy = make_policy(**policy_config).to(args.device)
    learner = BPTT(policy, gamma=args.gamma, learning_rate=args.lr, max_grad_norm=args.max_grad_norm)

    def save_checkpoint(iteration):
        """재생·분석용 정책과 학습 설정 저장"""
        torch.save(
            {
                "task": "ant",
                "algorithm": "bptt",
                "iteration": iteration,
                "policy_config": policy_config,
                "policy": policy.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
                "loss_convention": "reset_discount_mean_over_horizon_and_envs",
                "env_config": env_config,
                "train_config": config,
            },
            output_dir / f"checkpoint_{iteration:06d}.pt",
        )

    env.reset()
    save_checkpoint(0)
    episode_returns = torch.zeros(args.num_envs, device=args.device)
    episode_lengths = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "env_steps", "loss", "completed_episodes", "episode_return", "episode_length"])
        for iteration in range(1, args.iterations + 1):
            rewards, resets = rollout(policy, env, args.horizon)
            loss = learner.update(rewards, resets)
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
