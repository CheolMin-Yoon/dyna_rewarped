"""Checkpoint task의 stock Rewarped 환경 재생·첫 episode return 측정"""

import argparse
from pathlib import Path

import torch
from world_model_skills import PolicyModel

from source.dflex import load_normalizer, load_policy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--steps", type=int, help="최대 재생 step, 생략 시 episode 길이")
    parser.add_argument("--renderer", choices=["none", "usd", "opengl"], default="usd")
    parser.add_argument("--output-dir", type=Path, default=Path("media/play"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        args.num_envs < 1
        or (args.episode_length is not None and args.episode_length < 1)
        or (args.steps is not None and args.steps < 1)
    ):
        parser.error("환경 수·episode 길이·step 수는 양수여야 함")

    from rewarped.envs.dflex.ant import Ant
    from rewarped.envs.dflex.humanoid import Humanoid

    policy, checkpoint = load_policy(args.checkpoint, args.device)
    normalizer = load_normalizer(checkpoint, args.device)
    env_config = checkpoint["env_config"] | {"randomize": False}
    if args.episode_length is not None:
        env_config["episode_length"] = args.episode_length
    torch.manual_seed(env_config["seed"])
    if checkpoint["task"] == "allegro_hand":
        from rewarped.envs.isaacgymenvs.allegro_hand import AllegroHand

        env_class = AllegroHand
    else:
        env_class = {"ant": Ant, "humanoid": Humanoid}[checkpoint["task"]]
    env = env_class(
        num_envs=args.num_envs,
        device=args.device,
        no_grad=True,
        render=args.renderer != "none",
        render_mode=args.renderer,
        **env_config,
    )
    if args.renderer != "none":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        env.render_dir = str(args.output_dir)
    obs = env.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    returns = torch.zeros(args.num_envs, device=args.device)
    lengths = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
    alive = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
    policy_kwargs = {"deterministic": True} if isinstance(policy, PolicyModel) else {}
    try:
        for _ in range(args.steps or env.episode_length):
            policy_obs = normalizer(obs) if normalizer is not None else obs
            obs, reward, done, _ = env.step(policy(policy_obs, **policy_kwargs))
            obs = obs["obs"] if isinstance(obs, dict) else obs
            if not torch.isfinite(reward[alive]).all():
                raise FloatingPointError("재생 보상이 유한하지 않음")
            returns += torch.where(alive, reward, 0.0)
            lengths += alive
            alive &= ~done.bool()
            if not alive.any() or (args.renderer == "opengl" and env.renderer.has_exit):
                break
    finally:
        if env.renderer is not None:
            saved = env.renderer.save()
            if args.renderer == "usd":
                if saved is False:
                    raise OSError("USD 저장 실패")
                print(f"saved {args.output_dir / f'{env.sim_name}_{env.num_envs}.usd'}", flush=True)
    for index, (value, length, unfinished) in enumerate(
        zip(returns.tolist(), lengths.tolist(), alive.tolist(), strict=True)
    ):
        status = "미종료 구간" if unfinished else "완료 episode"
        print(f"env {index} | return {value:.3f} | steps {length} | {status}", flush=True)


if __name__ == "__main__":
    main()
