"""학습한 정책 parameter 주변 loss landscape를 Rewarped env에서 평가해 SAPO 논문 형식 3D 그림·원값 저장

방향: Li et al. 2018 filter 정규화 무작위 방향 2개, 1차원 parameter(bias·norm·log_std)는 0
값: SAPO 부록 F.7처럼 할인 없는 episode 보상, 높이·색은 symlog(−return)
그림: 정책마다 축 없는 3D surface 한 칸, 빨강(높은 loss)–노랑–파랑(낮은 loss) 색과 빛 음영, 칸마다 색 범위 따로
checkpoint: scripts.train의 모델 설정·state_dict, policy(obs [N, O]) → 결정론 action [N, A]
policy forward는 torch.func.vmap 가능해야 함(제자리 연산·.item() 없음)
"""

import argparse
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource, Normalize
from torch.func import functional_call

from source.ant import load_policy


def make_env(spec, num_envs, episode_length, device):
    """'module:Class' 이름의 Rewarped env를 미분 없이 생성"""
    module_name, class_name = spec.split(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    kwargs = {"num_envs": num_envs, "device": device, "render": False, "no_grad": True, "randomize": False}
    if episode_length is not None:
        kwargs["episode_length"] = episode_length
    return env_class(**kwargs)


def make_direction(params, generator):
    """
    filter 정규화 무작위 방향

    2차원 이상 parameter는 첫 축(출력 unit)마다 θ*의 norm에 맞춤, 1차원 parameter는 0
    """
    direction = {}
    for name, value in params.items():
        noise = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
        if value.dim() <= 1:
            direction[name] = torch.zeros_like(value)
            continue
        value_norm = value.flatten(1).norm(dim=1)
        noise_norm = noise.flatten(1).norm(dim=1)
        scale = value_norm / (noise_norm + 1e-10)
        direction[name] = noise * scale.view(-1, *([1] * (value.dim() - 1)))
    return direction


def make_grid_params(params, generator, coeff_a, coeff_b):
    """θ* + α·d₁ + β·d₂를 grid 점마다 쌓은 name → [G, ...]"""
    direction_a, direction_b = make_direction(params, generator), make_direction(params, generator)
    return {
        key: value[None]
        + coeff_a.view(-1, *([1] * value.dim())) * direction_a[key][None]
        + coeff_b.view(-1, *([1] * value.dim())) * direction_b[key][None]
        for key, value in params.items()
    }


@torch.no_grad()
def evaluate_returns(env, policy, grid_params, buffers):
    """
    env i는 grid 점 i의 parameter로 한 episode, 첫 종료 전 보상만 합산

    grid_params: name → [G, ...], 반환 [G], 물리 발산(보상 비유한)한 점은 NaN
    """

    def act(params, obs):
        return functional_call(policy, (params, buffers), (obs[None],))[0]

    batched_act = torch.vmap(act)
    obs = env.reset()
    returns = torch.zeros(env.num_envs, device=obs.device)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=obs.device)
    diverged = torch.zeros_like(alive)
    for _ in range(env.episode_length):
        actions = batched_act(grid_params, torch.nan_to_num(obs)).clamp(-1.0, 1.0)
        obs, reward, done, _ = env.step(actions)
        # 끝난 env의 NaN 보상은 곱하기 0으로 지워지지 않음 → where로 제외
        diverged |= alive & ~torch.isfinite(reward)
        alive &= ~diverged
        returns += torch.where(alive, reward, torch.zeros_like(reward))
        alive &= ~done.bool()
        if not alive.any():
            break
    return torch.where(diverged, torch.full_like(returns, float("nan")), returns)


def symlog(x):
    """sign(x)·log(1 + |x|)"""
    return np.sign(x) * np.log1p(np.abs(x))


def plot_landscapes(alphas, losses, labels, caption, path, elevation, azimuth):
    """정책마다 축 없는 3D surface 한 칸, 색은 칸마다 최솟값(파랑)–최댓값(빨강), 발산한 점은 최댓값으로"""
    ink, surface = "#1f1f1f", "#ffffff"
    colormap = plt.get_cmap("RdYlBu_r")
    light = LightSource(azdeg=315, altdeg=45)
    grid_a, grid_b = np.meshgrid(alphas, alphas, indexing="ij")

    figure = plt.figure(figsize=(4.4 * len(losses), 4.2), facecolor=surface)
    for index, (loss, label) in enumerate(zip(losses, labels, strict=True)):
        z = symlog(np.where(np.isfinite(loss), loss, np.nanmax(loss)))
        shaded = light.shade(z, cmap=colormap, norm=Normalize(z.min(), z.max()), blend_mode="soft")
        axis = figure.add_subplot(1, len(losses), index + 1, projection="3d", facecolor=surface)
        axis.plot_surface(grid_a, grid_b, z, facecolors=shaded, rstride=1, cstride=1, linewidth=0, shade=False)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_box_aspect((1.0, 1.0, 0.7), zoom=1.15)
        axis.set_axis_off()
        axis.set_title(label, y=-0.12, color=ink, fontsize=13)
    figure.text(0.5, 0.01, caption, ha="center", color=ink, fontsize=11)
    figure.subplots_adjust(left=0.02, right=0.98, top=1.0, bottom=0.14, wspace=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="정책마다 한 칸")
    parser.add_argument("--labels", nargs="+", default=None, help="칸 이름, 생략 시 checkpoint 이름")
    parser.add_argument("--env", default="rewarped.envs.dflex.ant:Ant", help="'module:Class'")
    parser.add_argument("--name", default="loss_landscape", help="그림·원값 파일 이름")
    parser.add_argument("--grid-size", type=int, default=31, help="축마다 점 수, grid 점은 그 제곱")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="한 번에 평가하는 grid 점(env) 수, Ant는 약 200 이상에서 일부 env가 첫 step부터 NaN",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="alpha·beta 범위 [-radius, radius]")
    parser.add_argument("--episode-length", type=int, default=None, help="공통 평가 길이, 생략 시 checkpoint 설정")
    parser.add_argument("--seed", type=int, default=0, help="방향 난수 seed, 정책마다 같은 seed")
    parser.add_argument("--elevation", type=float, default=30.0, help="시점 높이각(도)")
    parser.add_argument("--azimuth", type=float, default=-60.0, help="시점 방위각(도)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    labels = args.labels or [path.stem for path in args.checkpoints]
    if len(labels) != len(args.checkpoints) or len(set(labels)) != len(labels):
        parser.error("checkpoint마다 서로 다른 label 하나가 필요함")
    if args.grid_size < 3 or args.grid_size % 2 == 0 or not 0 < args.radius < float("inf") or args.batch_size < 1:
        parser.error("grid-size는 3 이상의 홀수, radius는 유한한 양수, batch-size는 양수여야 함")
    if args.episode_length is not None and args.episode_length < 1:
        parser.error("episode-length는 양수여야 함")
    policies = [load_policy(path, args.device) for path in args.checkpoints]
    if args.episode_length is None:
        lengths = {checkpoint["env_config"]["episode_length"] for _, checkpoint in policies}
        if len(lengths) != 1:
            parser.error("checkpoint의 episode 길이가 다름 → 공통 --episode-length 지정 필요")
        args.episode_length = lengths.pop()
    if any(checkpoint["policy_config"] != policies[0][1]["policy_config"] for _, checkpoint in policies):
        parser.error("같은 모델 구조의 checkpoint가 필요함")

    alphas = torch.linspace(-args.radius, args.radius, args.grid_size, device=args.device)
    coeff_a, coeff_b = torch.meshgrid(alphas, alphas, indexing="ij")
    coeff_a, coeff_b = coeff_a.reshape(-1), coeff_b.reshape(-1)
    num_points = coeff_a.numel()
    batch_size = min(args.batch_size, num_points)
    env = make_env(args.env, batch_size, args.episode_length, args.device)
    alphas_np = alphas.cpu().numpy()
    center = args.grid_size // 2

    losses = []
    for (policy, checkpoint), checkpoint_path, label in zip(policies, args.checkpoints, labels, strict=True):
        params = {key: value.detach() for key, value in policy.named_parameters()}
        buffers = {key: value.detach() for key, value in policy.named_buffers()}
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        grid_params = make_grid_params(params, generator, coeff_a, coeff_b)
        chunks = []
        for start in range(0, num_points, batch_size):
            # 마지막 묶음은 첫 점으로 채워 env 수를 고정, 채운 자리는 버림
            ids = torch.arange(start, start + batch_size, device=args.device).clamp(max=num_points - 1)
            chunk_params = {key: value[ids] for key, value in grid_params.items()}
            chunks.append(evaluate_returns(env, policy, chunk_params, buffers)[: num_points - start])
        returns = torch.cat(chunks).view(args.grid_size, args.grid_size).cpu().numpy()
        if not np.isfinite(returns).any():
            raise FloatingPointError(f"{label}: 모든 grid 점의 보상 발산")
        center_return = returns[center, center] if args.grid_size % 2 else float("nan")
        diverged = int(np.isnan(returns).sum())
        print(
            f"{label} | trained {center_return:.2f} | min {np.nanmin(returns):.2f} | max {np.nanmax(returns):.2f}"
            f" | diverged {diverged}/{returns.size}",
            flush=True,
        )

        data_path = root / "logs" / "loss_landscape" / f"{args.name}_{label}.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_path,
            alphas=alphas_np,
            betas=alphas_np,
            returns=returns,
            seed=args.seed,
            env=args.env,
            episode_length=args.episode_length,
            checkpoint=str(checkpoint_path.resolve()),
            iteration=checkpoint["iteration"],
        )
        losses.append(-returns)

    caption = f"{args.env.split(':')[-1]} | symlog(-episode return) | max {args.episode_length} steps"
    figure_path = root / "media" / "loss_landscape" / f"{args.name}.png"
    plot_landscapes(alphas_np, losses, labels, caption, figure_path, args.elevation, args.azimuth)
    print(f"saved {figure_path}", flush=True)


if __name__ == "__main__":
    main()
