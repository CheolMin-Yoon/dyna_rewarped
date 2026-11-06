"""Stock Rewarped의 미분 rollout·정책 checkpoint 복원"""

from copy import deepcopy

import torch
from torch import nn
from torch_skills.modules import RunningNormalization
from world_model_skills import PolicyModel


def make_policy(obs_dim, action_dim, hidden_dims=(64, 64)):
    """기존 raw MLP checkpoint 복원용 정책"""
    layers = []
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(obs_dim, hidden_dim), nn.ELU()])
        obs_dim = hidden_dim
    layers.append(nn.Linear(obs_dim, action_dim))
    return nn.Sequential(*layers)


def rollout(policy, env, horizon, with_critic=False, obs_normalizer=None):
    """구간별 graph 분리, SHAC이면 reset 전 관측과 실패 종료까지 수집"""
    frozen_normalizer = deepcopy(obs_normalizer)
    obs = env.initialize_trajectory()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    if obs_normalizer is not None:
        obs_normalizer.update(obs)
        obs = frozen_normalizer(obs)
    rewards, resets = [], []
    observations, next_observations, terminations = [], [], []
    for _ in range(horizon):
        if with_critic:
            observations.append(obs.clone())
        obs, reward, done, extras = env.step(policy(obs))
        obs = obs["obs"] if isinstance(obs, dict) else obs
        rewards.append(reward)
        resets.append(done.detach().bool().clone())
        if with_critic:
            next_obs = extras["obs_before_reset"]
            next_obs = obs if next_obs is None else next_obs
            next_obs = next_obs["obs"] if isinstance(next_obs, dict) else next_obs
            if frozen_normalizer is not None:
                next_obs = frozen_normalizer(next_obs)
            next_observations.append(next_obs.clone())
            terminations.append(extras["terminated"].detach().clone())
        if obs_normalizer is not None:
            obs_normalizer.update(obs)
            obs = frozen_normalizer(obs)
    rewards = torch.stack(rewards)
    resets = torch.stack(resets).to(rewards)
    if with_critic:
        return (
            rewards,
            torch.stack(terminations).to(rewards),
            resets,
            torch.stack(observations),
            torch.stack(next_observations),
        )
    return rewards, resets


def load_policy(path, device):
    """설정·state_dict로 정책 복원, 평가 모드 정책과 checkpoint 반환"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["task"] not in ("ant", "humanoid", "allegro_hand"):
        raise ValueError("Ant·Humanoid·AllegroHand checkpoint가 필요함")
    policy_type = checkpoint.get("policy_type", "legacy_mlp")
    if policy_type == "PolicyModel":
        policy = PolicyModel(**checkpoint["policy_config"])
    elif policy_type == "legacy_mlp":
        policy = make_policy(**checkpoint["policy_config"])
    else:
        raise ValueError(f"지원하지 않는 정책 종류: {policy_type}")
    policy = policy.to(device)
    policy.load_state_dict(checkpoint["policy"])
    return policy.eval(), checkpoint


def load_normalizer(checkpoint, device):
    """관측 통계 복원, 정규화 없는 checkpoint는 None 반환"""
    if checkpoint.get("obs_normalizer") is None:
        return None
    normalizer = RunningNormalization(**checkpoint["obs_normalizer_config"]).to(device)
    normalizer.load_state_dict(checkpoint["obs_normalizer"])
    return normalizer.eval()
