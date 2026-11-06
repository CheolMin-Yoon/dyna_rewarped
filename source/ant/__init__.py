"""Stock Rewarped Ant에서 사용하는 결정론적 MLP·checkpoint 복원"""

import torch
from torch import nn


def make_policy(obs_dim, action_dim, hidden_dims=(64, 64)):
    """관측 [N, O] → 행동 [N, A], 행동 제한은 stock Ant 소유"""
    layers = []
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(obs_dim, hidden_dim), nn.ELU()])
        obs_dim = hidden_dim
    layers.append(nn.Linear(obs_dim, action_dim))
    return nn.Sequential(*layers)


def rollout(policy, env, horizon):
    """구간 시작에서 simulator graph를 끊고 미분 가능한 보상과 실수 reset mask 반환"""
    obs = env.initialize_trajectory()
    rewards, resets = [], []
    for _ in range(horizon):
        obs, reward, done, _ = env.step(policy(obs))
        rewards.append(reward)
        resets.append(done.detach().bool().clone())
    rewards = torch.stack(rewards)
    return rewards, torch.stack(resets).to(rewards)


def load_policy(path, device):
    """설정·state_dict로 정책 복원, 평가 모드 정책과 checkpoint 반환"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["task"] != "ant":
        raise ValueError("Ant checkpoint가 필요함")
    policy = make_policy(**checkpoint["policy_config"]).to(device)
    policy.load_state_dict(checkpoint["policy"])
    return policy.eval(), checkpoint
