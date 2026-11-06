"""Ant rollout의 graph 경계·종료 snapshot·정책 checkpoint 검사"""

import io
import unittest

import torch
from torch import nn
from world_model_skills import BPTT

from source.ant import load_policy, make_policy, rollout


class _ToyEnv:
    """해석 가능한 전이 x′ = x + a와 제곱 비용"""

    def __init__(self):
        self.state = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
        self.starts = 0
        self.steps = 0
        self.done = torch.tensor([False, True, False])

    def initialize_trajectory(self):
        self.starts += 1
        self.state = self.state.detach()
        return self.state

    def step(self, action):
        self.steps += 1
        self.state = self.state + action
        reward = -(self.state.square() + 0.1 * action.square()).sum(dim=-1)
        self.done.logical_not_()
        return self.state, reward, self.done, {}


class _LinearPolicy(nn.Module):
    """스칼라 parameter의 선형 행동 a = θx"""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        self.calls = 0

    def forward(self, obs):
        self.calls += 1
        return self.theta * obs


class BPTTTest(unittest.TestCase):
    def test_rollout_preserves_graph_and_copies_done_buffers(self):
        """보상 미분 유지·재사용 done buffer의 시점별 복사·실수 reset 변환"""
        policy, env = _LinearPolicy(), _ToyEnv()

        rewards, resets = rollout(policy, env, 2)

        self.assertEqual((env.starts, env.steps, policy.calls), (1, 2, 2))
        self.assertTrue(rewards.requires_grad)
        self.assertFalse(resets.requires_grad)
        self.assertEqual(resets.dtype, rewards.dtype)
        self.assertEqual(resets.device, rewards.device)
        torch.testing.assert_close(
            rewards,
            torch.tensor([[-0.644, -0.161, -2.576], [-0.41216, -0.10304, -1.64864]], dtype=torch.float64),
        )
        torch.testing.assert_close(resets, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64))
        env.done.fill_(False)
        self.assertEqual(resets.sum().item(), 3.0)
        rewards.sum().backward()
        self.assertIsNotNone(policy.theta.grad)
        self.assertTrue(torch.isfinite(policy.theta.grad))
        self.assertNotEqual(policy.theta.grad.item(), 0.0)

    def test_successive_library_updates_preserve_state_and_cut_graph(self):
        """library learner와 두 window 연결 시 물리값 유지·이전 graph 분리"""
        policy, env = _LinearPolicy(), _ToyEnv()
        learner = BPTT(policy, gamma=0.99, learning_rate=0.01, max_grad_norm=float("inf"))
        rewards, resets = rollout(policy, env, 2)
        learner.update(rewards, resets)
        state = env.state.detach().clone()
        theta = policy.theta.detach().clone()

        rewards, resets = rollout(policy, env, 2)
        loss = learner.update(rewards, resets)

        self.assertEqual((env.starts, env.steps, policy.calls), (2, 4, 4))
        self.assertFalse(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(policy.theta.grad))
        self.assertFalse(torch.equal(policy.theta, theta))
        torch.testing.assert_close(env.state, state * (1 + theta).square())

    def test_deterministic_policy_gradients_and_checkpoint(self):
        """결정론적 MLP의 경사·기존 checkpoint 복원·vmap 일치"""
        config = {"obs_dim": 3, "action_dim": 2, "hidden_dims": (4,)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            policy = make_policy(**config)
            obs = torch.tensor([[0.2, -0.3, 0.7], [-0.4, 0.1, 0.5]])
            actions = policy(obs)

        self.assertEqual(actions.shape, (2, 2))
        self.assertTrue(torch.isfinite(actions).all())
        torch.testing.assert_close(actions, policy(obs), rtol=0, atol=0)
        actions.square().sum().backward()
        parameters = list(policy.parameters())
        self.assertTrue(parameters)
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(any(parameter.grad.abs().max().item() > 0 for parameter in parameters))

        checkpoint = {"task": "ant", "policy_config": config, "policy": policy.state_dict()}
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        with torch.random.fork_rng(devices=[]):
            restored, checkpoint = load_policy(buffer, "cpu")
        self.assertFalse(restored.training)
        self.assertEqual(checkpoint["policy_config"], config)
        torch.testing.assert_close(restored(obs), actions.detach(), rtol=0, atol=0)
        torch.testing.assert_close(torch.vmap(restored)(obs), restored(obs))


if __name__ == "__main__":
    unittest.main()
