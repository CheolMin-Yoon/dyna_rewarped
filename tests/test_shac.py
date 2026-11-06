"""Library SHAC·정책·critic과 consumer rollout·평가 연결 검사"""

import io
import unittest

import torch
from torch import nn
from torch_skills.modules import RunningNormalization
from world_model_skills import SHAC, CriticModel, PolicyModel

from scripts.plot_loss_landscape import evaluate_returns
from source.dflex import load_normalizer, load_policy, rollout


class _ResetEnv:
    """실패·시간 제한·진행 중 환경을 함께 가진 미분 가능한 전이"""

    num_envs = 3
    episode_length = 2

    def __init__(self):
        self.starts = 0
        self.done = torch.zeros(3, dtype=torch.bool)
        self.terminated = torch.zeros(3, dtype=torch.bool)
        self.reset()

    def reset(self):
        self.steps = 0
        self.state = torch.tensor([[1.0], [2.0], [3.0]])
        return self.state

    def initialize_trajectory(self):
        self.starts += 1
        self.state = self.state.detach()
        return self.state

    def step(self, action):
        self.steps += 1
        next_obs = self.state + action
        reward = -next_obs.square().sum(-1)
        boundary = self.steps % 2 == 1
        self.done.copy_(torch.tensor([boundary, boundary, False]))
        self.terminated.copy_(torch.tensor([boundary, False, False]))
        terminal_obs = next_obs.clone() if boundary else None
        self.state = torch.where(self.done[:, None].clone(), torch.tensor([[10.0], [20.0], [30.0]]), next_obs)
        return (
            self.state,
            reward,
            self.done,
            {
                "terminated": self.terminated,
                "truncated": self.done & ~self.terminated,
                "obs_before_reset": terminal_obs,
            },
        )


class _DictResetEnv(_ResetEnv):
    """동일 전이의 dictionary 관측과 reset 전 관측 반환"""

    def reset(self):
        return {"obs": super().reset()}

    def initialize_trajectory(self):
        return {"obs": super().initialize_trajectory()}

    def step(self, action):
        obs, reward, done, extras = super().step(action)
        if extras["obs_before_reset"] is not None:
            extras["obs_before_reset"] = {"obs": extras["obs_before_reset"]}
        return {"obs": obs}, reward, done, extras


class SHACConsumerTest(unittest.TestCase):
    def test_dictionary_rollout_matches_tensor_observations_and_gradients(self):
        """Dictionary 관측의 BPTT·SHAC rollout·정규화·reset 전 미분 동등성 확인"""
        for normalize in (False, True):
            for with_critic in (False, True):
                with self.subTest(normalize=normalize, with_critic=with_critic):
                    outputs, gradients, statistics = [], [], []
                    for env_class in (_ResetEnv, _DictResetEnv):
                        policy = nn.Linear(1, 1, bias=False)
                        with torch.no_grad():
                            policy.weight.fill_(0.1)
                        normalizer = RunningNormalization(1, 1e-5) if normalize else None
                        if normalizer is not None:
                            normalizer.update(torch.tensor([[2.0], [6.0]]))
                        result = rollout(policy, env_class(), 2, with_critic=with_critic, obs_normalizer=normalizer)
                        outputs.append(result)
                        objective = result[-1][0].sum() if with_critic else result[0].sum()
                        gradient = torch.autograd.grad(objective, policy.weight)[0]
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0)
                        gradients.append(gradient)
                        if normalizer is not None:
                            statistics.append(normalizer.state_dict())
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                    if normalize:
                        torch.testing.assert_close(statistics[0], statistics[1], rtol=0, atol=0)
                    if with_critic and not normalize:
                        torch.testing.assert_close(outputs[1][-1][0, :, 0], torch.tensor([1.1, 2.2, 3.3]))
                        torch.testing.assert_close(gradients[1], torch.tensor([[6.0]]))

    def test_rollout_keeps_pre_reset_observations_and_gradients(self):
        """실패·timeout mask 복사와 reset 전 bootstrap 관측의 입력 미분 유지"""
        policy = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.weight.fill_(0.1)
        env = _ResetEnv()

        rewards, terminations, resets, obs, next_obs = rollout(policy, env, 2, with_critic=True)

        self.assertEqual(env.starts, 1)
        self.assertTrue(rewards.requires_grad)
        self.assertTrue(next_obs.requires_grad)
        self.assertEqual(terminations.dtype, rewards.dtype)
        self.assertEqual(resets.dtype, rewards.dtype)
        self.assertFalse(terminations.requires_grad)
        self.assertFalse(resets.requires_grad)
        torch.testing.assert_close(terminations, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(resets, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(obs[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(obs[1, :, 0], torch.tensor([10.0, 20.0, 3.3]))
        torch.testing.assert_close(next_obs[0, :, 0], torch.tensor([1.1, 2.2, 3.3]))
        torch.testing.assert_close(next_obs[1, :, 0], torch.tensor([11.0, 22.0, 3.63]))
        gradient = torch.autograd.grad(next_obs[0].sum(), policy.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[6.0]]))

    def test_shac_with_library_models_updates_across_windows(self):
        """실제 library 모델·learner의 두 window 갱신과 target 고정 확인"""
        torch.manual_seed(9)
        policy = PolicyModel(1, 1, (4,), "elu", 1.0)
        critic = CriticModel(1, (4,), "elu")
        learner = SHAC(policy, critic, 0.99, 0.95, 0.001, 0.001, 1.0, 1, 1, 0.4)
        env = _ResetEnv()
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
        target_before = [parameter.detach().clone() for parameter in learner.target_critic.parameters()]

        for _ in range(2):
            metrics = learner.update(*rollout(policy, env, 2, with_critic=True))
            for value in metrics.values():
                self.assertTrue(torch.isfinite(value))
                self.assertFalse(value.requires_grad)

        self.assertEqual(env.starts, 2)
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(policy.parameters(), before, strict=True)))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(critic.parameters(), critic_before, strict=True)))
        self.assertTrue(
            any(
                not torch.equal(p, old)
                for p, old in zip(learner.target_critic.parameters(), target_before, strict=True)
            )
        )
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in learner.target_critic.parameters()))

    def test_rollout_uses_frozen_statistics_and_updates_raw_observations(self):
        """구간 시작 통계 고정·raw 관측 통계 누적과 정규화 bootstrap 미분 유지"""
        policy = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.weight.fill_(0.1)
        normalizer = RunningNormalization(1, 1e-5)
        history = torch.tensor([[2.0], [6.0]])
        normalizer.update(history)
        mean, scale = normalizer.mean.clone(), (normalizer.var + normalizer.eps).sqrt()
        raw_start = torch.tensor([[1.0], [2.0], [3.0]])
        expected_obs0 = (raw_start - mean) / scale
        raw_next0 = raw_start + 0.1 * expected_obs0
        raw_obs1 = torch.cat((torch.tensor([[10.0], [20.0]]), raw_next0[2:]))
        expected_obs1 = (raw_obs1 - mean) / scale
        raw_next1 = raw_obs1 + 0.1 * expected_obs1
        env = _ResetEnv()

        _, _, _, obs, next_obs = rollout(policy, env, 2, with_critic=True, obs_normalizer=normalizer)

        torch.testing.assert_close(obs, torch.stack((expected_obs0, expected_obs1)))
        torch.testing.assert_close(next_obs, (torch.stack((raw_next0, raw_next1)) - mean) / scale)
        torch.testing.assert_close(env.state, raw_next1)
        gradient = torch.autograd.grad(next_obs[0].sum(), policy.weight)[0]
        torch.testing.assert_close(gradient, (expected_obs0.sum() / scale).reshape_as(gradient))
        expected_normalizer = RunningNormalization(1, 1e-5)
        expected_normalizer.update(torch.cat((history, raw_start, raw_obs1, raw_next1)))
        torch.testing.assert_close(normalizer.state_dict(), expected_normalizer.state_dict())
        self.assertEqual(normalizer.count.item(), len(history) + 3 * env.num_envs)

    def test_library_policy_checkpoint_restores_deterministic_actions(self):
        """PolicyModel 설정·가중치 복원과 noise 없는 평가 행동 일치"""
        config = {"obs_dim": 1, "action_dim": 1, "hidden_dims": (4,), "activation": "elu", "init_std": 1.0}
        policy = PolicyModel(**config)
        observations = torch.tensor([[1.0], [2.0], [3.0]])
        expected = policy(observations, deterministic=True)
        buffer = io.BytesIO()
        torch.save(
            {
                "task": "allegro_hand",
                "policy_type": "PolicyModel",
                "policy_config": config,
                "policy": policy.state_dict(),
            },
            buffer,
        )
        buffer.seek(0)

        restored, checkpoint = load_policy(buffer, "cpu")

        self.assertIsInstance(restored, PolicyModel)
        self.assertEqual(checkpoint["policy_type"], "PolicyModel")
        self.assertEqual(checkpoint["task"], "allegro_hand")
        self.assertFalse(restored.training)
        torch.testing.assert_close(restored(observations, deterministic=True), expected, rtol=0, atol=0)

    def test_landscape_policy_evaluation_is_deterministic(self):
        """확률 정책의 landscape 평가에서 난수 소비 없는 동일 return"""
        policy = PolicyModel(1, 1, (4,), "elu", 1.0).eval()
        params = {
            name: value.detach()[None].expand(3, *value.shape).clone() for name, value in policy.named_parameters()
        }
        env = _ResetEnv()
        rng = torch.get_rng_state()

        first = evaluate_returns(env, policy, params, dict(policy.named_buffers()))
        second = evaluate_returns(env, policy, params, dict(policy.named_buffers()))

        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_normalizer_checkpoint_restores_statistics_and_actions(self):
        """정규화 설정·통계 복원과 결정론 행동 일치, 기존 checkpoint 호환"""
        policy_config = {"obs_dim": 1, "action_dim": 1, "hidden_dims": (4,), "activation": "elu", "init_std": 1.0}
        normalizer_config = {"input_dim": 1, "eps": 1e-5, "initial_count": 1e-4}
        policy = PolicyModel(**policy_config)
        normalizer = RunningNormalization(**normalizer_config)
        observations = torch.tensor([[1.0], [2.0], [3.0]])
        normalizer.update(observations)
        expected = policy(normalizer(observations), deterministic=True)
        buffer = io.BytesIO()
        torch.save(
            {
                "task": "humanoid",
                "policy_type": "PolicyModel",
                "policy_config": policy_config,
                "policy": policy.state_dict(),
                "obs_normalizer_config": normalizer_config,
                "obs_normalizer": normalizer.state_dict(),
            },
            buffer,
        )
        buffer.seek(0)

        restored_policy, checkpoint = load_policy(buffer, "cpu")
        restored = load_normalizer(checkpoint, "cpu")

        self.assertIsNone(load_normalizer({}, "cpu"))
        self.assertFalse(restored.training)
        self.assertEqual(restored.eps, normalizer.eps)
        torch.testing.assert_close(restored.state_dict(), normalizer.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(
            restored_policy(restored(observations), deterministic=True), expected, rtol=0, atol=0
        )

    def test_normalized_landscape_preserves_statistics_and_raw_physics(self):
        """정규화 정책 평가 return·난수 미소비·통계 동결과 raw 물리 관측 유지"""
        policy = PolicyModel(1, 1, (4,), "elu", 1.0).eval()
        normalizer = RunningNormalization(1, 1e-5)
        normalizer.update(torch.tensor([[2.0], [6.0]]))
        before = {key: value.clone() for key, value in normalizer.state_dict().items()}
        params = {
            name: value.detach()[None].expand(3, *value.shape).clone() for name, value in policy.named_parameters()
        }
        reference_env = _ResetEnv()
        obs = reference_env.reset()
        expected = torch.zeros(3)
        alive = torch.ones(3, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(reference_env.episode_length):
                action = policy(normalizer(obs), deterministic=True).clamp(-1.0, 1.0)
                obs, reward, done, _ = reference_env.step(action)
                expected += torch.where(alive, reward, 0.0)
                alive &= ~done
        env = _ResetEnv()
        rng = torch.get_rng_state()

        first = evaluate_returns(env, policy, params, dict(policy.named_buffers()), normalizer)
        second = evaluate_returns(env, policy, params, dict(policy.named_buffers()), normalizer)

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(env.state, reference_env.state)
        torch.testing.assert_close(normalizer.state_dict(), before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
