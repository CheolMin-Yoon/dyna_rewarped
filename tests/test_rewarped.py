"""Rewarped GPU 물리 전진·PyTorch 역전파 검사"""

import unittest

import torch
from rewarped.envs.dflex.ant import Ant


class RewarpedTest(unittest.TestCase):
    def test_cuda_physics_gradient(self):
        """두 환경·두 step의 관측·보상·action gradient 검사"""
        self.assertTrue(torch.cuda.is_available())
        env = Ant(num_envs=2, randomize=False, render=False, device="cuda:0")
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (2, 37))
        actions = torch.full((2, 8), 0.01, device="cuda:0", requires_grad=True)
        rewards = []
        for _ in range(2):
            obs, reward, done, extras = env.step(actions)
            self.assertEqual(tuple(reward.shape), (2,))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue(torch.isfinite(reward).all())
            rewards.append(reward)
        loss = -torch.stack(rewards).sum(dim=0).mean()
        loss.backward()
        torch.cuda.synchronize()
        self.assertIsNotNone(actions.grad)
        self.assertEqual(tuple(actions.grad.shape), (2, 8))
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(float(actions.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
