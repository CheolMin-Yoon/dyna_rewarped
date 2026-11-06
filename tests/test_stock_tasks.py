"""Stock task 선택과 checkpoint 기반 landscape 환경 연결 검사"""

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from scripts import plot_loss_landscape, train
from source.dflex import train as train_source


class StockTaskTest(unittest.TestCase):
    def test_train_selects_stock_tasks_and_episode_lengths(self):
        """Stock 생성자·task별 기본 episode 길이와 명시 override 확인"""
        for task, options, episode_length in (
            ("humanoid", [], 1000),
            ("ant", ["--task", "ant"], 1000),
            ("allegro_hand", ["--task", "allegro_hand"], 600),
            ("allegro_hand", ["--task", "allegro_hand", "--episode-length", "17"], 17),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                modules = {}
                constructors = {}
                for name, class_name, family in (
                    ("ant", "Ant", "dflex"),
                    ("humanoid", "Humanoid", "dflex"),
                    ("allegro_hand", "AllegroHand", "isaacgymenvs"),
                ):
                    module = types.ModuleType(f"rewarped.envs.{family}.{name}")
                    constructors[name] = Mock(side_effect=RuntimeError("환경 생성 경계"))
                    setattr(module, class_name, constructors[name])
                    modules[module.__name__] = module
                arguments = [
                    "train",
                    *options,
                    "--device",
                    "cpu",
                    "--num-envs",
                    "2",
                    "--output-dir",
                    str(Path(temporary) / "run"),
                ]
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments):
                    with self.assertRaisesRegex(RuntimeError, "환경 생성 경계"):
                        train.main()
                constructors[task].assert_called_once()
                for name, constructor in constructors.items():
                    if name != task:
                        constructor.assert_not_called()
                self.assertEqual(constructors[task].call_args.kwargs["episode_length"], episode_length)
                self.assertEqual(constructors[task].call_args.kwargs["num_envs"], 2)
                self.assertEqual(constructors[task].call_args.kwargs["device"], "cpu")
                self.assertTrue(constructors[task].call_args.kwargs["no_env_offset"])

    def test_loss_compile_option_preserves_model_compilation(self):
        """손실 compile 선택과 무관한 model compile 유지·측정 설정 기록"""
        for algorithm in ("bptt", "shac"):
            for enabled in (False, True):
                with self.subTest(algorithm=algorithm, enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                    env = types.SimpleNamespace(
                        num_observations=1,
                        num_actions=1,
                        reset=Mock(side_effect=RuntimeError("rollout 시작 경계")),
                    )
                    modules = {}
                    for name, class_name in (("ant", "Ant"), ("humanoid", "Humanoid")):
                        module = types.ModuleType(f"rewarped.envs.dflex.{name}")
                        setattr(module, class_name, Mock(return_value=env))
                        modules[module.__name__] = module
                    arguments = [
                        "train",
                        "--algorithm",
                        algorithm,
                        "--device",
                        "cpu",
                        "--num-envs",
                        "2",
                        "--horizon",
                        "2",
                        "--output-dir",
                        str(Path(temporary) / "run"),
                        "--compile-losses" if enabled else "--no-compile-losses",
                    ]
                    with (
                        patch.dict(sys.modules, modules),
                        patch.object(sys, "argv", arguments),
                        patch.object(
                            train_source.torch, "compile", side_effect=lambda function: function
                        ) as compile_call,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "rollout 시작 경계"):
                            train.main()
                    names = [call.args[0].__name__ for call in compile_call.call_args_list]
                    self.assertEqual(names.count("forward"), 1 if algorithm == "bptt" else 3)
                    expected_losses = [] if not enabled else ["loss"]
                    if enabled and algorithm == "shac":
                        expected_losses += ["critic_targets", "critic_loss"]
                    self.assertCountEqual([name for name in names if name != "forward"], expected_losses)
                    config = json.loads((Path(temporary) / "run" / "config.json").read_text())
                    self.assertEqual(config["compile_losses"], enabled)
                    self.assertEqual(config["precision"], "fp32_tf32")
                    self.assertEqual(config["model_compile_mode"], "default")
                    self.assertTrue(config["torch_version"])

    def test_failed_update_saves_detached_rollout_and_reraises(self):
        """갱신 실패 시 CPU rollout 저장과 원래 예외 유지"""
        for algorithm, error in (("bptt", RuntimeError("갱신 실패")), ("shac", FloatingPointError("비유한 미분"))):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temporary:
                output_dir = Path(temporary) / "run"
                env = types.SimpleNamespace(num_observations=1, num_actions=1, reset=Mock())
                modules = {}
                for name, class_name in (("ant", "Ant"), ("humanoid", "Humanoid")):
                    module = types.ModuleType(f"rewarped.envs.dflex.{name}")
                    setattr(module, class_name, Mock(return_value=env))
                    modules[module.__name__] = module
                rewards = torch.tensor([[1.0, 2.0]], requires_grad=True)
                resets = torch.zeros(1, 2)
                observations = torch.ones(1, 2, 1, requires_grad=True)
                batch = (
                    (rewards, resets)
                    if algorithm == "bptt"
                    else (rewards, resets, resets, observations, observations * 2)
                )
                arguments = [
                    "train",
                    "--algorithm",
                    algorithm,
                    "--device",
                    "cpu",
                    "--num-envs",
                    "2",
                    "--horizon",
                    "1",
                    "--iterations",
                    "1",
                    "--num-mini-batches",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ]
                learner_class = train_source.BPTT if algorithm == "bptt" else train_source.SHAC
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(sys, "argv", arguments),
                    patch.object(train_source.torch, "compile", side_effect=lambda function: function),
                    patch.object(train_source, "rollout", return_value=batch),
                    patch.object(learner_class, "update", side_effect=error),
                ):
                    with self.assertRaises(type(error)) as raised:
                        train.main()
                self.assertIs(raised.exception, error)
                saved = torch.load(output_dir / "failed_rollout.pt", weights_only=True)
                self.assertEqual(saved["iteration"], 1)
                torch.testing.assert_close(saved["batch"], batch, rtol=0, atol=0)
                self.assertTrue(all(value.device.type == "cpu" and not value.requires_grad for value in saved["batch"]))
                self.assertTrue((output_dir / "checkpoint_000000.pt").is_file())
                self.assertFalse((output_dir / "checkpoint_000001.pt").exists())

    def test_landscape_infers_stock_environment_from_checkpoint(self):
        """Stock checkpoint의 task와 episode 길이로 평가 환경 선택"""
        for task, class_name, family in (
            ("ant", "Ant", "dflex"),
            ("humanoid", "Humanoid", "dflex"),
            ("allegro_hand", "AllegroHand", "isaacgymenvs"),
        ):
            with self.subTest(task=task):
                checkpoint = {
                    "task": task,
                    "env_config": {"episode_length": 17, "no_env_offset": True, "seed": 42, "randomize": True},
                    "policy_config": {"obs_dim": 3, "action_dim": 2, "hidden_dims": [4]},
                }
                arguments = ["landscape", "--checkpoints", "model.pt", "--device", "cpu", "--grid-size", "3"]
                constructor = Mock(side_effect=RuntimeError("환경 생성 경계"))
                module = types.SimpleNamespace(**{class_name: constructor})
                with (
                    patch.object(sys, "argv", arguments),
                    patch.object(plot_loss_landscape, "load_policy", return_value=(None, checkpoint)),
                    patch.object(plot_loss_landscape.importlib, "import_module", return_value=module) as import_module,
                ):
                    with self.assertRaisesRegex(RuntimeError, "환경 생성 경계"):
                        plot_loss_landscape.main()
                import_module.assert_called_once_with(f"rewarped.envs.{family}.{task}")
                constructor.assert_called_once_with(
                    num_envs=9,
                    episode_length=17,
                    device="cpu",
                    no_env_offset=True,
                    seed=42,
                    randomize=False,
                    render=False,
                    no_grad=True,
                )
                self.assertTrue(checkpoint["env_config"]["randomize"])

    def test_landscape_rejects_checkpoints_from_different_tasks(self):
        """같은 모델 폭이어도 서로 다른 task의 공동 landscape 차단"""
        checkpoints = [
            (
                None,
                {
                    "task": task,
                    "env_config": {"episode_length": 17},
                    "policy_config": {"obs_dim": 3, "action_dim": 2, "hidden_dims": [4]},
                },
            )
            for task in ("ant", "humanoid")
        ]
        arguments = ["landscape", "--checkpoints", "ant.pt", "humanoid.pt", "--device", "cpu"]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(plot_loss_landscape, "load_policy", side_effect=checkpoints),
            patch.object(plot_loss_landscape, "make_env") as make_env,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                plot_loss_landscape.main()
        self.assertEqual(raised.exception.code, 2)
        make_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
