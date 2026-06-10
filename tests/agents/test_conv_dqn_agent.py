"""Tests pour ConvDQNAgent V3.6 (wrap V2-W MW_IA)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("torch", reason="suite complete : requiert torch")
pytest.importorskip("mw_ia", reason="suite complete : requiert le repo sibling MW_IA")

from mw_ia.config import ConvDQNConfig

from aetherlife.agents.conv_dqn_agent import ConvDQNAgent
from aetherlife.world.seasonal_grid import SeasonalConfig, SeasonalMultiAgentConfig
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


@pytest.fixture
def tiny_env() -> SoloSeasonalEnv:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, initial_food_density=0.1,
        food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=10),
    )
    return SoloSeasonalEnv(cfg)


@pytest.fixture
def tiny_conv_cfg() -> ConvDQNConfig:
    return ConvDQNConfig(
        conv_channels=(8,), kernel_size=3, padding=1, fc_hidden=16,
        batch_size=4, replay_capacity=64, min_replay_to_learn=4,
        target_sync_steps=20, train_every=1, use_amp=False,
        double_dqn=True,
    )


def test_conv_dqn_constructs(
    tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=tiny_conv_cfg, device="cpu", seed=0,
    )
    assert agent.global_step == 0


def test_conv_dqn_act_expects_2d_shape(
    tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=tiny_conv_cfg, device="cpu", seed=0,
    )
    tiny_env.reset(seed=0)
    obs2d = tiny_env.observation_2d()
    assert obs2d.shape == (4, 4, 4)
    a = agent.act(obs2d)
    assert 0 <= a < 4


def test_observation_2d_shape() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=6, cols=8, n_agents=3, initial_food_density=0.2,
        food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=10),
    )
    env = SoloSeasonalEnv(cfg)
    env.reset(seed=0)
    obs2d = env.observation_2d()
    assert obs2d.shape == (4, 6, 8)
    assert obs2d.dtype == np.float32
    # canal 0 (self) doit avoir exactement 1 case à 1.0
    assert obs2d[0].sum() == 1.0
    # canal 2 (food) doit être positif
    assert obs2d[2].sum() > 0
    # canal 3 (temp) doit être dans [0, 1]
    assert obs2d[3].min() >= 0.0
    assert obs2d[3].max() <= 1.0


def test_conv_dqn_observe_increments_step(
    tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=tiny_conv_cfg, device="cpu", seed=0,
    )
    tiny_env.reset(seed=0)
    obs = tiny_env.observation_2d()
    a = agent.act(obs)
    tiny_env.step(a)
    next_obs = tiny_env.observation_2d()
    agent.observe(obs, a, -1.0, next_obs, False)
    assert agent.global_step == 1


def test_conv_dqn_save_load(
    tmp_path: Path, tiny_env: SoloSeasonalEnv, tiny_conv_cfg: ConvDQNConfig
) -> None:
    agent = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=tiny_conv_cfg, device="cpu", seed=0,
    )
    ckpt = tmp_path / "conv.pt"
    agent.save(ckpt)
    assert ckpt.exists()
    agent2 = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=tiny_conv_cfg, device="cpu", seed=99,
    )
    agent2.load(ckpt)


def test_double_dqn_enabled_by_default(tiny_conv_cfg: ConvDQNConfig) -> None:
    """V2-W default = double_dqn=True."""
    assert tiny_conv_cfg.double_dqn is True
    cfg_default = ConvDQNConfig()
    assert cfg_default.double_dqn is True


def test_conv_dqn_no_double_variant() -> None:
    """V2-Z baseline accessible via double_dqn=False."""
    cfg = ConvDQNConfig(double_dqn=False)
    assert cfg.double_dqn is False
    agent = ConvDQNAgent(
        in_channels=4, rows=4, cols=4, n_actions=4,
        cfg=cfg, device="cpu", seed=0,
    )
    assert agent._impl.trainer.double_dqn is False  # noqa: SLF001
