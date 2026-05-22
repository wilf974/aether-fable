"""Tests pour RandomAgent et GreedyAgent."""
from __future__ import annotations

import numpy as np

from aetherlife.agents.greedy_agent import GreedyAgent
from aetherlife.agents.random_agent import RandomAgent
from aetherlife.config import FoodGridConfig
from aetherlife.env.single_agent_env import SoloForagerEnv
from aetherlife.world.food_grid import Action, FoodGrid


def test_random_agent_returns_valid_action() -> None:
    agent = RandomAgent(n_actions=4, seed=0)
    obs = np.zeros(513, dtype=np.float32)
    for _ in range(100):
        a = agent.act(obs)
        assert 0 <= a < 4


def test_random_agent_deterministic_with_seed() -> None:
    a1 = RandomAgent(n_actions=4, seed=42)
    a2 = RandomAgent(n_actions=4, seed=42)
    obs = np.zeros(513, dtype=np.float32)
    actions1 = [a1.act(obs) for _ in range(50)]
    actions2 = [a2.act(obs) for _ in range(50)]
    assert actions1 == actions2


def test_greedy_agent_moves_north_toward_food() -> None:
    cfg = FoodGridConfig(rows=5, cols=5, initial_food_density=0.0, start_position=(2, 2))
    env = FoodGrid(cfg)
    obs, _ = env.reset(seed=0)
    env._food_mask[0, 2] = True  # noqa: SLF001
    obs = env._observation()  # noqa: SLF001
    agent = GreedyAgent(rows=5, cols=5)
    action = agent.act(obs)
    assert action == Action.NORTH


def test_greedy_agent_moves_east_toward_food() -> None:
    cfg = FoodGridConfig(rows=5, cols=5, initial_food_density=0.0, start_position=(2, 2))
    env = FoodGrid(cfg)
    env.reset(seed=0)
    env._food_mask[2, 4] = True  # noqa: SLF001
    obs = env._observation()  # noqa: SLF001
    agent = GreedyAgent(rows=5, cols=5)
    action = agent.act(obs)
    assert action == Action.EAST


def test_greedy_agent_handles_no_food() -> None:
    cfg = FoodGridConfig(rows=5, cols=5, initial_food_density=0.0)
    env = FoodGrid(cfg)
    obs, _ = env.reset(seed=0)
    agent = GreedyAgent(rows=5, cols=5, seed=0)
    action = agent.act(obs)
    assert 0 <= action < 4


def test_greedy_agent_picks_nearest_food() -> None:
    cfg = FoodGridConfig(rows=10, cols=10, initial_food_density=0.0, start_position=(5, 5))
    env = FoodGrid(cfg)
    env.reset(seed=0)
    env._food_mask[0, 0] = True  # noqa: SLF001  loin
    env._food_mask[4, 5] = True  # noqa: SLF001  proche (1 step NORTH)
    obs = env._observation()  # noqa: SLF001
    agent = GreedyAgent(rows=10, cols=10)
    action = agent.act(obs)
    assert action == Action.NORTH


def test_greedy_agent_beats_random_on_dense_food() -> None:
    """Sur une grille avec food dense, Greedy doit clairement battre Random."""
    cfg = FoodGridConfig(
        rows=10, cols=10, max_energy=50.0, start_energy=25.0, metabolism=1.0,
        food_value=10.0, initial_food_density=0.15, food_respawn_lambda=0.3,
        max_steps=200,
    )
    env = SoloForagerEnv(cfg)
    random_agent = RandomAgent(n_actions=4, seed=0)
    greedy_agent = GreedyAgent(rows=10, cols=10, seed=0)

    def run(agent: object, n_episodes: int) -> float:
        survivals = 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)  # type: ignore[attr-defined]
                obs, _, terminated, truncated, _ = env.step(action)
            if truncated:
                survivals += 1
        return survivals / n_episodes

    n = 20
    rand_rate = run(random_agent, n)
    greedy_rate = run(greedy_agent, n)
    assert greedy_rate > rand_rate, (
        f"GreedyAgent ({greedy_rate:.2f}) doit battre RandomAgent ({rand_rate:.2f})"
    )
