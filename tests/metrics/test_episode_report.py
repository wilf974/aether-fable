"""Tests pour EpisodeStatsTracker + EpisodeReport."""
from __future__ import annotations

import pytest

from aetherlife.config import FoodGridConfig
from aetherlife.metrics.episode_report import (
    AgentEpisodeStats,
    EpisodeReport,
    EpisodeStatsTracker,
    format_report_lines,
)
from aetherlife.world.food_grid import Action, FoodGrid
from aetherlife.world.multi_agent_grid import (
    MultiAgentFoodGrid,
    MultiAgentForagerConfig,
)
from aetherlife.world.seasonal_grid import (
    SeasonalConfig,
    SeasonalMultiAgentConfig,
    SeasonalMultiAgentFoodGrid,
)


def test_tracker_reset_initializes_stats() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=3, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=3)
    tracker.reset(env)
    report = tracker.finalize(env)
    assert report.n_agents == 3
    assert report.n_alive_final == 3
    assert all(s.lifespan == 0 for s in report.by_agent)


def test_tracker_records_food_eaten() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=2, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    # force food sous l'agent 0
    env._agents[0].pos = (1, 1)  # noqa: SLF001
    env._agents[1].pos = (3, 3)  # noqa: SLF001
    env._food_mask[0, 1] = True  # noqa: SLF001  → agent 0 NORTH-> (0,1) ?
    # Plus simple : agent 0 sur (1,1), food sur (0,1), agent 0 NORTH
    tracker = EpisodeStatsTracker(n_agents=2)
    tracker.reset(env)
    _, _, _, _, infos = env.step({0: 0, 1: 0})
    tracker.on_step(env, infos)
    report = tracker.finalize(env)
    assert report.by_agent[0].food_eaten == 1
    assert report.by_agent[1].food_eaten == 0
    assert report.total_food_eaten == 1


def test_tracker_records_death() -> None:
    cfg = MultiAgentForagerConfig(
        rows=4, cols=4, n_agents=2, max_energy=5.0, start_energy=1.0,
        metabolism=1.0, food_value=2.0, death_penalty=2.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=2)
    tracker.reset(env)
    _, _, _, _, infos = env.step({0: 0, 1: 1})
    tracker.on_step(env, infos)
    report = tracker.finalize(env)
    assert report.n_dead_final == 2
    assert report.n_alive_final == 0
    assert report.by_agent[0].death_step == 1
    assert report.by_agent[1].death_step == 1


def test_tracker_records_distance() -> None:
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=1, initial_food_density=0.0,
        food_respawn_lambda=0.0, max_steps=20,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    env._agents[0].pos = (4, 4)  # noqa: SLF001
    tracker = EpisodeStatsTracker(n_agents=1)
    tracker.reset(env)
    for action in [Action.NORTH, Action.NORTH, Action.EAST, Action.EAST, Action.EAST]:
        _, _, _, _, infos = env.step({0: int(action)})
        tracker.on_step(env, infos)
    report = tracker.finalize(env)
    assert report.by_agent[0].distance_traveled == 5


def test_tracker_seasonal_tracking() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=1, max_energy=20.0, start_energy=10.0,
        metabolism=1.0, food_value=5.0, death_penalty=2.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=20,
        seasonal=SeasonalConfig(season_period=4),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=1, track_seasons=True)
    tracker.reset(env)
    # 4 ticks → 1 par saison
    for _ in range(4):
        _, _, _, _, infos = env.step({0: 0})
        tracker.on_step(env, infos)
    report = tracker.finalize(env)
    # Le tracker a vu 4 saisons distinctes (1 step chacune)
    assert sum(report.by_agent[0].season_steps.values()) == 4
    assert len(report.by_agent[0].season_steps) >= 2


def test_tracker_seasonal_mortality() -> None:
    cfg = SeasonalMultiAgentConfig(
        rows=4, cols=4, n_agents=2, max_energy=5.0, start_energy=1.0,
        metabolism=1.0, food_value=2.0, death_penalty=2.0,
        initial_food_density=0.0, food_respawn_lambda=0.0, max_steps=10,
        seasonal=SeasonalConfig(season_period=4),
    )
    env = SeasonalMultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=2, track_seasons=True)
    tracker.reset(env)
    _, _, _, _, infos = env.step({0: 0, 1: 1})
    tracker.on_step(env, infos)
    report = tracker.finalize(env)
    assert report.mortality_by_season is not None
    assert sum(report.mortality_by_season.values()) == 2


def test_format_report_lines_multi_agent() -> None:
    cfg = MultiAgentForagerConfig(
        rows=8, cols=8, n_agents=4, initial_food_density=0.1,
        food_respawn_lambda=0.5, max_steps=20,
    )
    env = MultiAgentFoodGrid(cfg)
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=4)
    tracker.reset(env)
    for _ in range(5):
        _, _, _, _, infos = env.step({aid: 0 for aid in env.alive_agent_ids})
        tracker.on_step(env, infos)
    report = tracker.finalize(env)
    lines = format_report_lines(report)
    assert any("Episode Report" in line for line in lines)
    assert any("alive" in line for line in lines)
    assert any("top performers" in line for line in lines)


def test_format_report_lines_single_agent() -> None:
    env = FoodGrid(FoodGridConfig(rows=5, cols=5, initial_food_density=0.0,
                                  food_respawn_lambda=0.0, max_steps=10))
    env.reset(seed=0)
    tracker = EpisodeStatsTracker(n_agents=1)
    tracker.reset(env)
    for _ in range(3):
        _, _, _, _, info = env.step(0)
        tracker.on_step(env, {0: info})
    report = tracker.finalize(env)
    lines = format_report_lines(report)
    assert any("Episode Report" in line for line in lines)
    assert any("agent:" in line for line in lines)


def test_dominant_season_property() -> None:
    s = AgentEpisodeStats(agent_id=0, season_steps={0: 3, 1: 7, 2: 2})
    assert s.dominant_season == 1
    s2 = AgentEpisodeStats(agent_id=1)
    assert s2.dominant_season is None


def test_survived_property() -> None:
    alive = AgentEpisodeStats(agent_id=0, lifespan=100)
    dead = AgentEpisodeStats(agent_id=1, lifespan=50, death_step=50)
    assert alive.survived is True
    assert dead.survived is False
