"""MA ConvDQN training runner V3.7 — multi-agent saisonnier 2D + assessment held-out."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from aetherlife.agents.independent_conv_dqn import IndependentConvDQNAgent
from aetherlife.training.best_checkpoint import BestCheckpointTracker
from aetherlife.world.seasonal_grid import SeasonalMultiAgentFoodGrid


@dataclass
class MAConvEpisodeMetric:
    episode: int
    n_alive_final: int
    mean_lifespan: float
    total_food_eaten: int
    total_reward: float
    epsilon: float
    last_loss: float | None


@dataclass
class MAConvAssessmentMetric:
    train_episode: int
    mean_alive_rate: float
    mean_lifespan: float
    mean_total_food: float
    mean_reward_per_agent: float


@dataclass
class MAConvRunnerResult:
    train_metrics: list[MAConvEpisodeMetric] = field(default_factory=list)
    assessment_metrics: list[MAConvAssessmentMetric] = field(default_factory=list)
    best_assessment_score: float = float("-inf")
    best_assessment_episode: int = 0
    stopped_early: bool = False
    final_episode: int = 0


def _run_ma_conv_assess_episode(
    env: SeasonalMultiAgentFoodGrid, agent: IndependentConvDQNAgent, seed: int
) -> tuple[int, float, int, float]:
    env.reset(seed=seed)
    lifespans: dict[int, int] = {aid: 0 for aid in env.alive_agent_ids}
    total_food = 0
    total_reward = 0.0
    while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
        obs_2d = env.observation_2d_dict()
        actions = agent.act_dict(obs_2d, greedy=True)
        if not actions:
            break
        _, rewards, terminated, _, infos = env.step(actions)
        for aid, info in infos.items():
            if env.agent_state(aid).alive or terminated.get(aid, False):
                lifespans[aid] = env.step_count
            if info.get("ate"):
                total_food += 1
            total_reward += rewards.get(aid, 0.0)
        if all(terminated.get(aid, False) for aid in actions):
            break
    mean_life = statistics.mean(lifespans.values()) if lifespans else 0.0
    return env.n_alive, mean_life, total_food, total_reward


def ma_conv_assess(
    env: SeasonalMultiAgentFoodGrid,
    agent: IndependentConvDQNAgent,
    n_episodes: int = 5,
    *,
    base_seed: int = 100_000,
) -> MAConvAssessmentMetric:
    """Eval held-out sur seeds [base_seed, base_seed+n_episodes)."""
    results = [
        _run_ma_conv_assess_episode(env, agent, seed=base_seed + i)
        for i in range(n_episodes)
    ]
    alives = [r[0] for r in results]
    lifespans = [r[1] for r in results]
    foods = [r[2] for r in results]
    rewards = [r[3] for r in results]
    return MAConvAssessmentMetric(
        train_episode=agent.global_step,
        mean_alive_rate=statistics.mean(alives) / env.cfg.n_agents,
        mean_lifespan=statistics.mean(lifespans),
        mean_total_food=statistics.mean(foods),
        mean_reward_per_agent=statistics.mean(rewards) / env.cfg.n_agents,
    )


def run_ma_conv_training(
    env: SeasonalMultiAgentFoodGrid,
    agent: IndependentConvDQNAgent,
    *,
    n_episodes: int,
    assess_every: int = 25,
    assess_episodes: int = 5,
    checkpoint_path: str | Path = "checkpoints/ma_conv_best.pt",
    patience: int = 20,
    min_delta: float = 0.01,
    base_seed: int = 0,
    on_episode_end: Callable[[MAConvEpisodeMetric], None] | None = None,
    on_assess: Callable[[MAConvAssessmentMetric, bool], None] | None = None,
) -> MAConvRunnerResult:
    """Boucle MA ConvDQN+DDQN sur env saisonnier complexe."""
    tracker = BestCheckpointTracker(
        save_path=Path(checkpoint_path), patience=patience, min_delta=min_delta
    )
    result = MAConvRunnerResult()

    for episode_idx in range(n_episodes):
        env.reset(seed=base_seed + episode_idx)
        lifespans: dict[int, int] = {aid: 0 for aid in env.alive_agent_ids}
        total_food = 0
        total_reward = 0.0

        while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
            obs_2d = env.observation_2d_dict()
            if not obs_2d:
                break
            actions = agent.act_dict(obs_2d)
            _, rewards, terminated, truncated, infos = env.step(actions)
            next_obs_2d = env.observation_2d_dict()
            done_dict = {
                aid: terminated.get(aid, False) or truncated.get(aid, False)
                for aid in actions
            }
            # Pour les agents qui meurent au step courant, next_obs n'inclut pas
            # leur id ; on remplit avec leur prev obs comme placeholder.
            full_next = {aid: next_obs_2d.get(aid, obs_2d[aid]) for aid in actions}
            agent.observe_dict(obs_2d, actions, rewards, full_next, done_dict)
            for aid, info in infos.items():
                if env.agent_state(aid).alive or terminated.get(aid, False) or truncated.get(aid, False):
                    lifespans[aid] = env.step_count
                if info.get("ate"):
                    total_food += 1
                total_reward += rewards.get(aid, 0.0)
            if all(terminated.get(aid, False) for aid in actions):
                break

        metric = MAConvEpisodeMetric(
            episode=episode_idx,
            n_alive_final=env.n_alive,
            mean_lifespan=statistics.mean(lifespans.values()) if lifespans else 0.0,
            total_food_eaten=total_food,
            total_reward=total_reward,
            epsilon=agent.epsilon,
            last_loss=agent.last_loss,
        )
        result.train_metrics.append(metric)
        if on_episode_end is not None:
            on_episode_end(metric)

        if (episode_idx + 1) % assess_every == 0:
            a_m = ma_conv_assess(env, agent, n_episodes=assess_episodes)
            improved = tracker.report(episode_idx, a_m.mean_alive_rate, agent)
            result.assessment_metrics.append(a_m)
            if on_assess is not None:
                on_assess(a_m, improved)
            if tracker.should_stop:
                result.stopped_early = True
                result.final_episode = episode_idx
                break

    result.best_assessment_score = tracker.best_score
    result.best_assessment_episode = tracker.best_step
    if not result.stopped_early:
        result.final_episode = n_episodes - 1
    return result
