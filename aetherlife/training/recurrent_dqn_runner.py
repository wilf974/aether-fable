"""DRQN training runner pour V3.5 — boucle avec hidden state + assessment + best-checkpoint."""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # import torch/mw_ia uniquement pour le typage
    from aetherlife.agents.recurrent_dqn_agent import RecurrentDQNAgent
from aetherlife.training.best_checkpoint import BestCheckpointTracker
from aetherlife.telemetry import MetricsLogger
from aetherlife.world.seasonal_solo_env import SoloSeasonalEnv


@dataclass
class DRQNEpisodeMetric:
    episode: int
    total_reward: float
    lifespan: int
    food_eaten: int
    survived: bool
    epsilon: float
    last_loss: float | None


@dataclass
class DRQNAssessmentMetric:
    train_episode: int
    survival_rate: float
    mean_lifespan: float
    mean_reward: float
    mean_food: float


@dataclass
class DRQNRunnerResult:
    train_metrics: list[DRQNEpisodeMetric] = field(default_factory=list)
    assessment_metrics: list[DRQNAssessmentMetric] = field(default_factory=list)
    best_assessment_score: float = float("-inf")
    best_assessment_episode: int = 0
    stopped_early: bool = False
    final_episode: int = 0


def _run_drqn_assess_episode(
    env: SoloSeasonalEnv, agent: RecurrentDQNAgent, seed: int
) -> tuple[bool, int, float, int]:
    agent.reset_hidden()
    agent.begin_episode()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    food_eaten = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("ate"):
            food_eaten += 1
    return (truncated, env.step_count, total_reward, food_eaten)


def drqn_assess(
    env: SoloSeasonalEnv,
    agent: RecurrentDQNAgent,
    n_episodes: int = 10,
    *,
    base_seed: int = 100_000,
) -> DRQNAssessmentMetric:
    results = [
        _run_drqn_assess_episode(env, agent, seed=base_seed + i)
        for i in range(n_episodes)
    ]
    survivals = [r[0] for r in results]
    lifespans = [r[1] for r in results]
    rewards = [r[2] for r in results]
    foods = [r[3] for r in results]
    return DRQNAssessmentMetric(
        train_episode=agent.global_step,
        survival_rate=sum(survivals) / n_episodes,
        mean_lifespan=statistics.mean(lifespans),
        mean_reward=statistics.mean(rewards),
        mean_food=statistics.mean(foods),
    )


def run_drqn_training(
    env: SoloSeasonalEnv,
    agent: RecurrentDQNAgent,
    *,
    n_episodes: int,
    assess_every: int = 25,
    assess_episodes: int = 5,
    checkpoint_path: str | Path = "checkpoints/drqn_best.pt",
    patience: int = 15,
    min_delta: float = 0.001,
    base_seed: int = 0,
    on_episode_end: Callable[[DRQNEpisodeMetric], None] | None = None,
    on_assess: Callable[[DRQNAssessmentMetric, bool], None] | None = None,
    metrics_dir: str | Path | None = None,
) -> DRQNRunnerResult:
    """Boucle d'entraînement DRQN sur env saisonnier single-agent.

    Pattern V2-Y MW_IA : reset_hidden + begin_episode au début, act/observe
    pendant l'épisode, end_episode après. Assessment greedy périodique
    avec best-checkpoint sur survival_rate.
    """
    tracker = BestCheckpointTracker(
        save_path=Path(checkpoint_path), patience=patience, min_delta=min_delta
    )
    result = DRQNRunnerResult()

    # V2.5 — telemetrie optionnelle (metrics.jsonl), aucun effet si None
    mlog: MetricsLogger | None = None
    if metrics_dir is not None:
        mlog = MetricsLogger(metrics_dir)


    for episode_idx in range(n_episodes):
        agent.reset_hidden()
        agent.begin_episode()
        obs, _ = env.reset(seed=base_seed + episode_idx)
        total_reward = 0.0
        food_eaten = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if info.get("ate"):
                food_eaten += 1

        agent.end_episode()

        metric = DRQNEpisodeMetric(
            episode=episode_idx,
            total_reward=total_reward,
            lifespan=env.step_count,
            food_eaten=food_eaten,
            survived=truncated,
            epsilon=agent.epsilon,
            last_loss=agent.last_loss,
        )
        result.train_metrics.append(metric)
        if mlog is not None:
            mlog.log(episode_idx, phase="train", **asdict(metric))
        if on_episode_end is not None:
            on_episode_end(metric)

        if (episode_idx + 1) % assess_every == 0:
            a_m = drqn_assess(env, agent, n_episodes=assess_episodes)
            improved = tracker.report(episode_idx, a_m.survival_rate, agent)
            result.assessment_metrics.append(a_m)
            if mlog is not None:
                mlog.log(episode_idx, phase="assess", improved=improved,
                         **asdict(a_m))
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
    if mlog is not None:
        mlog.summary(
            best_assessment_score=result.best_assessment_score,
            best_assessment_episode=result.best_assessment_episode,
            final_episode=result.final_episode,
            stopped_early=result.stopped_early,
        )
        mlog.close()
    return result
