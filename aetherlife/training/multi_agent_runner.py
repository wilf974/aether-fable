"""MA training runner pour V2 — boucle IDQN shared-weights avec assessment greedy."""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # import torch/mw_ia uniquement pour le typage
    from aetherlife.agents.independent_dqn import IndependentDQNAgent
from aetherlife.training.best_checkpoint import BestCheckpointTracker
from aetherlife.telemetry import MetricsLogger
from aetherlife.world.multi_agent_grid import MultiAgentFoodGrid


@dataclass
class MAEpisodeMetric:
    episode: int
    n_alive_final: int
    mean_lifespan: float
    total_food_eaten: int
    total_reward: float
    epsilon: float
    last_loss: float | None


@dataclass
class MAAssessmentMetric:
    train_episode: int
    mean_lifespan: float
    mean_alive_rate: float       # n_alive_final / n_agents, moyenne sur épisodes
    mean_total_food: float
    mean_reward_per_agent: float


@dataclass
class MARunnerResult:
    train_metrics: list[MAEpisodeMetric] = field(default_factory=list)
    assessment_metrics: list[MAAssessmentMetric] = field(default_factory=list)
    best_assessment_score: float = float("-inf")
    best_assessment_episode: int = 0
    stopped_early: bool = False
    final_episode: int = 0


def _run_ma_episode_greedy(
    env: MultiAgentFoodGrid, agent: IndependentDQNAgent, seed: int
) -> tuple[int, float, int, float]:
    """Une éval greedy MA : retourne (n_alive_final, mean_lifespan, total_food, total_reward)."""
    obs_dict, _ = env.reset(seed=seed)
    lifespans: dict[int, int] = {aid: 0 for aid in env.alive_agent_ids}
    total_food = 0
    total_reward = 0.0
    while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
        actions = agent.act_dict(obs_dict, greedy=True)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        for aid, info in infos.items():
            if info["alive"] or terminated.get(aid, False) or truncated.get(aid, False):
                lifespans[aid] = env.step_count
            if info.get("ate"):
                total_food += 1
            total_reward += rewards.get(aid, 0.0)
        obs_dict = {aid: obs for aid, obs in next_obs.items()
                    if env.agent_state(aid).alive}
        if all(terminated.get(aid, False) for aid in actions):
            break
    mean_life = statistics.mean(lifespans.values()) if lifespans else 0.0
    return env.n_alive, mean_life, total_food, total_reward


def ma_assess(
    env: MultiAgentFoodGrid,
    agent: IndependentDQNAgent,
    n_episodes: int = 5,
    *,
    base_seed: int = 100_000,
) -> MAAssessmentMetric:
    results = [
        _run_ma_episode_greedy(env, agent, seed=base_seed + i)
        for i in range(n_episodes)
    ]
    alives = [r[0] for r in results]
    lifespans = [r[1] for r in results]
    foods = [r[2] for r in results]
    rewards = [r[3] for r in results]
    return MAAssessmentMetric(
        train_episode=agent.global_step,
        mean_lifespan=statistics.mean(lifespans),
        mean_alive_rate=statistics.mean(alives) / env.cfg.n_agents,
        mean_total_food=statistics.mean(foods),
        mean_reward_per_agent=statistics.mean(rewards) / env.cfg.n_agents,
    )


def run_ma_training(
    env: MultiAgentFoodGrid,
    agent: IndependentDQNAgent,
    *,
    n_episodes: int,
    assess_every: int = 25,
    assess_episodes: int = 5,
    checkpoint_path: str | Path = "checkpoints/ma_best.pt",
    patience: int = 10,
    min_delta: float = 0.01,
    base_seed: int = 0,
    on_episode_end: Callable[[MAEpisodeMetric], None] | None = None,
    on_assess: Callable[[MAAssessmentMetric, bool], None] | None = None,
    metrics_dir: str | Path | None = None,
) -> MARunnerResult:
    tracker = BestCheckpointTracker(
        save_path=Path(checkpoint_path), patience=patience, min_delta=min_delta
    )
    result = MARunnerResult()

    # V2.5 — telemetrie optionnelle (metrics.jsonl), aucun effet si None
    mlog: MetricsLogger | None = None
    if metrics_dir is not None:
        mlog = MetricsLogger(metrics_dir)


    for episode_idx in range(n_episodes):
        obs_dict, _ = env.reset(seed=base_seed + episode_idx)
        lifespans: dict[int, int] = {aid: 0 for aid in env.alive_agent_ids}
        total_food = 0
        total_reward = 0.0

        while env.n_alive > 0 and env.step_count < env.cfg.max_steps:
            actions = agent.act_dict(obs_dict)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            done_dict = {aid: terminated.get(aid, False) or truncated.get(aid, False)
                         for aid in actions}
            agent.observe_dict(obs_dict, actions, rewards, next_obs, done_dict)
            for aid, info in infos.items():
                if env.agent_state(aid).alive or terminated.get(aid, False) or truncated.get(aid, False):
                    lifespans[aid] = env.step_count
                if info.get("ate"):
                    total_food += 1
                total_reward += rewards.get(aid, 0.0)
            obs_dict = {aid: obs for aid, obs in next_obs.items()
                        if env.agent_state(aid).alive}
            if all(terminated.get(aid, False) for aid in actions):
                break

        metric = MAEpisodeMetric(
            episode=episode_idx,
            n_alive_final=env.n_alive,
            mean_lifespan=statistics.mean(lifespans.values()) if lifespans else 0.0,
            total_food_eaten=total_food,
            total_reward=total_reward,
            epsilon=agent.epsilon,
            last_loss=agent.last_loss,
        )
        result.train_metrics.append(metric)
        if mlog is not None:
            mlog.log(episode_idx, phase="train", **asdict(metric))
        if on_episode_end is not None:
            on_episode_end(metric)

        if (episode_idx + 1) % assess_every == 0:
            a_m = ma_assess(env, agent, n_episodes=assess_episodes)
            improved = tracker.report(episode_idx, a_m.mean_alive_rate, agent)
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
