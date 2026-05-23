"""EpisodeStatsTracker — collecte les stats per-agent au fil d'un épisode et génère un rapport.

Utilisation :
    tracker = EpisodeStatsTracker(n_agents=env.cfg.n_agents)
    tracker.reset(env)
    for tick in range(...):
        # ... env.step(actions) ...
        tracker.on_step(env, infos)
    report = tracker.finalize(env)
    for line in format_report_lines(report): print(line)

Compatible single-agent (V1, n_agents=1) et multi-agent (V2/V3).
Pour V3, le tracker collecte aussi les morts par saison si la clé "season"
est présente dans `info`.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEpisodeStats:
    """Stats per-agent collectées pendant un épisode."""

    agent_id: int
    lifespan: int = 0
    food_eaten: int = 0
    distance_traveled: int = 0
    max_energy_reached: float = 0.0
    final_energy: float = 0.0
    death_step: int | None = None
    death_season: int | None = None
    last_pos: tuple[int, int] | None = None
    season_steps: dict[int, int] = field(default_factory=dict)  # V3 only

    @property
    def survived(self) -> bool:
        return self.death_step is None

    @property
    def dominant_season(self) -> int | None:
        if not self.season_steps:
            return None
        return max(self.season_steps.items(), key=lambda kv: kv[1])[0]


@dataclass
class EpisodeReport:
    """Rapport agrégé d'un épisode."""

    n_agents: int
    final_step: int
    n_alive_final: int
    n_dead_final: int
    alive_rate: float
    mean_lifespan: float
    std_lifespan: float
    median_lifespan: float
    total_food_eaten: int
    mean_food_per_agent: float
    by_agent: list[AgentEpisodeStats]
    # V3 — saisonnalité (None si l'env ne fournit pas "season" dans info)
    mortality_by_season: dict[int, int] | None = None
    food_by_season: dict[int, int] | None = None


class EpisodeStatsTracker:
    """Tracker d'un épisode : appelé après chaque step.

    Args:
        n_agents: nombre total d'agents au reset.
        track_seasons: si True, collecte mortalité et food par saison
            (nécessite que `info` contient "season").
    """

    def __init__(self, n_agents: int, *, track_seasons: bool = False) -> None:
        self.n_agents = n_agents
        self.track_seasons = track_seasons
        self._stats: list[AgentEpisodeStats] = [
            AgentEpisodeStats(agent_id=i) for i in range(n_agents)
        ]
        self._mortality_by_season: dict[int, int] = {}
        self._food_by_season: dict[int, int] = {}
        self._final_step: int = 0

    def reset(self, env: Any) -> None:
        """Réinitialise le tracker à partir d'un env (positions de départ)."""
        self._stats = [AgentEpisodeStats(agent_id=i) for i in range(self.n_agents)]
        self._mortality_by_season = {}
        self._food_by_season = {}
        self._final_step = 0
        for i in range(self.n_agents):
            try:
                pos = env.agent_state(i).pos
            except (AttributeError, IndexError):
                pos = env.pos if hasattr(env, "pos") else None
            self._stats[i].last_pos = pos
            self._stats[i].max_energy_reached = (
                env.agent_state(i).energy if hasattr(env, "agent_state")
                else env.energy if hasattr(env, "energy") else 0.0
            )

    def on_step(self, env: Any, infos: dict | None = None) -> None:
        """Met à jour les stats après un step. infos = dict per-agent ou dict step single-agent."""
        self._final_step = env.step_count
        # Normaliser infos en dict[int, dict] (single agent → {0: info})
        if infos is None:
            infos = {}
        elif not isinstance(infos, dict):
            infos = {0: infos}
        elif infos and not isinstance(next(iter(infos.keys())), int):
            # single agent info dict ; wrap
            infos = {0: infos}

        for aid, s in enumerate(self._stats):
            if s.death_step is not None:
                continue  # déjà mort, plus de mise à jour
            # position courante
            try:
                cur_pos = env.agent_state(aid).pos
                cur_energy = env.agent_state(aid).energy
                alive = env.agent_state(aid).alive
            except (AttributeError, IndexError):
                cur_pos = getattr(env, "pos", None)
                cur_energy = getattr(env, "energy", 0.0)
                alive = cur_energy > 0
            # distance
            if s.last_pos is not None and cur_pos is not None:
                dr = abs(cur_pos[0] - s.last_pos[0])
                dc = abs(cur_pos[1] - s.last_pos[1])
                s.distance_traveled += dr + dc
            s.last_pos = cur_pos
            # énergie
            if cur_energy > s.max_energy_reached:
                s.max_energy_reached = float(cur_energy)
            s.final_energy = float(cur_energy)
            # info per-agent
            info = infos.get(aid, infos.get(0, {}) if aid == 0 else {})
            season = info.get("season") if self.track_seasons else None
            if info.get("ate"):
                s.food_eaten += 1
                if season is not None:
                    self._food_by_season[int(season)] = (
                        self._food_by_season.get(int(season), 0) + 1
                    )
            if self.track_seasons and season is not None:
                s.season_steps[int(season)] = s.season_steps.get(int(season), 0) + 1
            # lifespan & death
            if alive:
                s.lifespan = env.step_count
            elif s.death_step is None:
                s.death_step = env.step_count
                s.lifespan = env.step_count
                if self.track_seasons and season is not None:
                    s.death_season = int(season)
                    self._mortality_by_season[int(season)] = (
                        self._mortality_by_season.get(int(season), 0) + 1
                    )

    def finalize(self, env: Any) -> EpisodeReport:
        """Génère le rapport final."""
        lifespans = [s.lifespan for s in self._stats]
        n_alive = sum(1 for s in self._stats if s.death_step is None)
        n_dead = self.n_agents - n_alive
        total_food = sum(s.food_eaten for s in self._stats)
        return EpisodeReport(
            n_agents=self.n_agents,
            final_step=self._final_step,
            n_alive_final=n_alive,
            n_dead_final=n_dead,
            alive_rate=n_alive / self.n_agents if self.n_agents else 0.0,
            mean_lifespan=statistics.mean(lifespans) if lifespans else 0.0,
            std_lifespan=statistics.pstdev(lifespans) if len(lifespans) > 1 else 0.0,
            median_lifespan=statistics.median(lifespans) if lifespans else 0.0,
            total_food_eaten=total_food,
            mean_food_per_agent=total_food / self.n_agents if self.n_agents else 0.0,
            by_agent=list(self._stats),
            mortality_by_season=dict(self._mortality_by_season) or None,
            food_by_season=dict(self._food_by_season) or None,
        )


_SEASON_NAMES = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}


def format_report_lines(report: EpisodeReport, *, max_agent_lines: int = 6) -> list[str]:
    """Formate un rapport en lignes ASCII pour affichage console ou GUI."""
    lines: list[str] = []
    lines.append(
        f"=== Episode Report (final step {report.final_step}) ==="
    )
    lines.append(
        f"  alive {report.n_alive_final}/{report.n_agents} "
        f"({report.alive_rate:.1%})  dead {report.n_dead_final}"
    )
    lines.append(
        f"  lifespan: mean={report.mean_lifespan:.1f}  "
        f"median={report.median_lifespan:.0f}  std={report.std_lifespan:.1f}"
    )
    lines.append(
        f"  food: total={report.total_food_eaten}  "
        f"mean/agent={report.mean_food_per_agent:.1f}"
    )
    if report.mortality_by_season is not None:
        # V3 seasonal breakdown
        season_str = "  mortality by season: " + "  ".join(
            f"{_SEASON_NAMES.get(s, str(s))}={n}"
            for s, n in sorted(report.mortality_by_season.items())
        )
        lines.append(season_str)
        if report.food_by_season:
            food_str = "  food eaten by season: " + "  ".join(
                f"{_SEASON_NAMES.get(s, str(s))}={n}"
                for s, n in sorted(report.food_by_season.items())
            )
            lines.append(food_str)

    # Top by lifespan
    sorted_by_life = sorted(
        report.by_agent, key=lambda s: (-s.lifespan, -s.food_eaten)
    )
    if report.n_agents > 1:
        lines.append("  -- top performers (lifespan, food) --")
        for s in sorted_by_life[:max_agent_lines // 2]:
            mark = "[alive]" if s.survived else f"[died@{s.death_step}]"
            extra = ""
            if s.dominant_season is not None:
                extra = f" dom={_SEASON_NAMES.get(s.dominant_season, '?')}"
            lines.append(
                f"    #{s.agent_id:2d} life={s.lifespan:4d} food={s.food_eaten:3d} "
                f"dist={s.distance_traveled:4d} {mark}{extra}"
            )
        if report.n_dead_final > 0:
            lines.append("  -- earliest deaths --")
            died = sorted(
                [s for s in report.by_agent if s.death_step is not None],
                key=lambda s: s.death_step or 0,
            )
            for s in died[:max_agent_lines // 2]:
                extra = ""
                if s.death_season is not None:
                    extra = f" died-in-{_SEASON_NAMES.get(s.death_season, '?')}"
                lines.append(
                    f"    #{s.agent_id:2d} life={s.lifespan:4d} food={s.food_eaten:3d}"
                    f"{extra}"
                )
    else:
        s = sorted_by_life[0]
        mark = "[survived]" if s.survived else f"[died@{s.death_step}]"
        lines.append(
            f"  agent: life={s.lifespan}  food={s.food_eaten}  "
            f"distance={s.distance_traveled}  max_energy={s.max_energy_reached:.1f}  {mark}"
        )
    return lines
