"""V8-B1.2 — LineageAgent : policy multi-agent qui route chaque action via
le LineageRegistry, et apprend par RL via les observations égocentriques.

Drop-in remplacement de `SmartHeuristicAgent` : expose la même API
`act_dict(obs_dict, greedy)` mais récupère/crée le brain de chaque agent
selon son `root_ancestor_id`.

Observation égocentrique (vision_radius=5 → 11×11) :
    - canal 1 : food
    - canal 2 : nests
    - canal 3 : plants (matures + immatures)
    - canal 4 : autres agents
    - + scalars : énergie, age_norm, season_phase
    Total dim = 4 × (2r+1)² + 3

Pourquoi égocentrique :
    - généralise aux chunks infinis plus tard (l'agent ne sait pas où il
      est, juste ce qu'il voit)
    - dim compacte (pas O(rows × cols))
    - invariance par translation
"""
from __future__ import annotations

import numpy as np

from aetherlife.agents.lineage_brain import BrainConfig
from aetherlife.agents.lineage_registry import LineageRegistry


def egocentric_obs(env, agent, vision_radius: int = 5) -> np.ndarray:
    """Construit l'observation égocentrique d'un agent.

    Args:
        env: SeasonalMultiAgentFoodGrid (a `food_mask`, `nests`, `plants`,
             `_agents`, `phase`, `cfg.max_energy`, `biome_map` V8-B1.5).
        agent: _AgentState (a `pos`, `energy`, `age_norm` calculé ici).
        vision_radius: rayon de vision (fenêtre carrée 2r+1).

    Returns:
        np.ndarray float32 shape (5 * (2r+1)² + 3,)

    Canaux :
        0: food
        1: nests
        2: plants
        3: agents (autres)
        4: biome (encoding [0, 0.33, 0.66, 1.0] selon BiomeType)
        + scalars: energy_norm, age_norm, season_phase
    """
    r = vision_radius
    size = 2 * r + 1
    ar, ac = agent.pos
    rows = env.cfg.rows
    cols = env.cfg.cols
    # 5 canaux V8-B1.5
    food_view = np.zeros((size, size), dtype=np.float32)
    nest_view = np.zeros((size, size), dtype=np.float32)
    plant_view = np.zeros((size, size), dtype=np.float32)
    agent_view = np.zeros((size, size), dtype=np.float32)
    biome_view = np.zeros((size, size), dtype=np.float32)
    food_mask = env.food_mask
    biome_map_local = getattr(env, "_biome_map", None)
    nest_set = {n.pos for n in env.nests.values()}
    plants = getattr(env, "plants", {})
    agents = env._agents  # noqa: SLF001
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            gr = ar + dr
            gc = ac + dc
            if not (0 <= gr < rows and 0 <= gc < cols):
                continue
            i, j = dr + r, dc + r
            if food_mask[gr, gc]:
                food_view[i, j] = 1.0
            if (gr, gc) in nest_set:
                nest_view[i, j] = 1.0
            if (gr, gc) in plants:
                plant_view[i, j] = 1.0
            if biome_map_local is not None:
                # 0→0.0, 1→0.33, 2→0.66, 3→1.0
                biome_view[i, j] = float(biome_map_local[gr, gc]) / 3.0
    for other in agents:
        if not other.alive or other.agent_id == agent.agent_id:
            continue
        odr = other.pos[0] - ar
        odc = other.pos[1] - ac
        if -r <= odr <= r and -r <= odc <= r:
            agent_view[odr + r, odc + r] = 1.0
    energy_norm = float(agent.energy / max(env.cfg.max_energy, 1e-6))
    age_norm = float(
        (env.step_count - agent.birth_tick) / max(env.cfg.max_steps, 1)
    )
    season_phase = float(getattr(env, "phase", 0.0))
    parts = [
        food_view.flatten(),
        nest_view.flatten(),
        plant_view.flatten(),
        agent_view.flatten(),
        biome_view.flatten(),
        np.array([energy_norm, age_norm, season_phase], dtype=np.float32),
    ]
    return np.concatenate(parts)


def egocentric_obs_dim(vision_radius: int) -> int:
    """Calcule la dim de l'observation égocentrique pour un radius.

    V8-B1.5 : 5 canaux × (2r+1)² + 3 scalars.
    """
    return 5 * (2 * vision_radius + 1) ** 2 + 3


class LineageAgent:
    """Policy RL multi-agent par lignée.

    API compatible avec SmartHeuristicAgent (act_dict).
    Additionnellement expose observe_dict() pour l'apprentissage.
    """

    def __init__(
        self,
        env,
        cfg: BrainConfig | None = None,
        *,
        n_actions: int = 4,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.cfg = cfg or BrainConfig(enabled=True)
        self.n_actions = n_actions
        self._seed = seed
        self._next_seed = seed + 10_000
        self.obs_dim = egocentric_obs_dim(self.cfg.vision_radius)
        self.registry = LineageRegistry(
            cfg=self.cfg, obs_dim=self.obs_dim, n_actions=n_actions,
        )
        # Init un brain par lignée fondatrice (agents id 0..n_agents-1)
        for agent in env._agents:  # noqa: SLF001
            if agent.alive:
                self.registry.get_or_create(
                    root_id=agent.root_ancestor_id,
                    parent_brain=None,
                    seed=self._fresh_seed(),
                )

    def _fresh_seed(self) -> int:
        s = self._next_seed
        self._next_seed += 1
        return s

    def _ensure_brain_for(self, agent) -> None:
        """Crée le brain de la lignée si nouveau-né d'une lignée inédite.

        Note V5.2 : tous les descendants partagent `root_ancestor_id` du
        fondateur. Donc en pratique on n'aura pas de "nouvelle lignée"
        après le reset, sauf si on introduit fork explicit en B2+.
        """
        root = agent.root_ancestor_id
        if root in self.registry:
            return
        # Si parent connu : hériter, sinon init random
        parent_brain = None
        if agent.parent_id is not None:
            parent_agent = self._get_agent_by_id(agent.parent_id)
            if parent_agent is not None:
                parent_brain = self.registry.get(parent_agent.root_ancestor_id)
        self.registry.get_or_create(
            root_id=root, parent_brain=parent_brain, seed=self._fresh_seed(),
        )

    def _get_agent_by_id(self, aid: int):
        for a in self.env._agents:  # noqa: SLF001
            if a.agent_id == aid:
                return a
        return None

    def act_dict(
        self,
        obs_dict: dict[int, np.ndarray],
        *,
        greedy: bool = False,
    ) -> dict[int, int]:
        """Pour chaque agent dans obs_dict, retourne son action via son brain.

        Note : `obs_dict` n'est PAS utilisé directement — on construit
        l'observation égocentrique à partir de l'env. C'est pour rester
        compatible avec l'API SmartHeuristicAgent.
        """
        actions: dict[int, int] = {}
        env = self.env
        for aid in obs_dict:
            try:
                agent = env.agent_state(aid)
            except Exception:
                actions[aid] = 0
                continue
            if not agent.alive:
                actions[aid] = 0
                continue
            self._ensure_brain_for(agent)
            brain = self.registry.get(agent.root_ancestor_id)
            if brain is None:
                actions[aid] = 0
                continue
            obs = egocentric_obs(env, agent, self.cfg.vision_radius)
            actions[aid] = brain.act(obs, greedy=greedy)
        return actions

    def observe_dict(
        self,
        prev_obs_ego: dict[int, np.ndarray],
        actions: dict[int, int],
        rewards: dict[int, float],
        next_obs_ego: dict[int, np.ndarray],
        dones: dict[int, bool],
        agent_root_ids: dict[int, int],
    ) -> dict[str, float]:
        """Push transitions dans les brains correspondants + train.

        Args:
            prev_obs_ego: obs égocentriques avant action (computed par caller)
            actions: action choisie par agent
            rewards: reward reçu
            next_obs_ego: obs après action
            dones: terminal flag
            agent_root_ids: agent_id → root_ancestor_id (pour routing)

        Returns:
            metrics agrégées (moyenne loss, max epsilon).
        """
        losses: list[float] = []
        epsilons: list[float] = []
        for aid, obs in prev_obs_ego.items():
            if aid not in actions or aid not in rewards:
                continue
            root = agent_root_ids.get(aid)
            if root is None:
                continue
            brain = self.registry.get(root)
            if brain is None:
                continue
            n_obs = next_obs_ego.get(aid, obs)
            done = dones.get(aid, False)
            m = brain.observe(
                obs, actions[aid], rewards[aid], n_obs, done,
            )
            if "loss" in m:
                losses.append(m["loss"])
            epsilons.append(m["epsilon"])
        out: dict[str, float] = {}
        if losses:
            out["mean_loss"] = float(np.mean(losses))
        if epsilons:
            out["max_epsilon"] = float(max(epsilons))
            out["mean_epsilon"] = float(np.mean(epsilons))
        return out
