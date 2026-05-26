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


def egocentric_obs(
    env, agent, vision_radius: int = 5,
    *, listener_vocab=None,
) -> np.ndarray:
    """Construit l'observation égocentrique d'un agent.

    Args:
        env: SeasonalMultiAgentFoodGrid (a `food_mask`, `nests`, `plants`,
             `_agents`, `phase`, `cfg.max_energy`, `biome_map` V8-B1.5).
        agent: _AgentState (a `pos`, `energy`, `age_norm` calculé ici).
        vision_radius: rayon de vision (fenêtre carrée 2r+1).
        listener_vocab: V8-B2.0 — Vocabulary de l'AUDITEUR pour décoder
            les tokens entendus selon SON propre dict. Si None, pas de
            heard_embeddings (compat V8-B1.x).

    Returns:
        np.ndarray float32 shape (5 * (2r+1)² + 3 + embedding_dim,)
        embedding_dim = 0 si listener_vocab is None

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
    # 5 canaux V8-B1.5 + 1 canal gather V8-C3 (si actif)
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
    # V8-C2 — food masquée à la vue sauf sur la tile centrale (l'agent
    # ne sait pas où est la food à distance, seule le listen peut le
    # révéler via les vocalises des voisins). Mécanique manger préservée.
    hidden_food = getattr(
        getattr(env.cfg, "biomes", None), "hidden_food", False,
    )
    # V8-C3 — gather spots
    coop_enabled = getattr(
        getattr(env.cfg, "cooperative", None), "enabled", False,
    )
    gather_view = np.zeros((size, size), dtype=np.float32) if coop_enabled else None
    gather_spots = getattr(env, "_gather_spots", {}) if coop_enabled else {}
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            gr = ar + dr
            gc = ac + dc
            if not (0 <= gr < rows and 0 <= gc < cols):
                continue
            i, j = dr + r, dc + r
            # V8-C2 : food visible UNIQUEMENT sur la tile centrale si masqué
            if food_mask[gr, gc]:
                if hidden_food and (dr, dc) != (0, 0):
                    pass  # food invisible à distance
                else:
                    food_view[i, j] = 1.0
            if (gr, gc) in nest_set:
                nest_view[i, j] = 1.0
            if (gr, gc) in plants:
                plant_view[i, j] = 1.0
            if biome_map_local is not None:
                # 0→0.0, 1→0.33, 2→0.66, 3→1.0
                biome_view[i, j] = float(biome_map_local[gr, gc]) / 3.0
            # V8-C3 — gather spots visibles (locaux à la vision)
            if gather_view is not None and (gr, gc) in gather_spots:
                gather_view[i, j] = 1.0
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
    ]
    if gather_view is not None:
        parts.append(gather_view.flatten())
    parts.append(np.array(
        [energy_norm, age_norm, season_phase], dtype=np.float32,
    ))
    # V8-B2.0 — heard tokens : moyenne des embeddings des tokens
    # vocalize par les voisins audibles (Manhattan ≤ listen_radius)
    # Décodés via le vocabulary de l'AUDITEUR (sinon dialectes impossibles)
    if listener_vocab is not None:
        embedding_dim = listener_vocab.cfg.embedding_dim
        listen_r = listener_vocab.cfg.listen_radius
        heard_vec = np.zeros(embedding_dim, dtype=np.float32)
        tokens_dict = getattr(env, "_tokens_this_tick", None)
        if tokens_dict:
            count = 0
            for other in env._agents:  # noqa: SLF001
                if not other.alive or other.agent_id == agent.agent_id:
                    continue
                d = abs(other.pos[0] - ar) + abs(other.pos[1] - ac)
                if d > listen_r:
                    continue
                tok = tokens_dict.get(other.agent_id)
                if tok is None:
                    continue
                heard_vec += listener_vocab.get_embedding(tok)
                count += 1
            if count > 0:
                heard_vec /= count
        parts.append(heard_vec)
    return np.concatenate(parts)


def egocentric_obs_dim(
    vision_radius: int, vocab_dim: int = 0, coop: bool = False,
) -> int:
    """Calcule la dim de l'observation égocentrique pour un radius.

    V8-B1.5 : 5 canaux × (2r+1)² + 3 scalars.
    V8-B2.0 : + vocab_dim si vocabulary actif.
    V8-C3 : +1 canal gather si coop actif.
    """
    n_channels = 6 if coop else 5
    return n_channels * (2 * vision_radius + 1) ** 2 + 3 + vocab_dim


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
        # V8-B2.0 — extend action space si vocabulary actif
        vcfg = getattr(env.cfg, "vocabulary", None)
        self.vocab_cfg = vcfg if (vcfg is not None and vcfg.enabled) else None
        if self.vocab_cfg is not None:
            self.n_actions = n_actions + self.vocab_cfg.n_tokens
            vocab_dim = self.vocab_cfg.embedding_dim
        else:
            self.n_actions = n_actions
            vocab_dim = 0
        # V8-C3 — extension action_space pour gather_collective
        ccfg = getattr(env.cfg, "cooperative", None)
        self.coop_cfg = ccfg if (ccfg is not None and ccfg.enabled) else None
        if self.coop_cfg is not None:
            self.n_actions += 1  # +1 pour l'action gather
            coop_obs = True
        else:
            coop_obs = False
        # Recalculer obs_dim avec coop
        from aetherlife.agents.lineage_agent import egocentric_obs_dim as _dim
        # (auto reference, mais on est dans la classe)
        self._seed = seed
        self._next_seed = seed + 10_000
        self.obs_dim = egocentric_obs_dim(
            self.cfg.vision_radius, vocab_dim=vocab_dim,
            coop=self.coop_cfg is not None,
        )
        bcfg = getattr(env.cfg, "biomes", None)
        seed_bank_max = (
            bcfg.seed_bank_max_per_affinity if bcfg is not None else 2
        )
        self.registry = LineageRegistry(
            cfg=self.cfg, obs_dim=self.obs_dim, n_actions=self.n_actions,
            seed_bank_max_per_affinity=seed_bank_max,
        )
        self._vocab_rng = np.random.default_rng(seed + 5555)
        # Init un brain par lignée fondatrice
        for agent in env._agents:  # noqa: SLF001
            if agent.alive:
                brain = self.registry.get_or_create(
                    root_id=agent.root_ancestor_id,
                    parent_brain=None,
                    seed=self._fresh_seed(),
                )
                brain.biome_affinity = agent.biome_affinity
                # V8-B2.0 — init vocabulary random pour fondateur
                if self.vocab_cfg is not None and brain.vocabulary is None:
                    from aetherlife.world.vocabulary import Vocabulary
                    brain.vocabulary = Vocabulary.random(
                        self.vocab_cfg, self._vocab_rng,
                    )
        # V8-B1.7 — tracking dernière vie d'une affinity (pour respawn)
        self._affinity_last_seen: dict[int, int] = {}
        self._last_respawn_for_aff: dict[int, int] = {}
        # V8-B2.0 — tracking pour reward social : qui a vocalize quand
        # speakers_recent[agent_id] = tick de dernière vocalize
        self._speakers_recent: dict[int, int] = {}

    def _fresh_seed(self) -> int:
        s = self._next_seed
        self._next_seed += 1
        return s

    def _ensure_brain_for(self, agent) -> None:
        """Crée le brain de la lignée si nouveau-né d'une lignée inédite.

        V5.2 : tous les descendants partagent `root_ancestor_id` du fondateur.
        V8-B1.7 : si nouvelle lignée (respawn fondateur) ET seed bank a
        un brain pour cette affinity, hériter du brain archivé.
        """
        root = agent.root_ancestor_id
        if root in self.registry:
            return
        # 1) Si parent connu : hériter du parent
        parent_brain = None
        if agent.parent_id is not None:
            parent_agent = self._get_agent_by_id(agent.parent_id)
            if parent_agent is not None:
                parent_brain = self.registry.get(parent_agent.root_ancestor_id)
        # 2) V8-B1.7 : fondateur sans parent → check seed bank
        if parent_brain is None and agent.biome_affinity is not None:
            parent_brain = self.registry.get_seed_brain_for_affinity(
                agent.biome_affinity
            )
        brain = self.registry.get_or_create(
            root_id=root, parent_brain=parent_brain, seed=self._fresh_seed(),
        )
        # V8-B1.7 — attacher l'affinity au brain
        brain.biome_affinity = agent.biome_affinity
        # V8-B2.0 — init vocab si pas hérité d'un parent et vocabulary actif
        if self.vocab_cfg is not None and brain.vocabulary is None:
            from aetherlife.world.vocabulary import Vocabulary
            brain.vocabulary = Vocabulary.random(self.vocab_cfg, self._vocab_rng)

    def _get_agent_by_id(self, aid: int):
        for a in self.env._agents:  # noqa: SLF001
            if a.agent_id == aid:
                return a
        return None

    def make_obs(self, agent) -> np.ndarray:
        """V8-B2.0 — Helper pour construire l'obs d'un agent avec le bon vocab.

        Utilisé par le bench overnight pour éviter de dupliquer la logique
        de décodage des tokens entendus.
        """
        listener_vocab = None
        if self.vocab_cfg is not None:
            brain = self.registry.get(agent.root_ancestor_id)
            if brain is not None:
                listener_vocab = brain.vocabulary
        return egocentric_obs(
            self.env, agent, self.cfg.vision_radius,
            listener_vocab=listener_vocab,
        )

    def maybe_respawn_extinct_affinities(self) -> int:
        """V8-B1.7 — Check si une affinity est éteinte depuis trop longtemps,
        et la relance via env.spawn_founder() + seed bank.

        Retourne le nombre de respawns effectués ce tick.
        """
        env = self.env
        bcfg = getattr(env.cfg, "biomes", None)
        if bcfg is None or not bcfg.affinity_enabled or not bcfg.respawn_enabled:
            return 0
        cur_tick = env._step_count  # noqa: SLF001
        if cur_tick % bcfg.respawn_check_every != 0:
            return 0
        # Compter vivants par affinity
        alive_per_aff: dict[int, int] = {a: 0 for a in range(4)}
        for ag in env._agents:  # noqa: SLF001
            if ag.alive and ag.biome_affinity is not None:
                alive_per_aff[ag.biome_affinity] = alive_per_aff.get(
                    ag.biome_affinity, 0
                ) + 1
        # Mettre à jour last_seen et déclencher respawn si seuil
        n_respawned = 0
        for aff, n_alive in alive_per_aff.items():
            if n_alive >= bcfg.respawn_threshold:
                self._affinity_last_seen[aff] = cur_tick
                continue
            last_seen = self._affinity_last_seen.get(aff, 0)
            ticks_since = cur_tick - last_seen
            if ticks_since < bcfg.respawn_extinct_after_ticks:
                continue
            # Cooldown anti-respawn-spam
            last_resp = self._last_respawn_for_aff.get(aff, -10**9)
            if cur_tick - last_resp < bcfg.respawn_check_every * 4:
                continue
            # Respawn
            new_id = env.spawn_founder(aff)
            if new_id is None:
                continue
            self._last_respawn_for_aff[aff] = cur_tick
            # Brain via seed bank si dispo, sinon random
            seed_brain = self.registry.get_seed_brain_for_affinity(aff)
            if seed_brain is not None:
                brain = self.registry.get_or_create(
                    root_id=new_id, parent_brain=seed_brain,
                    seed=self._fresh_seed(),
                )
            else:
                brain = self.registry.get_or_create(
                    root_id=new_id, parent_brain=None,
                    seed=self._fresh_seed(),
                )
            brain.biome_affinity = aff
            n_respawned += 1
        return n_respawned

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

        V8-B2.0 : passe le vocab du brain à egocentric_obs pour décoder
        les tokens entendus selon SON propre dict. Tracker usage pour
        métriques d'émergence.
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
            # V8-B2.0 — décode heard tokens via vocab du brain de l'auditeur
            listener_vocab = brain.vocabulary if self.vocab_cfg else None
            obs = egocentric_obs(
                env, agent, self.cfg.vision_radius,
                listener_vocab=listener_vocab,
            )
            action_id = brain.act(obs, greedy=greedy)
            actions[aid] = action_id
            # V8-B2.0 — tracker l'usage du token si vocalize
            if self.vocab_cfg is not None and action_id >= 4:
                token_id = action_id - 4
                if brain.vocabulary is not None:
                    brain.vocabulary.record_use(token_id)
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
