"""OBS V3.0 — Policy Fingerprint : sondes synthétiques + empreinte + distance.

Teste H2 (village vs mobile = politiques apprises différentes ?). Les sondes sont
construites via le VRAI `egocentric_obs` sur un env numpy contrôlé -> garantit que
le vecteur 505-dim correspond exactement a ce sur quoi le cerveau s'est entraine.
"""
from __future__ import annotations

import numpy as np

from aetherlife.agents.lineage_agent import egocentric_obs
from aetherlife.world.multi_agent_grid import _AgentState

# Espace d'actions coordination_collective : 4 moves + 4 vocalize + 1 gather.
ACTION_LABELS = [
    "MOVE_0", "MOVE_1", "MOVE_2", "MOVE_3",
    "VOC_0", "VOC_1", "VOC_2", "VOC_3", "GATHER",
]

PROBE_LABELS = [
    "Food_N", "Food_S", "Food_E", "Food_W",
    "Gather_adjacent", "Token_heard_0", "Token_heard_1",
    "Low_energy", "High_energy", "Alone", "Dense_neighbors",
]

_VISION = 4
_EMB = 16


def make_probe_env(seed: int = 1):
    """Construit un env coordination_collective (numpy, CPU) pour les sondes."""
    import sys, os
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "scripts"),
    )
    from overnight_v8b1 import build_env
    env = build_env(seed, regime="coordination_collective",
                    vocalize_energy_cost=0.05)
    env.reset(seed=seed)
    return env


def _clean_center_agent(env) -> _AgentState:
    """Reinitialise l'env a un etat controle : 1 agent au centre, pas de food,
    pas de voisins, pas de tokens, pas de spots. Retourne l'agent-sonde."""
    rows, cols = env.cfg.rows, env.cfg.cols
    cr, cc = rows // 2, cols // 2
    env._food_mask[:] = False  # noqa: SLF001
    env._tokens_this_tick = {}  # noqa: SLF001
    env._gather_spots = {}  # noqa: SLF001
    env._biome_map[:] = 0  # noqa: SLF001 — biome uniforme (PLAIN) : standardise
    # l'obs entre seeds -> policy_distance = pure difference de politique (pas geo)
    agent = _AgentState(
        agent_id=0, pos=(cr, cc), energy=env.cfg.max_energy * 0.5,
        alive=True, root_ancestor_id=0, birth_tick=0,
        biome_affinity=0,
    )
    env._agents = [agent]  # noqa: SLF001
    return agent


def _add_neighbor(env, dr: int, dc: int, agent_id: int = 1) -> _AgentState:
    ar, ac = env._agents[0].pos  # noqa: SLF001
    nb = _AgentState(
        agent_id=agent_id, pos=(ar + dr, ac + dc),
        energy=env.cfg.max_energy * 0.5, alive=True,
        root_ancestor_id=1, birth_tick=0, biome_affinity=0,
    )
    env._agents.append(nb)  # noqa: SLF001
    return nb


def build_probe_obs(env, label: str, *, listener_vocab=None) -> np.ndarray:
    """Construit l'observation d'une sonde via le vrai egocentric_obs.

    ATTENTION : Token_heard_0/1 produisent des obs IDENTIQUES si listener_vocab=None
    (heard-embeddings pades a zero). Pour les distinguer, passer listener_vocab=brain.vocabulary
    (fait automatiquement par fingerprint()).
    """
    agent = _clean_center_agent(env)
    ar, ac = agent.pos
    if label == "Food_N":
        env._food_mask[ar - 2, ac] = True  # noqa: SLF001
    elif label == "Food_S":
        env._food_mask[ar + 2, ac] = True  # noqa: SLF001
    elif label == "Food_E":
        env._food_mask[ar, ac + 2] = True  # noqa: SLF001
    elif label == "Food_W":
        env._food_mask[ar, ac - 2] = True  # noqa: SLF001
    elif label == "Gather_adjacent":
        from aetherlife.world.cooperative import GatherSpot
        spot_pos = (ar, ac + 1)
        env._gather_spots = {  # noqa: SLF001
            spot_pos: GatherSpot(
                pos=spot_pos, spawned_tick=0, expires_at=50,
            ),
        }
        _add_neighbor(env, 0, 1, agent_id=1)
    elif label == "Token_heard_0":
        nb = _add_neighbor(env, 0, 1, agent_id=1)
        env._tokens_this_tick = {nb.agent_id: 0}  # noqa: SLF001
    elif label == "Token_heard_1":
        nb = _add_neighbor(env, 0, 1, agent_id=1)
        env._tokens_this_tick = {nb.agent_id: 1}  # noqa: SLF001
    elif label == "Low_energy":
        agent.energy = env.cfg.max_energy * 0.1
    elif label == "High_energy":
        agent.energy = env.cfg.max_energy * 0.9
    elif label == "Alone":
        pass  # deja seul
    elif label == "Dense_neighbors":
        for i, (dr, dc) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            _add_neighbor(env, dr, dc, agent_id=i + 1)
    else:
        raise ValueError(f"sonde inconnue : {label}")
    obs = egocentric_obs(
        env, agent, _VISION, listener_vocab=listener_vocab, embedding_dim=_EMB,
    )
    return obs.astype(np.float32)


def fingerprint(brain, env) -> np.ndarray:
    """Matrice (n_sondes × 9) des Q-values du brain sur la batterie de sondes."""
    torch = brain._torch  # noqa: SLF001
    rows = []
    for label in PROBE_LABELS:
        obs = build_probe_obs(env, label, listener_vocab=brain.vocabulary)
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(brain.device)
            q = brain.online(x).cpu().numpy().reshape(-1)
        rows.append(q)
    return np.array(rows, dtype=np.float32)


def policy_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Distance cosine entre deux empreintes aplaties. 0=identique, 1=orthogonal."""
    a = np.asarray(fp_a, dtype=np.float64).reshape(-1)
    b = np.asarray(fp_b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos
