"""V8-C3 — Actions coopératives lourdes : `gather_collective`.

Une mécanique qu'un agent seul ne peut PAS résoudre :
    1. Des 'gather_spots' apparaissent aléatoirement sur la carte
    2. Pour les exploiter (+30 énergie), il faut ≥2 agents adjacents
    3. Les spots disparaissent après N ticks
    4. Action 8 = 'gather_collective' (au-delà des 4+4 = move+vocalize)

Hypothèse falsifiable :
    - Si le langage devient sélectionné pour coordination spatiale,
      l'ablation du canal vocalize devrait faire chuter le nombre de
      gather successes (et donc la population/fécondité)
    - Si pas d'effet → langage reste sub-fonctionnel, revoir architecture
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CooperativeConfig:
    """Configuration des actions coopératives V8-C3."""

    enabled: bool = False
    # Nombre de partenaires adjacents requis (= total agents - 1)
    # min_partners_adjacent=1 → 2 agents au total nécessaires
    min_partners_adjacent: int = 1

    # Fenêtre temporelle (informative, pour design futur)
    signal_window_ticks: int = 5

    # Récompense énergétique du gather collectif
    bonus_energy: float = 30.0

    # Densité de spawn des gather_spots (par tick, sur tiles libres)
    spawn_lambda: float = 0.5  # ~0.5 spot par tick en moyenne

    # Durée de vie d'un spot avant disparition (ticks)
    decay_ticks: int = 50

    # Cap max de spots simultanés (évite saturation)
    max_active_spots: int = 30

    # Action ID pour gather_collective (au-delà de 4 move + N vocab)
    # = (4 + n_tokens) → calculé dynamiquement, ce champ est juste un offset
    action_offset_from_vocab: int = 0  # +0 = juste après le dernier vocalize

    def __post_init__(self) -> None:
        if self.min_partners_adjacent < 1:
            raise ValueError(
                f"min_partners_adjacent doit être >= 1 "
                f"(got {self.min_partners_adjacent})"
            )
        if self.signal_window_ticks < 1:
            raise ValueError(
                f"signal_window_ticks doit être >= 1 "
                f"(got {self.signal_window_ticks})"
            )
        if self.bonus_energy <= 0:
            raise ValueError(
                f"bonus_energy doit être > 0 (got {self.bonus_energy})"
            )
        if self.spawn_lambda < 0:
            raise ValueError(
                f"spawn_lambda doit être >= 0 (got {self.spawn_lambda})"
            )
        if self.decay_ticks < 1:
            raise ValueError(
                f"decay_ticks doit être >= 1 (got {self.decay_ticks})"
            )
        if self.max_active_spots < 1:
            raise ValueError(
                f"max_active_spots doit être >= 1 (got {self.max_active_spots})"
            )


@dataclass
class GatherSpot:
    """Une tile spéciale qui rapporte +bonus_energy à un groupe d'agents."""
    pos: tuple[int, int]
    spawned_tick: int
    expires_at: int

    def is_expired(self, current_tick: int) -> bool:
        return current_tick >= self.expires_at
