"""CacheConfig — stockage de food sur le nid (V5.3 caches).

Tension : manger maintenant vs stocker pour plus tard.

Mécanique :
- Quand un agent est sur son nid (ou nid familial si family_inheritance) :
  - Si energy >= deposit_threshold ET cache < capacity → déposer
    `deposit_amount` énergie dans le cache (cost prélevé sur l'agent)
  - Si energy < withdrawal_threshold ET cache > 0 → consommer
    `withdrawal_amount` du cache (gain pour l'agent)
- Le cache est attaché au nid (par owner_id) et persiste avec le nid.
- Quand le nid disparaît (mort owner sans descendant en V5.2), le cache
  est perdu.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    """Configuration des caches food V5.3.

    Args:
        enabled: si False (default), aucun cache. Compat V5.0/V5.2.
        deposit_threshold: énergie minimale pour déposer.
        withdrawal_threshold: énergie maximale pour consommer.
        max_capacity: capacité maximale du cache (par nid).
        deposit_amount: quantité d'énergie déplacée vers le cache par dépôt.
        withdrawal_amount: quantité regagnée par retrait.
    """

    enabled: bool = False
    deposit_threshold: float = 120.0
    withdrawal_threshold: float = 40.0
    max_capacity: float = 60.0
    deposit_amount: float = 5.0
    withdrawal_amount: float = 5.0

    def __post_init__(self) -> None:
        if self.deposit_threshold <= 0:
            raise ValueError(
                f"deposit_threshold doit être > 0 (got {self.deposit_threshold})"
            )
        if self.withdrawal_threshold < 0:
            raise ValueError(
                f"withdrawal_threshold doit être >= 0 (got {self.withdrawal_threshold})"
            )
        if self.withdrawal_threshold >= self.deposit_threshold:
            raise ValueError(
                f"withdrawal_threshold ({self.withdrawal_threshold}) doit être < "
                f"deposit_threshold ({self.deposit_threshold})"
            )
        if self.max_capacity <= 0:
            raise ValueError(f"max_capacity doit être > 0 (got {self.max_capacity})")
        if self.deposit_amount <= 0:
            raise ValueError(f"deposit_amount doit être > 0 (got {self.deposit_amount})")
        if self.withdrawal_amount <= 0:
            raise ValueError(
                f"withdrawal_amount doit être > 0 (got {self.withdrawal_amount})"
            )
