"""AetherLife Historian — module d'observation et reporting des runs.

Le Historian observe uniquement les données déjà produites par le moteur
de simulation (JSON résultats, événements). Il ne touche jamais aux
agents, à l'env, ou au RL.

Outputs typiques pour un `run_id` donné :

    reports/<run_id>/summary.md            — résumé court (1 page)
    reports/<run_id>/scientific_report.md  — rapport rigoureux
    reports/<run_id>/public_article.md     — article blog/newsletter
    reports/<run_id>/discoveries.md        — hypothèses détectées
    reports/<run_id>/lineages.md           — vie des familles
    reports/<run_id>/dialects.md           — analyse linguistique
    reports/<run_id>/metrics.json          — données brutes structurées
    reports/<run_id>/events.jsonl          — timeline événements
    reports/<run_id>/charts.csv            — data pour graphiques

Principes :
    - **Aucune influence sur le système agent** (read-only sur logs/JSON)
    - **Langage probabiliste** : "le pattern suggère...", pas "les agents
      ont découvert X"
    - **Fail-safe** : si une métrique manque, on génère "non observable",
      jamais d'exception qui casse le run
"""
from __future__ import annotations

from aetherlife.historian.discoveries import (
    Discovery, DiscoveriesDetector,
)
from aetherlife.historian.historian import Historian

__all__ = ["Historian", "DiscoveriesDetector", "Discovery"]
