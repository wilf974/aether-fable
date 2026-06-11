# Pistes d'intégration depuis l'écosystème `IA Inst` (notes — non implémenté)

> Survol des projets voisins (`C:\Users\Wilfred\Documents\IA Inst`) repérés comme
> potentiellement réutilisables pour AetherLife. **Aucun projet original n'est
> modifié.** Ce fichier est une liste d'idées à trancher plus tard.

## Intégré (inspiration)

- **AetherMind_OS** (`Projet Codex/AetherMind_OS`) — pattern « contrat auditable »
  (`MissionProfile` → `ExperimentPlan` gaté par invariants → JSON auditable +
  `launch_command` reproductible). A inspiré `aetherlife/analysis/prereg.py`
  (PreregSpec figé → plan de lancement reproductible → audit contre critères
  pré-spécifiés). Pattern repris, code non copié (AetherMind cible GridWorld/MW_IA).

## Couche diffusion / vulgarisation (pour plus tard)

- **NeuroGlyph** (`jeu ia2`) — roguelite 2D (arcade + numpy, ~60 fichiers). Pourrait
  servir de **démo jouable** illustrant les dynamiques émergentes d'AetherLife
  (écosystèmes, niches, coopération). Intégration possible : exporter un run
  AetherLife (events v8) → rejouer comme niveau/scénario NeuroGlyph. Effort : élevé.
- **AetherLife Origins** (`jeu ia`) — idle game mobile PWA déjà « inspiré d'AetherLife ».
  Vitrine grand public. Intégration légère : alimenter ses constantes/équilibrage
  à partir de findings réels (taux de survie, valeurs de portfolio effect).
- **Chatterbox** (`chatterbox`) — TTS open-source multilingue (Resemble AI). Pourrait
  **narrer automatiquement** les findings du Historian (résumé audio d'un run /
  d'un rapport scientifique). Intégration : Historian `summary.md` → script TTS →
  `.wav`. Effort : faible, dépendance lourde (torch + modèle).

## Écarté pour AetherLife (hors périmètre scientifique)

- **ImpactOS_Studio**, **NeuroMaze_Arena** (`Projet Codex`) — cockpits/jeux HTML
  autonomes, pas de code Python réutilisable pour la plateforme RL.
- **Labo_RL**, **NeuroScalp** — au stade note/concept (« à lire »), rien à intégrer.

## Rappel d'hygiène repo

Deux copies de MW_IA coexistent sur le disque : la canonique
`IA Inst/MW_IA` et la copie synchronisée `AetherLife/MW_IA` (liée à GitHub
`wilf974/mw-ia`). À terme, n'en garder qu'une pour éviter une nouvelle divergence.
