# Spec — OBS Viewer 2.0 (lite) : Live V8 Observer + Historien + jours

**Date** : 2026-06-03
**Sous-projet** : OBS — Brain Observatory, **Viewer 2.0 (lite)** (live observer).
**Statut** : design validé, prêt pour plan TDD.
**Branche cible** : `master` (via `feature/obs-v2-live-observer`).

---

## 1. Objectif

Une interface où l'on **regarde le monde V8 vivre**, on **règle la durée en jours**,
et on **lance l'analyse de l'Historien d'un bouton** pour voir les findings — sans
relancer un run headless. Le GUI actuel (`launch_gui_v3`) tourne en V3
(écologie/saisons) : il ne montre ni langage, ni coopération, ni lignées, et son
rapport d'épisode n'est pas l'Historien. V2.0 lite tourne la **config V8**
(`coordination_collective`) pour que le compte rendu Historien soit **riche**.

> Boucle d'usage : *je regarde vivre → je lance l'analyse (H) → je vois les findings.*

## 2. Réutilisation (≈70 % existant — ne pas réécrire)

| Brique | Source réutilisée |
|---|---|
| Rendu V8 (grille, agents/lignée, halos tokens, gather spots, HUD) | `aetherlife/viz/pygame_viewer_v8._draw_frame` (Viewer 1) |
| Sim V8 (biomes + vocab + coop + lignées) | `build_env` + `LineageAgent` (overnight_v8b1, **non modifié**) |
| mobility_score live | `historian/spatial_mobility` (`OccupancyAccumulator`, `window_bounds`, `build_spatial_mobility_block`) |
| Coop metrics | `env.coop_metrics.finalize()` |
| Analyse + découvertes | `Historian.from_report` + `DiscoveriesDetector` |

## 3. Composants (code nouveau)

| Fichier | Responsabilité | GPU |
|---|---|---|
| `aetherlife/viz/live_report.py` | `build_live_report(env, policy, occ_s, occ_e, windows, n_ticks)` → report dict | non |
| `aetherlife/viz/live_viewer_v8.py` | boucle live : step env → event dict/tick → `_draw_frame` + accumulation + clavier + overlay Historien | oui (env) |
| `scripts/launch_gui_v8.py` | entrée CLI : `--regime --days --ticks-per-day --device` | oui |

### 3.1 `build_live_report` (le seul morceau délicat)

Assemble un report dict consommable par `Historian`/`DiscoveriesDetector`, à partir
de l'état env + policy + accumulateurs, **sans toucher au runner taggé**. Blocs :
- `final_state` : `n_alive`, `n_births_total`, `top_lineages` (root_id/alive/pct),
  `affinity_distribution`.
- `criterion_3_selection` : `n_lineages_initial` (= n_agents initial), `n_lineages_final` (= len(registry)).
- `language_metrics_v8b2` : reproduit le calcul de `overnight_v8b1` (≈ lignes 616-662) —
  lit `policy.registry` brains `vocabulary.usage_count` : `total_vocalize_count`,
  `tokens_per_1000_ticks`, `mean_token_lineage_concentration`, `entropy_ratio`,
  `mean_inter_lineage_distance`. (Factorisé ici, pas dans overnight.)
- `cooperative_v8c3` : `enabled`, `gather_successes_total`, `gather_failures_total`.
- `cooperative_metrics_v8c3` : `env.coop_metrics.finalize()`.
- `spatial_mobility_v8c3` : `build_spatial_mobility_block(occ_s, occ_e, ...)`.

Détecteurs omis au MVP (Niveau 2 ultérieur) : `criterion_1_inheritance`
(lifespan quartiles — nécessite tracking des morts par tick). Le `DiscoveriesDetector`
retourne `[]` pour les blocs absents → pas de bruit.

### 3.2 `live_viewer_v8` (boucle)

```
build_env(regime) + LineageAgent  (comme overnight, vision_radius/BrainConfig idem)
boucle:
  events pygame (clavier)
  si pas pause et day_courant < budget:
    actions = policy.act_dict(obs_stub) ; env.step(actions)
    accumuler occupation (fenêtres window_bounds sur le budget) + compteurs
    construire event dict du tick (t, season, n_alive, n_lin, agents[id,lin,r,c,e],
      vocal, spots) — même schéma que le recorder
    _draw_frame(event, meta, cell_px) -> blit à l'écran + HUD live
  si overlay Historien actif: dessiner le panneau
  flip
```

HUD live : `day X/N · pop · births · lignées · vocal · gather · mobility`.

### 3.3 Contrôles clavier (cohérent `pygame_viewer_v3`)

| Touche | Action |
|---|---|
| `ESPACE` | pause / reprise |
| `+` / `-` | augmenter / diminuer le budget en **jours** (live) |
| `H` | overlay Historien (build_live_report → discoveries + résumé) |
| `E` | export Niveau 2 : `Historian.write_all()` → `results/gui_run/report/` (markdown + json) |
| `↑` / `↓` | vitesse (delay_ms) |
| `ESC` / `Q` | quitter |

« via bouton **ou autre** » → touches (le GUI pygame n'a pas d'infra de boutons souris ;
les touches sont le pattern établi du projet).

## 4. « Régler le run en jours »

**1 jour = `ticks_per_day` ticks (défaut 1000)**, configurable via `--ticks-per-day`.
Budget total = `jours × ticks_per_day`. HUD : « day X/N ». `+`/`-` ajustent le nombre
de jours **en live** (le budget s'étend/réduit pendant le run). Quand le budget est
atteint, la sim se met en pause (l'overlay Historien reste accessible).

## 5. Niveau 1 vs Niveau 2

- **Niveau 1 (touche H, live)** : panneau overlay dans le GUI — compteurs
  (pop/births/lignées/vocal/gather/mobility_score) + **liste des découvertes
  principales** (slug + confiance + headline, via `Historian.discoveries`). Lecture
  immédiate, pas d'écriture disque.
- **Niveau 2 (touche E)** : `Historian.write_all("results/gui_run/report")` → les 9
  fichiers (summary.md, discoveries.md, scientific_report.md, public_article.md,
  metrics.json, …). Pour relecture/partage hors GUI.

## 6. Décisions actées

1. Contrôles **clavier** (pas boutons souris).
2. **1 jour = 1000 ticks** (défaut configurable).
3. Détecteurs MVP : régime / langage / coopération / mobilité / sélection. Héritage
   cognitif omis (Niveau 2).
4. Régime défaut : `coordination_collective`.
5. `overnight_v8b1.py` **non modifié** (build_env réutilisé en lecture seule).

## 7. Tests (TDD, sans GPU)

| Test | Vérifie |
|---|---|
| `build_live_report` sur un petit env (CPU) → contient les 6 blocs, types corrects | assemblage |
| `Historian.from_report(build_live_report(...))` → liste de Discovery (≥0, pas de crash) | intégration Historian |
| `language_metrics_v8b2` reproduit : sur un registry avec vocab usage connu, concentration/total cohérents | fidélité au calcul overnight |
| smoke loop `live_viewer_v8` SDL dummy, ~20 ticks, budget court → pas de crash, event dict valide, HUD rendu | boucle |

`build_live_report` testable CPU (env numpy + brains CPU). La boucle : smoke headless
(SDL dummy) court.

## 8. Hors scope (V2.1+ / ultérieur)

- Boutons souris cliquables (infra UI pygame).
- Héritage cognitif / activations internes (= OBS V3, déjà livré pour l'introspection
  statique).
- Multi-fenêtres, comparaison de runs côte à côte.
- Sauvegarde/replay de la session live (= Viewer 1 replay).

## 9. Livrables

- `aetherlife/viz/live_report.py` (build_live_report)
- `aetherlife/viz/live_viewer_v8.py` (boucle live + overlay Historien + clavier/jours)
- `scripts/launch_gui_v8.py` (entrée)
- `tests/test_live_report.py` (assemblage + Historian, sans GPU)
- `tests/test_live_viewer_v8.py` (smoke loop headless)
