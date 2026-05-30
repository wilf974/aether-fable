# Spec — V8 Replay Viewer (Viewer 1 : replay / export)

**Date** : 2026-05-30
**Sous-projet** : OBS — Brain Observatory, **Viewer 1** (le premier des trois :
Viewer 1 replay/export → Viewer 2 live observer → Viewer 3 debug scientifique).
**Statut** : design validé, prêt pour plan TDD.
**Branche cible** : `master` (repo `aetherlife_pkg`).

---

## 1. Objectif

Rendre **visible** l'émergence linguistique/coordinative V8-C3 — aujourd'hui
étudiée uniquement en headless via `overnight_v8b1.py` (aucun viewer pygame
n'existe pour V8). Viewer 1 est une **machine à preuves visuelles** : à partir
d'un seed important (very_good, convention-candidat, étouffement…), produire un
clip compréhensible (MP4/GIF/PNG) **sans relancer le training**, réutilisable
pour Historian, public_article, preprint, site, démonstrations.

Ce qu'il rend visible : agents par lignée, **tokens vocalisés**, gather
collectif, clustering spatial, dynamique de population.

## 2. Contrainte structurante (découverte au brainstorming)

Les runs existants (phase D, P2) ne sauvent **que le rapport agrégé**
(`overnight_v8b1_seed*.json`) + les markdowns Historian. **Aucun état par-tick**
(positions, tokens, spots). Donc :

> Le replay/export **impose de re-simuler** le seed pour capturer la trajectoire.
> Les seeds sont reproductibles (RNG distribués) → un re-run redonne un run
> statistiquement équivalent. Caveat assumé : CUDA non bit-exact, donc le clip
> est un **instance représentative** du régime du seed, pas une réplique au pixel
> de l'original phase D. (Acceptable : on illustre un régime, pas une mesure.)

## 3. Architecture — séparation simulation / rendu

Principe directeur : **découpler la simulation (lourde, GPU, torch) du rendu
(léger, testable sans GPU)**. Le fichier `events.jsonl` est le **contrat**
entre les deux. Bénéfices : re-styliser un clip sans re-simuler ; tester le
renderer sans torch ni GPU ; events.jsonl = artefact réutilisable.

```
record_events_v8.py  ──►  events.jsonl + meta.json  ──►  pygame_viewer_v8.py  ──►  frames/clip
   (sim, GPU, torch)          (CONTRAT)                    (rendu, pur, testable)
```

### 3.1 Contrat — `events.jsonl` + `meta.json`

`meta.json` (1 fichier/run) :
```json
{"rows":24,"cols":24,"n_tokens":4,"listen_radius":10,"seed":25,
 "regime":"coordination_collective","vcost":0.05,
 "total_ticks":16000,"record_every":10,"schema_version":1}
```

`events.jsonl` (1 objet JSON / tick enregistré) :
```json
{"t":1240,"season":2,"n_alive":61,"n_births":186,"n_lin":7,
 "agents":[{"id":4,"lin":12,"r":8,"c":15,"e":63.2,"er":0.42,"age":812,"aff":2}],
 "vocal":{"4":2},
 "spots":[{"r":3,"c":9,"n":2}]}
```
- `agents[].lin` = `root_ancestor_id` (id de lignée).
- `agents[].er` = **energy ratio** = `energy / cfg.max_energy` (plus utile
  visuellement que l'énergie brute — calculé par le recorder).
- `agents[].age` = **dérivé** : `t − agent.birth_tick` (pas de champ `age` dans
  `_AgentState` ; le recorder le calcule, aucune modif du core).
- `agents[].aff` = `biome_affinity` — **un code entier** (`int | null`,
  `null` = pas de spéciation), PAS une string. Permet de colorier par niche
  plutôt que par lignée brute dans un futur overlay.
- `agents[].e` = énergie brute conservée (debug).
- `is_speaker_this_tick` : **non dupliqué par agent** — déductible du map
  `vocal` (membership = speaker, valeur = token). L'info est déjà dans le contrat.
- `vocal` = `{agent_id: token_id}` lu depuis `env._tokens_this_tick`.
- `spots[].n` = nb d'agents adjacents (≥2 ⇒ gather_collective potentiel).
- Champs courts pour limiter la taille (16k ticks × ~60 agents, `record_every=10`
  → ~1600 lignes × ~60 agents). Ces champs (er/age/aff) sont enregistrés
  **maintenant** même si pas affichés tout de suite → futurs overlays sans re-record.

### 3.2 Recorder — `scripts/record_events_v8.py`

- **Réutilise** `build_env` + `LineageAgent` importés depuis
  `scripts/overnight_v8b1.py`. **Ne modifie PAS** `overnight_v8b1.py`
  (préservation du tag `v0.8.17-alpha` et de la logique science).
- Boucle : `actions = policy.act(...)` → `env.step(actions)` → sérialise un
  event tous les `--record-every N` ticks → écrit la ligne jsonl.
- Flags : `--seed --ticks --regime --vocalize-cost --max-pop-override
  --bonus-energy-override --record-every --out-dir --device`.
  **`--record-every` défaut = 10** (16k ticks → ~1600 frames ≈ 53 s à 30 fps,
  idéal premier clip public ; `--record-every 1` pour debug image-par-image).
- Sortie : `<out-dir>/events.jsonl` + `<out-dir>/meta.json`.
- Seul module à dépendances lourdes (torch, env, cuda).
- ⚠️ `PYTHONIOENCODING=utf-8` requis (piège cp1252 hérité).

### 3.3 Renderer — `aetherlife/viz/pygame_viewer_v8.py`

- Fonction pure `render_events(events_path, meta_path, out_dir, *, fmt,
  fps, from_tick, to_tick, focus_lineage, cell_px) -> list[str]`.
- **Zéro dépendance torch/env** : lit uniquement le contrat. Rend chaque frame
  en surface offscreen (SDL `dummy` video driver), sauve PNG, assemble
  GIF/MP4 via imageio.
- Testable sur events.jsonl synthétique → sans GPU.

### 3.4 CLI — `scripts/render_v8.py`

```bash
python scripts/render_v8.py --events results/seed25/events.jsonl \
    --out clips/seed25.mp4 --fps 30 [--from 0 --to 16000] \
    [--fmt mp4|gif|png] [--focus-lineage 12]
```

## 4. Overlays (Viewer 1 minimum)

| Élément | Rendu |
|---|---|
| Grille + food | cellules, fond neutre |
| Agents | cellule pleine, **teinte = couleur stable hashée de `lin`** |
| Vocalize | halo/marqueur au-dessus du speaker, **couleur = token id** (4 couleurs) |
| Gather spot | marqueur ; **surbrillance si `n ≥ 2`** (coopération qui tire) |
| listen_radius | anneau (toggle, ou seulement sur `--focus-lineage`) |
| HUD | `tick / alive / births / lineages / saison` |

Palette tokens : 4 couleurs distinctes saturées. Palette lignées : hash
déterministe `lin → HSV` (couleur stable d'un tick à l'autre = lisibilité).

## 5. Dépendances — extra optionnel `viz`

`imageio`, `imageio-ffmpeg`, `PIL` absents du venv. pygame sauve les PNG
nativement ; GIF/MP4 via imageio. **Le cœur RL reste propre** :

```toml
[project.optional-dependencies]
viz = ["pygame-ce", "imageio", "imageio-ffmpeg"]
```
Install : `pip install -e ".[viz]"`. `imageio-ffmpeg` bundle le binaire ffmpeg
(~30 Mo) → MP4 turnkey.

## 6. Stratégie de test (TDD)

| Test | Niveau | GPU ? |
|---|---|---|
| `render_events` sur events.jsonl synthétique 3 ticks → N frames PNG écrites, dimensions = meta | unitaire renderer | non |
| couleur token déterministe : pixel du halo == couleur attendue pour token_id connu | unitaire renderer | non |
| hash lignée stable : même `lin` → même couleur sur 2 frames | unitaire renderer | non |
| assemblage : 3 PNG → 1 GIF non vide (imageio) | intégration | non |
| recorder smoke : 1 seed × 200 ticks → events.jsonl ≥1 ligne, schéma valide, `meta.total_ticks==200` | smoke recorder | oui (court) |

Le gros du test est **sans GPU** (renderer pur). Le recorder n'a qu'un smoke.

## 7. Hors scope (YAGNI / réservé OBS 2-3)

- Activations internes / Q-values / mémoire LSTM → **OBS Viewer 3** (debug).
- Fenêtre live interactive comme primaire → **OBS Viewer 2**. Le renderer reste
  pointable vers un flux live plus tard (compatible), mais Viewer 1 = fichier→frames.
- Comparaison 2 agents / 2 générations côte à côte → ultérieur.
- Pas d'invariants Aether (observabilité, pas de logique RL porteuse d'invariant).

## 8. Critère de réussite

> Prendre un seed important et produire un `.mp4` **compréhensible** (on voit
> les lignées, les tokens vocalisés, le gather collectif) **sans relancer le
> training** — en 2 commandes : `record_events_v8.py` puis `render_v8.py`.

**Seeds de démonstration (ordre validé)** :
1. **seed25** en premier — very_good, `cl_trend +27.5`, régime visuellement
   fort : montre clairement la **coordination**.
2. **seed45** en second (clip comparatif) — 240 succès, token concentré mais
   `cl_trend ≈ 0` : montre « **convention apparente sans coordination** ». La
   comparaison des deux clips illustre directement le finding « un seul
   attracteur sain = coordination ».

## 9. Livrables

- `scripts/record_events_v8.py` (recorder)
- `aetherlife/viz/pygame_viewer_v8.py` (renderer pur)
- `scripts/render_v8.py` (CLI)
- `tests/test_pygame_viewer_v8.py` (tests renderer sans GPU + assemblage)
- `pyproject.toml` : extra `viz`
- 1 clip de démonstration commité ou documenté (seed héros)
