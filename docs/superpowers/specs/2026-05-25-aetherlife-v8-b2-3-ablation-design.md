# V8-B2.3 — Test d'ablation interventionnelle de la communication

> **Statut** : spec post-V8-B2.2 30k (signal causal modéré + croissant).
> **Date** : 2026-05-25
> **Verrou scientifique** : démontrer (ou réfuter) que la communication
> a un rôle FONCTIONNEL via une intervention directe sur le canal.

---

## 0. La question critique

V8-B2.2 a démontré sur 30k ticks :
- KL listener_shift = 0.086 mean, 0.121 max (token 0 atteint seuil "fort")
- Context consistency = 47 % sur 72 clusters = ×34 baseline aléatoire
- Le shift **augmente** avec l'apprentissage (0.057 @ 5k → 0.086 @ 30k)

**Ces signaux sont compatibles avec une communication causale modérée.**

Mais il reste une lecture concurrente : **la communication pourrait être**
**corrélée à autre chose** (densité de population, état d'énergie, etc.) sans
être elle-même utile. Les agents pourraient survivre tout aussi bien sans elle.

**Le seul test qui tranche est interventionnel** : couper le canal à mi-run et
voir si quelque chose s'effondre.

---

## 1. Protocole d'ablation

### Conditions expérimentales

| Condition | Description |
|---|---|
| **Témoin** | V8-B2.1 standard, 30k ticks, vocalize actif tout le run |
| **Ablation @ t=15k** | Identique au témoin, mais à partir de t=15000, toute action `vocalize_X` est convertie en `idle/noop` |

Les deux runs partagent :
- Le **même seed** (42)
- La **même config** (biomes, affinity, respawn, brain)
- Le **même tracker** de causalité

Seule différence : à t=15k, dans le run ablation, l'action ≥ 4 ne déclenche plus
l'émission de token (et n'enregistre pas dans `_tokens_this_tick`).

### Implémentation : champ config

```python
@dataclass(frozen=True)
class VocabularyConfig:
    ...
    # V8-B2.3 — Ablation interventionnelle :
    # si pas None, à partir de ce tick, action vocalize devient no-op
    # (l'agent peut toujours choisir l'action 4-7 mais rien ne se passe)
    disable_vocalize_after_tick: int | None = None
```

Dans `seasonal_grid.step()` :
```python
if (vcfg.disable_vocalize_after_tick is not None
    and self._step_count > vcfg.disable_vocalize_after_tick):
    # Convertir action vocalize en idle (l'agent reste sur place)
    if action_id >= 4 and vcfg.enabled:
        # noop : ne push pas dans _tokens_this_tick
        pass  # juste métabolism + biome cost s'appliquent
```

### Métriques d'intérêt

Avant et après l'ablation t=15k, mesurer :

| Métrique | Hypothèse "communication utile" | Hypothèse "décoratif" |
|---|---|---|
| **alive_curve** | Pop baisse après t=15k | Aucun changement |
| **lifespan moy.** | Q3-Q4 < Q1-Q2 | Q3-Q4 ≥ Q1-Q2 |
| **n_lineages_final** | Plus de lignées éteintes | Identique au témoin |
| **food_eaten/agent/tick** | Diminue (coordination perdue) | Identique |
| **n_births** | Diminue | Identique |
| **listener_shift KL** | Tombe à 0 après t=15k (logique) | n/a |

### Comparaison témoin vs ablation

| Variable | Avant t=15k | Après t=15k | Δ témoin | Δ ablation |
|---|---|---|---|---|
| Pop | identique | identique-OU-divergent | mesuré | mesuré |
| Lifespan | identique | identique-OU-divergent | mesuré | mesuré |
| Food/agent | identique | identique-OU-divergent | mesuré | mesuré |

### Verdict probabiliste

- **Δ_ablation > Δ_témoin de >20 %** sur ≥ 2 métriques → **communication utile**
- **Δ_ablation < Δ_témoin de <5 %** sur toutes métriques → **communication décorative**
- Sinon → ambigu

---

## 2. Architecture

### 2.1 Champ config (1 ligne)

`VocabularyConfig.disable_vocalize_after_tick: int | None = None`

### 2.2 Step modifié (3 lignes)

Détecte le tick + skip enregistrement émission. Agent perd quand même
l'opportunité de bouger (no-op), mais ne paie PAS le coût vocalize.

### 2.3 Bench overnight : flag CLI

```bash
python scripts/overnight_v8b1.py \
    --ticks 30000 --regime language \
    --vocalize-disable-after 15000 \
    --out-dir results/v8b2_3_ablation_15k
```

Run témoin :
```bash
python scripts/overnight_v8b1.py \
    --ticks 30000 --regime language \
    --out-dir results/v8b2_3_control
```

### 2.4 Analyse comparative (script séparé)

```bash
python scripts/compare_ablation.py \
    --control results/v8b2_3_control/overnight_v8b1_seed42.json \
    --ablation results/v8b2_3_ablation_15k/overnight_v8b1_seed42.json \
    --ablation-tick 15000
```

Produit un rapport markdown comparatif :
- alive curve avant/après t=15k
- lifespan delta
- n_births delta
- verdict probabiliste

---

## 3. Critères de succès V8-B2.3

V8-B2.3 livré (`v0.8.14-alpha`) quand :

- [ ] Champ config + step modifié + tests (PAS de régression V8-B2.2)
- [ ] Run témoin + ablation lancés avec même seed
- [ ] Script `compare_ablation.py` produit un rapport comparatif
- [ ] Rapport Historian étendu avec section "ablation interventionnelle"
- [ ] Verdict probabiliste documenté (utile / décoratif / ambigu)

---

## 4. Non-objectifs

- Pas d'ablation progressive (juste binaire on/off)
- Pas de réintroduction après ablation (one-shot)
- Pas d'ablation sélective par token
- Pas de modification du LineageBrain (l'agent peut toujours CHOISIR
  l'action vocalize, mais rien ne se passe)

---

## 5. Pièges anticipés

1. **L'agent continue à perdre de l'énergie sur vocalize** si on garde
   le coût. → Décision : appliquer aussi `vocalize_energy_cost = 0` après
   l'ablation pour que le test soit propre (sinon ablation = perte
   énergie sans bénéfice = biais).
2. **Le brain DQN va apprendre à éviter vocalize** une fois que le
   reward n'est plus là. → C'est attendu et fait partie de l'observation.
3. **Le tracker continue à push** les actions mais pas les émissions.
   → Métriques causality auront un trou après t=15k.

---

## 6. Lecture probabiliste (à mettre dans le rapport)

**Le test d'ablation ne prouve PAS la sémantique des tokens.** Il prouve
uniquement si **la disponibilité du canal** a un impact mesurable sur la
survie/coordination. Une absence d'impact ne réfute pas le langage
(les agents peuvent simplement avoir une stratégie alternative qui
compense). Une présence d'impact ne prouve pas non plus la
compréhension (corrélation indirecte possible).

Ce qu'on cherche : **un signe statistique que le canal a une fonction**.
