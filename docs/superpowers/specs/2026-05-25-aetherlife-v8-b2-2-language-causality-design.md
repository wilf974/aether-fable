# V8-B2.2 — Causalité comportementale du langage

> **Statut** : spec post-validation multi-seed V8-B2.1.
> **Date** : 2026-05-25
> **Prédécesseur** : V8-B2.1 (`v0.8.11-alpha`).
> **Verrou scientifique critique** : démontrer (ou réfuter) que
> l'écoute d'un token modifie réellement le comportement de l'auditeur.

---

## 0. Le verrou à débloquer

V8-B2.1 + multi-seed (5 seeds × 10k) a démontré que les **vocabulaires
divergent par lignée** de façon reproductible :
    - concentration par lignée 96.85 % ± 3.67 %
    - distance L2 inter-vocabs 2.68 ± 1.37

Mais : **divergence de tokens ≠ communication causale**. Les agents
pourraient simplement émettre des sons aléatoires différenciés par
lignée sans que personne ne les "comprenne".

V8-B2.2 répond à la question scientifique critique :

> Quand un agent entend le token X, son comportement change-t-il
> statistiquement par rapport à un agent qui ne l'entend pas ?

Si OUI → on a **communication causale**, pas juste divergence.
Si NON → l'hypothèse "dialecte" tombe à un niveau purement décoratif.

Dans les deux cas, on apprend quelque chose de **réel**.

---

## 1. Métriques cibles (4 niveaux d'analyse)

### M1 — `listener_behavior_shift_after_token`

Pour chaque token X émis :
1. Identifier les **listeners** (agents dans `listen_radius` du speaker au tick t)
2. Récolter leur distribution d'actions sur les ticks [t+1, t+T] où T=3
3. Comparer à la distribution d'actions **baseline** (tous agents, tous ticks)
4. Calculer KL(p_after_X || p_baseline)

Interprétation :
- KL ≈ 0 → l'écoute de X ne change rien (token décoratif)
- KL > seuil → l'écoute de X **biaise statistiquement** les actions
  futures de l'auditeur

### M2 — `same_token_same_context_rate`

Un même token est-il émis dans des contextes similaires ?

Pour chaque token X :
1. Capturer le **contexte d'émission** du speaker au tick d'émission :
   - food_visible_in_R (bool)
   - energy_norm (low/mid/high)
   - near_own_nest (bool)
   - n_neighbors (0, 1-2, 3+)
   - biome local (PLAIN/FOREST/DESERT/TUNDRA)
2. Cluster les contextes
3. Métrique = % d'émissions dans le cluster majoritaire

Interprétation :
- 25 % → uniforme aléatoire (4 clusters équilibrés)
- 60 % → légère consistance
- 80 %+ → **token contextuellement spécialisé** (X émis surtout
  quand `food_visible=True` par exemple)

### M3 — `delta_survival_after_communication` (futur, V8-B2.3)

Différence de lifespan entre agents qui ont entendu vs pas. Trop noisy
sur 30k ticks, repoussé.

### M4 — `token_conditioned_trajectories` (futur, V8-B2.3)

Visualisation des trajectoires post-écoute. Trop coûteux à logger
pour cette session.

---

## 2. Architecture

### 2.1 `LanguageCausalityTracker` (read-only)

```python
class LanguageCausalityTracker:
    """Trace les événements langage sans influencer les agents.

    Lifecyle : créé au début du run, push() à chaque tick,
    finalize() en fin de run → dict de métriques.
    """

    def __init__(self, n_tokens: int, baseline_window_ticks: int = 1000):
        self.n_tokens = n_tokens
        # Liste de (tick, speaker_id, token_id, listener_ids, context)
        self.emissions: list[dict] = []
        # Liste de (tick, agent_id, action) — baseline actions
        self.action_history: list[tuple[int, int, int]] = []
        # Liste de (tick, listener_id, action_id) après chaque écoute
        self.post_listen_actions: dict[int, list[int]] = {
            i: [] for i in range(n_tokens)
        }

    def push_emission(
        self, tick: int, speaker_id: int, token_id: int,
        listener_ids: list[int], context: dict,
    ) -> None: ...

    def push_action(self, tick: int, agent_id: int, action: int) -> None: ...

    def finalize(self) -> dict:
        """Renvoie les métriques M1 + M2."""
        return {
            "listener_shift": self._compute_listener_shift(),
            "context_consistency": self._compute_context_consistency(),
        }
```

### 2.2 Wiring dans `LineageAgent.observe_dict`

À chaque appel :
1. Récolter les `_tokens_this_tick` de l'env
2. Pour chaque émission : capturer les listeners + contexte
3. Push dans le tracker
4. Push actions de tous les agents (pour baseline)

**Le tracker n'influence rien**. Il observe uniquement les données
déjà disponibles. Aucune modification de reward, action, état.

### 2.3 Export dans le report

Ajouter à `language_metrics_v8b2` :
```json
{
  ...
  "causality_v8b2_2": {
    "listener_shift_per_token": {"0": 0.34, "1": 0.12, ...},
    "listener_shift_mean": 0.21,
    "listener_shift_max": 0.34,
    "context_consistency_per_token": {"0": 0.71, "1": 0.55, ...},
    "context_consistency_mean": 0.62,
    "n_emissions_total": 600000,
    "n_listeners_total": 1200000
  }
}
```

---

## 3. Critères d'interprétation

| Métrique | Seuil "communication" | Seuil "décoratif" |
|---|---|---|
| `listener_shift_mean` (KL) | > 0.10 | < 0.02 |
| `context_consistency_mean` | > 0.50 | < 0.30 |

**Décision** :
- Les deux métriques au-dessus du seuil "communication" → **hypothèse
  de communication causale renforcée**
- Une au-dessus, l'autre en-dessous → ambigu, plus de seeds
- Les deux en-dessous → l'hypothèse "dialecte communicatif" tombe,
  pattern réduit à "divergence ornementale"

**Toujours en langage probabiliste** : aucun token n'a "un sens".
Le verdict est sur l'existence d'un **signal statistique**, pas
sur la sémantique.

---

## 4. Non-objectifs V8-B2.2

- Pas de traduction sémantique
- Pas de LLM observateur
- Pas de visualisation (CSV uniquement)
- Pas de modification du mécanisme langage
- Pas de prouvée causalité au sens philosophique (juste statistique)

---

## 5. Plan TDD bite-sized

1. `LanguageCausalityTracker` class + tests stub (5 tests)
2. Wiring dans `LineageAgent.observe_dict` + test no-influence
3. Compute M1 (listener shift KL)
4. Compute M2 (context consistency)
5. Export dans `final_report["language_metrics_v8b2"]["causality_v8b2_2"]`
6. Update Historian pour rendu markdown causalité
7. Run 30k mode language avec tracker actif
8. Verdict scientifique
