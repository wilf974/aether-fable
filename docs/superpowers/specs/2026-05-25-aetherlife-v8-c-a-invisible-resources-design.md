# V8-C.A — Ressources invisibles : le silence devient coûteux

> **Statut** : spec post-finding V8-B2.3 (proto-culture sans fonction).
> **Date** : 2026-05-25
> **Verrou scientifique** : créer un environnement où la communication est
> nécessaire à la survie, et tester par ablation si elle devient
> fonctionnellement sélectionnée.

---

## 0. Le pivot conceptuel

V8-B2.3 a démontré que dans le régime "language" actuel, les agents peuvent
survivre **sans communication** (ablation @ t=15k → 4/4 métriques inchangées).

Lecture user (validée) : **les agents ne dépendent pas du langage parce que
vision locale + abondance + heuristiques suffisent**. Pour que le langage
devienne fonctionnel, il faut **rendre le silence coûteux**.

V8-C.A introduit la première mécanique du pivot V8-C :
**ressources invisibles éloignées**.

---

## 1. Changements de paramètres

### 1.1 Réduire vision_radius (5 → 2)

`BrainConfig.vision_radius` passe de 5 à 2. La fenêtre égocentrique est
maintenant 5×5 (vs 11×11 avant).

Conséquences :
- L'agent ne voit la food que dans un rayon de 2 cases
- L'observation devient `5 × 25 + 3 + vocab_dim = 128 + 16 = 144 dim`
  (au lieu de 5 × 121 + 3 + 16 = 624)
- Brain plus compact, mais aussi moins informé

### 1.2 Augmenter listen_radius (5 → 10)

`VocabularyConfig.listen_radius` passe de 5 à 10. Les agents s'entendent
maintenant à 10 cases Manhattan.

Conséquences :
- Un agent qui voit la food peut **signaler** à des agents à 10 cases
- Les agents distants ne voient PAS la food mais peuvent l'**entendre**
- Asymétrie clé : **vision < portée vocale** → la communication devient
  un canal d'information unique

### 1.3 Maintenir tout le reste

- biomes, affinity, seed bank, repro : inchangés
- DQN cfg : lr=1e-4, target_sync=200, reward_clip [-3, +3], grad_clip 0.5
- vocalize_energy_cost = 0.05 (coût conservé)
- 4 tokens, embedding_dim=16

---

## 2. Hypothèse falsifiable

**H0 (null)** : la communication restera décorative. Les agents trouveront
des stratégies sans communication (exploration aléatoire, mémoire implicite
de la map, suivi de voisins visibles).
→ Ablation @ t=15k : pop et lifespan restent dans ±5 %

**H1** : la communication devient fonctionnelle. Couper le canal entraîne
une baisse de la coordination → moins de food trouvée → pop chute après
l'ablation.
→ Ablation @ t=15k : pop / births / lifespan baissent de >20 %

**H2 (intermédiaire)** : émergence partielle. Effet observable mais modéré.
→ Ablation : 5-20 % de baisse

---

## 3. Protocole expérimental

### 3.1 Témoin V8-C.A
```bash
python scripts/overnight_v8b1.py \
    --ticks 30000 --seed 42 --device cpu \
    --regime coordination --snap-every 2500 \
    --out-dir results/v8c_a_control
```

### 3.2 Ablation V8-C.A @ t=15k
```bash
python scripts/overnight_v8b1.py \
    --ticks 30000 --seed 42 --device cpu \
    --regime coordination --snap-every 2500 \
    --vocalize-disable-after 15000 \
    --out-dir results/v8c_a_ablation_15k
```

### 3.3 Comparaison
```bash
python scripts/compare_ablation.py \
    --control results/v8c_a_control/overnight_v8b1_seed42.json \
    --ablation results/v8c_a_ablation_15k/overnight_v8b1_seed42.json \
    --ablation-tick 15000 \
    --out-json results/v8c_a_compare.json
```

---

## 4. Métriques d'intérêt spécifiques

En plus des métriques V8-B2.x standard :

- **food_per_agent_per_tick** avant/après ablation : si la communication
  aidait à trouver la food, ce ratio doit chuter
- **distance moyenne au food** : si les agents trouvent mal la food sans
  signaux, ils devraient parcourir plus de cases avant de manger
- **n_starvation_deaths** : morts par énergie négative

Ces 3 métriques sont déjà capturées implicitement (via lifespan et naissances)
mais on pourra les exposer plus finement en V8-C.B si besoin.

---

## 5. Pièges anticipés

1. **Le DQN doit RE-apprendre** une politique compatible avec obs_dim=144
   (vs 624 avant). Les brains V8-B2.1 sauvés sont inutilisables.
2. **Cold start aggravé** : avec vision_radius=2, les agents ont moins
   d'information → mortalité initiale élevée possible. Augmenter
   `start_energy` si nécessaire.
3. **Listen_radius=10 est cher** computationnellement (chaque agent
   scan plus de voisins). Si speed < 5 t/s, réduire listen_radius à 8.
4. **L'ablation peut être MASQUÉE** par le DQN qui se ré-adapte après
   t=15k. On observera la pop à plusieurs tick-windows.

---

## 6. Critères de succès V8-C.A

V8-C.A livré (`v0.8.15-alpha`) quand :

- [ ] Régime `coordination` dans overnight_v8b1 + smoke 10k OK
- [ ] Témoin 30k tourne sans crash, pop stable
- [ ] Ablation 30k tourne sans crash
- [ ] Comparaison montre soit H1 (chute >20%), soit H0 (pas d'effet),
  soit H2 (effet modéré) — **dans les 3 cas c'est un résultat**
- [ ] Finding documenté + tag git

---

## 7. Implications selon le résultat

| Résultat | Implication |
|---|---|
| **H1 confirmée** (chute >20%) | Communication est devenue **fonctionnellement nécessaire**. Le langage est sélectionné. Publication scientifique majeure. |
| **H0 confirmée** (pas d'effet) | Vision réduite ne suffit pas. Soit les agents compensent (mémoire ?), soit il faut combiner avec coordination forcée (V8-C.B). |
| **H2 intermédiaire** | Émergence partielle. Continuer avec V8-C.B / V8-C.C pour amplifier la pression. |

Le résultat sera **scientifique dans les 3 cas**.
