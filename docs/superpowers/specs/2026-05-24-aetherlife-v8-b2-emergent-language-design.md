# V8-B2.0 — Langage émergent (vocalize + reward social)

> **Statut** : spec post-V8-B1.9 (fondation stable acquise).
> **Date** : 2026-05-24
> **Prédécesseur** : V8-B1.9 (`v0.8.9-alpha`, commit `43f1b01`).
> **Pivot historique** : on passe d'évolution cognitive à **émergence de langage**.

---

## 0. Hypothèse fondamentale

Référence : DeepMind / MILA emergent communication (Lazaridou 2017,
Foerster 2016). **Le langage n'émerge que sous pression coopérative.**

Sans pression incitative (reward bonus si la communication aide), les
agents convergent vers des vocalisations aléatoires inutiles (token spam,
ou pas de vocalize du tout).

**Mécanisme V8-B2.0** :
1. Action `vocalize(token_id)` ajoutée au action space (4 mouvements →
   4 + N_VOCAB actions discrètes)
2. Action `listen` est passive : l'agent observe les tokens prononcés
   dans un rayon R lors de la dernière étape
3. Reward social : si je vocalize au tick t, et qu'un voisin mange une
   food au tick t+1..t+3, je reçois +α (l'auditeur a "agi suite à mon
   alerte")
4. Héritage : à la reproduction, l'enfant hérite du **dictionnaire
   d'embeddings** (un vecteur par token) du parent, avec mutation
   gaussienne

Conséquence théorique : une lignée dont les agents s'avertissent
mutuellement des sources de food (token X = "food là-bas") survit
mieux, propage son dictionnaire ⇒ **émergence d'un proto-langage par
sélection**.

---

## 1. Architecture

### 1.1 `Vocabulary` (par lignée)

```python
@dataclass(frozen=True)
class VocabularyConfig:
    enabled: bool = False
    n_tokens: int = 4              # vocab size par lignée (commence petit)
    embedding_dim: int = 16        # dim de chaque embedding
    mutation_std: float = 0.05     # bruit gaussien sur embeddings à l'héritage
    listen_radius: int = 5         # voisins audibles (Manhattan)


class Vocabulary:
    """Dictionnaire de N tokens, chacun = vecteur dense R^d."""

    embeddings: np.ndarray  # shape (n_tokens, embedding_dim)
    usage_count: np.ndarray  # shape (n_tokens,) — combien de fois utilisé

    def inherit(self, rng) -> "Vocabulary":
        """Clone + mutation gaussienne sur les embeddings."""

    @classmethod
    def random(cls, cfg, rng) -> "Vocabulary":
        """Init random."""
```

Une `Vocabulary` est attachée au `LineageBrain` (un vocab par lignée).

### 1.2 Action space étendu

Avant V8-B2 : `n_actions = 4` (NORTH/SOUTH/EAST/WEST).
Après V8-B2 : `n_actions = 4 + N_VOCAB` (default 4+4=8).

Action 0..3 = mouvement (comme avant).
Action 4..4+N-1 = vocalize_token(action-4).

Quand l'agent émet l'action vocalize_token(i), il :
- ne se déplace pas ce tick
- enregistre `agent.last_vocalize = (tick, token_i)` dans son état
- déclenche `_emit_token(agent, i)` dans l'env

### 1.3 Token broadcasting

À chaque step, l'env collecte les tokens émis par chaque agent :
```python
self._tokens_this_tick: dict[int, int] = {}  # agent_id -> token_id

def step():
    self._tokens_this_tick = {}
    for agent in alive:
        action = actions[agent.agent_id]
        if action >= 4:  # vocalize
            token_id = action - 4
            self._tokens_this_tick[agent.agent_id] = token_id
            agent.last_vocalize_tick = self._step_count
        else:
            # mouvement
            ...
```

### 1.4 Observation étendue : canal "tokens entendus"

Avant V8-B2 : 5 canaux × (2r+1)² + 3 = 608 dim (vision_radius=5).
Après V8-B2 : ajout d'un vecteur **agrégé** des embeddings entendus
dans la fenêtre de vision.

```python
# Pour chaque agent, agréger les embeddings des tokens prononcés par
# les voisins dans listen_radius (Manhattan ≤ R)
heard_embeddings = []
for other in agents:
    if not other.alive or other.agent_id == agent.agent_id:
        continue
    if manhattan(other.pos, agent.pos) > listen_radius:
        continue
    if other.agent_id in env._tokens_this_tick:
        token_id = env._tokens_this_tick[other.agent_id]
        # Récup l'embedding du token via le brain de l'AUDITEUR
        # (chaque lignée a son propre dict, donc même token_id != même sens)
        own_vocab = registry.get(agent.root_ancestor_id).vocabulary
        heard_embeddings.append(own_vocab.embeddings[token_id])

# Agrégation : moyenne, ou max pooling
if heard_embeddings:
    heard_vec = np.mean(heard_embeddings, axis=0)
else:
    heard_vec = np.zeros(embedding_dim)
```

Donc obs étendue : `608 + embedding_dim = 624 dim` (avec embedding_dim=16).

Note critique : on n'utilise PAS l'embedding du locuteur — on utilise
celui de l'AUDITEUR pour son propre token_id. Cela permet la divergence
linguistique entre lignées (même token, sens différents).

### 1.5 Reward social

À chaque tick, on tracke "qui a parlé à qui" :
```python
# Au tick t : qui vocalize ?
speakers_at_t = self._tokens_this_tick.copy()

# Au tick t+1..t+3 : un voisin du speaker a mangé food ?
for speaker_id, token_id in speakers_at_t.items():
    speaker = env.agent_state(speaker_id)
    for other in env._agents:
        if not other.alive:
            continue
        if manhattan(other.pos, speaker.pos) > listen_radius:
            continue
        if other.ate_this_tick or other.ate_next_3_ticks:
            # Bonus pour le speaker
            social_rewards[speaker_id] = social_rewards.get(speaker_id, 0) + 0.3
```

Implémentation simple : on tracke `recent_speakers: dict[agent_id, ticks_since_speak]`
et on ajoute un bonus dans le reward shaping de `observe_dict`.

### 1.6 Héritage vocabulary

Dans `_try_reproductions` :
```python
if vcfg.enabled:
    parent_brain = registry.get(parent.root_ancestor_id)
    child_brain = registry.get(child.root_ancestor_id) or ...
    child_vocab = parent_brain.vocabulary.inherit(rng)
```

À la création d'un nouveau brain (via inherit_from), copy le vocab du
parent + mutate.

À la seed bank : le brain archivé garde son vocab. À la résurrection, le
nouveau fondateur hérite du vocab archivé.

---

## 2. V8-B2.0 minimum viable (cette session)

Pour rester scope, V8-B2.0 implémente :
- VocabularyConfig + Vocabulary class
- Action space étendu (4+N)
- Token broadcasting basique
- Obs étendue avec heard_embeddings (mean pooling)
- Reward social simple (+0.3 par voisin food eaten dans 3 ticks)
- Héritage vocab à la repro

**Non-objectifs V8-B2.0** :
- Pas de translator LLM (vient en V8-B2.1)
- Pas de dialectes inter-lignées (les lignées sont déjà séparées par
  affinity)
- Pas d'écriture (V8-B2.2+)

---

## 3. Critères de succès V8-B2.0

V8-B2.0 livré (`v0.8.10-alpha`) si run 30k speciation+langage produit :

- [ ] Tests : `pytest -q` ≥ 380 passed
- [ ] Vocab usage non-trivial : au moins 50 % des tokens utilisés par
  lignée (pas que le token #0 spam)
- [ ] Reward social positif : `sum(social_rewards) > 0` cumulé sur 30k
- [ ] Les vocabularies des lignées **divergent** (KL/distance entre
  vocabularies > seuil)
- [ ] Survie pop ≥ V8-B1.9 baseline (pas de régression écologique)

---

## 4. Plan TDD bite-sized

1. `VocabularyConfig` + tests validation
2. `Vocabulary` class (random init, inherit+mutation, usage tracking)
3. `BrainConfig` accepte `n_actions_total` (4 + n_vocab) via vocab config
4. `LineageBrain.vocabulary` attribute
5. `LineageBrain.inherit_from` propage vocab
6. Env reconnaît actions >= 4 comme vocalize (pas mouvement)
7. Env stocke `_tokens_this_tick` dict
8. `lineage_agent.egocentric_obs` étend avec heard_embeddings
9. Reward social dans `observe_dict`
10. Heritage vocab à la repro / respawn / seed bank
11. Tests intégration B2.0
12. Run 30k validation

---

## 5. Pièges anticipés

1. **Token spam** : sans pénalité, l'agent peut vocaliser tout le temps.
   Mitigation : pas de bonus si pas de listener proche.
2. **Reward social trop fort** → agents se vocalize pour spammer le
   bonus, négligent food. Mitigation : α=0.3 max, beaucoup moins que
   manger food (=1.8).
3. **Vocabularies convergent vers zéro** par mutation cumulée. Mitigation :
   l2 norm clip sur embeddings (norme=1.0).
4. **Action space change** brise les cerveaux V8-B1.9 saved. C'est OK :
   on repart from scratch en B2.
5. **Heard embeddings pris du brain de l'AUDITEUR** : critique pour la
   divergence des langues. Bug si on prend du locuteur.
