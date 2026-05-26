# Update finding — V8-C2.b'' multi-seed null (falsification active)

> **Date** : 2026-05-25
> **Statut** : amendement au finding "proto-culture sans nécessité fonctionnelle"
> **Verdict** : `communication_decorative_ou_seed_dependent`

---

## 1. Le résultat à acter

Le finding préliminaire **V8-C2.b'' single-seed=42, 30k ticks, ablation@15k**
montrait **-12.9 % de naissances** après ablation. Présenté comme premier
signal causal d'un canal langagier fonctionnel.

Le **multi-seed (5 seeds × 10k ticks × 2 runs)** sous le même régime
`coordination_hard` (food invisible, vision=2, listen=10, max_pop=80) ne
reproduit PAS ce signal :

| Seed | Δ naissances | État |
|---|---|---|
| 42 | +0.5 % | survie 80, langage actif |
| 43 | +0.0 % | extinction précoce (27 births) |
| 44 | +0.8 % | survie 81 |
| 45 | +0.7 % | survie 80 |
| 46 | +0.0 % | extinction précoce (40 births) |

**Δ births mean = +0.41 % ± 0.35**, 0/5 seeds avec chute > 5 %.

---

## 2. Lecture probabiliste

Trois hypothèses également plausibles :

1. **L'effet est CUMULATIF dans le temps** : 5000 ticks post-ablation ne
   suffisent pas. Le -12.9 % à 30k correspond à 15000 ticks post-ablation,
   soit 3× plus de temps. La pénalité pourrait n'apparaître qu'avec
   beaucoup de cycles reproductifs.

2. **Le -12.9 % était un OUTLIER seed=42 long-run** : avec 1 seul tirage
   à 30k, on a vu un effet qui n'est pas une propriété systémique.

3. **Le langage n'est PAS fonctionnellement nécessaire** dans le régime
   actuel. Les autres canaux (vision locale même réduite, exploration
   aléatoire DQN, mémoire implicite) suffisent à compenser.

**Aucune n'est démontrée sans test 30k multi-seed.** L'absence de signal
à 10k ne réfute pas le -12.9 % à 30k, mais elle ne le confirme pas non plus.

---

## 3. Pourquoi ce résultat null est précieux

Le système AetherLife dispose d'un **mécanisme de falsification actif** :
1. Hypothèse formée (langage devient fonctionnel sous pression)
2. Test interventionnel posé (ablation)
3. Single-seed suggère oui
4. Multi-seed dit non
5. Hypothèse **retenue conditionnellement, avec limitation explicite**

C'est la **différence entre un projet qui valide ses biais et un projet qui
les corrige**. Sans ce multi-seed, on aurait probablement publié le -12.9 %
comme preuve. Avec le multi-seed, on sait que c'est plus fragile que ça.

---

## 4. Inventaire mis à jour des propriétés validées

### Toujours validées multi-seed (V8-B2.x)

| # | Propriété | Mesure |
|---|---|---|
| 1 | Évolution cognitive stable | loss bornée 5/5 seeds |
| 2 | Coexistence multi-lignée | 3-10 lignées finales |
| 3 | Divergence linguistique | concentration 96.85 % ± 3.67 % |
| 4 | Signal causal statistique | shift KL > 0 sur 5/5 seeds |
| 5 | Spécialisation contextuelle | ×41 baseline aléatoire |

### Non confirmées multi-seed

| # | Propriété | Statut |
|---|---|---|
| 6 | Héritage cognitif (+77 % isolé) | 1 seed (test isolé V8-B1) |
| 7 | -12.9 % naissances post-ablation | **NULL sur 5×10k** |

### Validées partiellement

| # | Propriété | Mesure |
|---|---|---|
| 8 | Convergence symbolique (token dominant 40 %) | observé V8-C2.b'' 30k, à valider multi-seed |
| 9 | Effet contextuel ×34 baseline | observé V8-B2.2 30k, à valider multi-seed |

---

## 5. Position scientifique consolidée

Au 25 mai 2026, après V8-C2.b'' multi-seed :

> **AetherLife produit, de façon reproductible sur 5 seeds, une
> proto-culture émergente** : signaux structurés, dialectes par lignée,
> spécialisation contextuelle, effet causal statistique observable.
>
> **MAIS** la nécessité fonctionnelle du canal — l'effet d'ablation sur
> la survie/fécondité — **n'est pas démontrée multi-seed** dans les
> régimes testés jusqu'ici (V8-B2.x et V8-C1/C2/C2.b''). Le -12.9 %
> observé sur un seed à 30k ne s'est pas confirmé sur 5 seeds à 10k.

Cette position est **scientifiquement honnête** et **publiable** dans cet
état exact.

---

## 6. La limite structurelle de C2

V8-C2.b'' a probablement atteint **la limite de ce qu'on peut tester
par information rare**. Tous les leviers du levier A ont été activés :
- vision réduite (2)
- food invisible
- max_pop réduit (80)
- listen radius étendu (10)

Et pourtant les agents survivent sans dépendance forte au canal. Cause
probable : **la coordination ne devient pas obligatoire**, juste utile.

Pour franchir le mur, il faut introduire une **mécanique de tâche
intrinsèquement coopérative** où un agent seul ne peut PAS résoudre.

---

## 7. Pivot validé : V8-C3 — actions coopératives lourdes

Design (avis user 2026-05-25) :

```
gather_collective :
  - nécessite ≥2 agents adjacents
  - dans les 5 ticks suivant un signal
  - récompense énergétique élevée (+30 énergie)
  - sinon : aucune récompense, voire pénalité
```

Hypothèse falsifiable :
- **H1** : sous régime coordination_collective + ablation @ t=15k →
  effondrement reproductif (-20 % ou plus de naissances), pop diminue,
  lignées s'éteignent
- **H0** : encore null. Le canal n'est intrinsèquement pas vecteur de
  fonction, même avec tâche coopérative. → conclusion forte : revoir
  l'architecture du langage (V8-D)

Si H1 est confirmé multi-seed, on aura franchi **un cap conceptuel
majeur** : un langage sélectionné par la pression coopérative réelle.
