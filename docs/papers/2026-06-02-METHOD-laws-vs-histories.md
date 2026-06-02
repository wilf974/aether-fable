# Distinguer les lois des histoires dans un système de vie artificielle : leçons méthodologiques d'AetherLife

*Article méthodologique — 2026-06-02. Ce document ne porte pas sur un résultat
particulier, mais sur la **méthode** qui a permis de produire — et surtout de ne
PAS sur-interpréter — les résultats d'AetherLife V8-C3.*

---

## Le problème : il est facile de se raconter une histoire fausse

Dans un système de vie artificielle, on observe en permanence des régularités. Une
métrique monte quand une autre monte ; un comportement coïncide avec un trait ; un
mode apparaît chez certains seeds et pas d'autres. La tentation, omniprésente, est
de **promouvoir une corrélation au rang de cause** : « les villages sont
monoculturels, donc la monoculture produit les villages », « le langage coûte de
l'énergie, donc augmenter le coût l'étouffera ».

Ces histoires sont séduisantes, internement cohérentes, et souvent **fausses**. Le
risque n'est pas l'absence de résultats — c'est l'abondance de résultats *plausibles
mais non causaux*. La question méthodologique centrale d'AetherLife a donc été :

> **Comment éviter de se raconter une histoire fausse ?**

## La méthode : un cycle, pas une métrique

La réponse n'a pas été de mesurer mieux, mais d'**intervenir**. Chaque hypothèse de
driver a suivi le même cycle :

```
   corrélation  →  instrument  →  intervention  →  réfutation  →  nouveau modèle
```

- **Corrélation** : on repère un lien apparent (X ↔ Y).
- **Instrument** : on construit l'outil qui rend X *manipulable* ou Y *mesurable
  proprement* — pas juste observable.
- **Intervention** : on force X et on regarde Y (au lieu de constater leur
  co-variation passive).
- **Réfutation** : le plus souvent, l'intervention casse l'histoire naïve.
- **Nouveau modèle** : la réfutation rétrécit l'espace des explications et fait
  émerger un modèle plus juste.

Le point clé est que **chaque réfutation est un progrès**, pas un échec. Elle
élimine une explication plausible et rapproche du mécanisme réel.

## Trois cas emblématiques

### Cas 1 — Mobilité et affinité : la corrélation à causalité inversée

```
  C0  corrélation  : mono-affinité  ↔  village (sédentarité)
         ↓ intervention (forcer la diversité d'affinité : k=1/2/4)
  C2  réfutation   : mono FORCÉE → EXTINCTION, pas village.  « mono → village » ❌
         ↓ analyse temporelle (précédence settle vs monoculture)
  C3  nouveau modèle : monoculture & village CO-ÉMERGENT d'un goulot commun.
                       Ni « mono → village » ni « village → mono ».
```

L'histoire naïve (« la monoculture cause le village ») était non seulement fausse :
la flèche n'allait dans *aucun* des deux sens supposés. Sans l'intervention, on
aurait publié une causalité inversée. **La corrélation était réelle ; son
interprétation, fausse.**

### Cas 2 — Coût vocal : l'intuition monotone qui s'inverse

```
  Hypothèse intuitive : coût du signal ↑  →  émergence linguistique ↓ (étouffement)
         ↓ intervention (vocalize_cost 0.1, 0.2)
  P2  réfutation : à coût élevé, l'émergence ne s'étouffe pas — elle se MAINTIENT,
                   voire remonte. Le coût ne supprime pas le signal : il le TRIE
                   (sélection du signal honnête, cf. handicap de Zahavi).
```

L'intuition « plus cher → moins de signal » est linéaire et plausible. L'expérience
montre l'inverse : le coût agit comme **filtre de sélection**, pas comme
suppresseur. Encore une histoire séduisante démontée par l'intervention.

### Cas 3 — Driver de mobilité : la contingence par élimination instrumentée

```
  H1  écologie cachée (la pop suit-elle la nourriture ?)
         ↓ Recorder V2 (food par tick)
      réfuté : profils food village ≈ mobile ; la pop ne suit pas la food.
  H2  politique apprise (les lignées ont-elles des stratégies différentes ?)
         ↓ OBS Viewer 3 (Q-values sur batterie de sondes standardisées)
      réfuté : politiques village/mobile statistiquement indistinguables.
         ↓
  H3  CONTINGENCE HISTORIQUE (seule explication restante)
```

Ici, aucune intervention n'a *confirmé* un driver — toutes l'ont *réfuté*. Le
résultat est un **négatif de très haute qualité** : on ne dit pas « on ne sait
pas », on dit « ce n'est NI l'écologie NI la politique, donc c'est contingent ».
La contingence n'est pas invoquée par paresse : elle arrive après une chaîne
d'instruments qui ferment les autres portes.

## La leçon de fond : deux couches

En traitant ainsi chaque hypothèse, le système a révélé deux régimes de causalité
**coexistants** :

| Couche 1 — LES LOIS | Couche 2 — L'HISTOIRE |
|---|---|
| diversité → survie | village vs migration |
| reproductible, dose-réponse | mêmes règles, mêmes politiques, même écologie |
| mécanisme compris (effet portefeuille) | → résultat différent |
| **prédictible** | **contingent** |
| *« pourquoi certaines populations survivent ? »* | *« pourquoi deux survivantes prennent des histoires différentes ? »* |

Certaines propriétés du système sont des **lois reproductibles** (l'effet
portefeuille : la diversité fixe la probabilité de passer le goulot démographique).
D'autres sont des **trajectoires historiques contingentes** (le mode spatial des
survivants : le chemin pris ensuite, indéterminé par l'environnement comme par la
stratégie).

> **Les distinguer exige des instruments, pas seulement des métriques.**

Une métrique vous dit *que* X et Y co-varient. Seul un instrument — qui manipule X,
ou qui isole Y de ses confonds — vous dit *si* l'un cause l'autre, ou si la
régularité est en réalité une contingence. La différence entre une loi et une
histoire n'est pas visible dans les données passives ; elle ne se révèle que sous
intervention.

## Pourquoi ce cadre est le plus durable

Dans six mois ou deux ans, AetherLife aura peut-être dix nouveaux findings.
L'effet portefeuille sera un résultat parmi d'autres. Ce qui restera la **colonne
vertébrale** du programme, c'est :

1. **Le cadre** : *loi vs histoire* — ne pas confondre une régularité reproductible
   avec une trajectoire contingente.
2. **La méthode** : *corrélation → instrument → intervention → réfutation → nouveau
   modèle* — et l'acceptation que la plupart des interventions réfutent, et que
   c'est précisément ce qui fait avancer le modèle.

C'est aussi ce qui explique pourquoi les résultats d'AetherLife méritent d'être pris
au sérieux : non parce qu'ils sont nombreux ou spectaculaires, mais parce que chacun
a survécu (ou non) à une tentative explicite de le réfuter. Un système de vie
artificielle qui peut **distinguer ses lois de ses histoires** ne produit plus des
anecdotes émergentes — il produit de la connaissance falsifiable.

## Annexe — pièges concrets évités par la méthode

- **Petit N** : la bimodalité convention/coordination (10 seeds) ne réplique pas à
  50 seeds → artefact d'échantillon. Toujours répliquer avant d'interpréter.
- **Fenêtre = définition** : la mobilité mesurée sur les 10 % initiaux capte la
  *fondation* (quasi universelle), pas la *migration installée*. Le choix de fenêtre
  encode ce qu'on appelle « mobilité ». L'instrument doit être calibré avant la
  mesure.
- **Confond de contexte** : comparer des politiques sans neutraliser le biome aurait
  lu « contexte observé différent » comme « politique différente ». La revue a
  imposé la neutralisation — sans elle, un faux positif H2.
- **Collinéarité** : la diversité survivante « prédisait » la mobilité, mais
  s'effondrait sous contrôle du creux (proxy collinéaire). La corrélation partielle
  sépare le proxy du driver.
