# Affinity Diversity Causally Buffers Population Survival via an Emergent Portfolio Effect in a Multi-Agent Reinforcement-Learning Ecosystem

**Preprint draft — AetherLife V8-C3 — 2026-06-02**
*(Quasi-paper consolidant le finding causal le plus robuste du programme. Source des
données et findings : `docs/findings/2026-06-01-finding-v8c3-diversity-as-ecological-insurance.md`,
`2026-06-01-finding-v8c3-c2-affinity-diversity-causal.md`.)*

---

## Abstract

We report a causal, dose-dependent, and mechanistically explained effect in
AetherLife, a multi-agent reinforcement-learning (MARL) ecosystem where lineages
of DQN agents forage, reproduce, and inherit policies under spatially structured
resources (Voronoi biomes with inherited habitat affinities). Using a **paired
intervention** on the number of founder affinities (`k ∈ {1, 2, 4}`, same seed →
same world, only initial affinity diversity varies), we find that affinity
diversity **causally reduces extinction risk**: extinction falls monotonically
from **60 %** (k=1) to **30 %** (k=2) to **10 %** (k=4) across 10 paired seeds. By
re-recording fine-grained trajectories through the founding demographic bottleneck,
we identify the mechanism: diversity maintains **multiple affinity×biome reservoirs
whose demographic fluctuations are temporally desynchronised** (cross-reservoir
asynchrony std ≈ 395 ticks for k=4 vs 0 for k=1, the latter trivially having a
single reservoir). When one reservoir collapses, others persist or rebound,
buffering the aggregate population away from the extinction floor (minimum
population ≈ 13.6 for k=4 vs ≈ 3.2 for k=1). This is the signature of the
**portfolio / insurance hypothesis** of biodiversity (Yachi & Loreau 1999): an
ecological principle that **emerges** here without being explicitly coded. We argue
this is a clean instance of *verifiable emergence* and discuss its relation to a
companion negative result — that the system's village-vs-migration spatial mode is
neither ecologically nor policy-determined, but historically contingent.

## 1. Introduction

A central goal of artificial-life and MARL ecosystems is to observe whether
non-trivial collective regularities **emerge** from local rules rather than being
hand-coded. Most reported regularities are correlational and rarely survive direct
intervention. Here we present a regularity that (i) is established by a paired
causal intervention, (ii) is dose-dependent and monotonic, (iii) is mechanistically
explained at the level of sub-population dynamics, and (iv) recovers a known
ecological principle that was never encoded.

The system, AetherLife (V8-C3), places lineages of agents in a 40×40 grid
partitioned into Voronoi **biomes**. Each lineage carries an inherited **biome
affinity**: foraging is rewarded (food ×1.3) inside its affinity biome and
penalised outside (food ×0.7, movement ×2.5). Agents act via per-lineage DQN
policies inherited with Gaussian mutation. Populations pass through a sharp
**founding demographic bottleneck** early in each run (a "creux", typically
t ≈ 600–900), after which the survivors rebuild.

## 2. Methods

**Intervention.** We add a single configuration lever, `n_initial_affinities`
(k), controlling how many distinct affinities the 20 founders receive
(round-robin). k=4 (default) gives a balanced 5/5/5/5; k=1 forces a single affinity
(all founders in one biome). The manipulation is **surgical** (one line in the
environment reset; positions, RNG, population size, and biome map are unchanged for
a given seed) and **paired**: each of 10 seeds is run under k=1, 2, 4, so the
Voronoi world is identical across conditions and only initial affinity diversity
differs. Runs are 16 000 ticks, CUDA.

**Survival readout.** Per condition we record extinction (final population = 0),
mean final population, and gather (cooperative foraging) successes.

**Mechanism readout.** To resolve the bottleneck dynamics, we re-record fine
trajectories (8 seeds × {k=1, k=4}, 4 000 ticks, sampling every 5 ticks) capturing
per-tick alive count, positions, and affinities. We compute, per run: minimum
population (creux depth), number of distinct affinities and occupied biomes at the
creux (reservoir count), and **crash asynchrony** = the standard deviation, across
affinities, of the tick at which each affinity sub-population reaches its own
minimum. High asynchrony indicates temporally decorrelated reservoirs.

## 3. Results

### 3.1 Affinity diversity causally reduces extinction (dose-response)

| Condition | Extinction (/10) | Mean final population | Mean gather successes |
|---|---|---|---|
| k=1 (mono) | **6/10 (60 %)** | 24.2 | 26.1 |
| k=2 | **3/10 (30 %)** | 43.0 | 42.6 |
| k=4 (multi) | **1/10 (10 %)** | 55.6 | 94.7 |

The effect is monotonic and large across the paired seeds. Cooperative foraging
also scales ×3.6 from k=1 to k=4.

### 3.2 Mechanism: asynchronous reservoirs buffer the bottleneck

| | Min population (creux) | Affinities @creux | Biomes @creux | Crash asynchrony |
|---|---|---|---|---|
| k=1 (n=8) | **3.2** (down to 1) | 1.0 | 1.2 | **0** |
| k=4 (n=8) | **13.6** | 3.0 | 3.1 | **395** |

k=1 confines all founders to one biome: a single reservoir that crashes as a block,
leaving the aggregate on a knife edge (down to 1 survivor). k=4 maintains ~3
affinity×biome reservoirs whose minima occur at markedly different times (asynchrony
≥ 94 in every k=4 run). Example (seed 4, k=4, population by affinity): affinity 0
dies early (one reservoir lost); affinity 2 bottoms at t≈400 then **rebounds**
(5 → 53) and carries the population; affinities 1 and 3 hold. The aggregate's creux
is only 18, far above the floor — the variance-reducing average of decorrelated
sub-populations.

### 3.3 An emergent portfolio effect

The dynamics match the **insurance / portfolio hypothesis of biodiversity** (Yachi &
Loreau 1999): functionally redundant types with **asynchronous** responses to
fluctuations stabilise aggregate function, just as a diversified portfolio reduces
return variance. Crucially, this principle is **not encoded** in AetherLife —
there is no fitness term rewarding diversity, no explicit risk-spreading rule. It
emerges from the interaction of Voronoi biomes, inherited affinities, and
demographic selection.

### 3.4 Generality: functional, not spatial diversity (topology experiment)

To test whether the effect is an artifact of the specific 8-seed Voronoi geography —
and whether the buffering reservoir is *spatial* (multiple patches) or *functional*
(multiple types) — we varied the spatial granularity of the partition,
`n_seed_points ∈ {4, 8, 16}`, crossed with `k ∈ {1, 4}` (8 paired seeds/cell). At
`n_seed_points=16`, a mono-affinity population (k=1) has its single affinity scattered
across ~4 disjoint patches: if reservoirs were merely spatial, this should rescue it.

| extinction | n=4 | n=8 | n=16 |
|---|---|---|---|
| k=1 (mono) | 3/8 | 5/8 | 5/8 |
| k=4 (multi) | 1/8 | 1/8 | 3/8 |

Two results. (i) **The effect generalises**: k=4 has lower extinction than k=1 at
every topology (gaps +25/+50/+25 pp) — not a Voronoi-8 artifact. (ii) **The mechanism
is functional, not spatial**: fragmenting space does NOT rescue the monoculture (k=1
extinction 38→62 %, never falls; k=1@n16 = 62 % vs k=4@n4 = 12 %, no convergence).
Spatial copies of the **same** affinity respond identically to the environment, so
their demographic fluctuations remain **correlated** (synchronous) and provide no
portfolio buffering. Only **distinct types** yield the decorrelated responses the
effect requires. This explicitly tests and rejects the spatial-multiplicity
alternative, sharpening the mechanism: *a reservoir is a distinct ecological response,
not a spatial location.* (Secondary observation: fine fragmentation is itself a
stressor — even k=4 worsens at n=16, extinction 12→38 %, gather 102→43 — raising the
coordination cost; this confounds the n=16 cell but not the main result, since spatial
multiplicity would *lower* k=1 extinction, whereas it rises.)

## 4. Discussion

This is a clean case of **verifiable emergence**: a regularity that survives a
paired causal intervention, is dose-dependent, mechanistically resolved, and maps
onto an a-priori ecological law. The companion result clarifies its scope: the
same lineage of experiments asked what drives the *spatial mode* of survivors
(sedentary "village" vs relocating "migration") and found, by elimination, that it
is **neither ecological** (local food dynamics are identical across modes) **nor
policy-determined** (village and migration lineages have statistically
indistinguishable DQN policies under a standardized probe battery). The spatial
mode appears **historically contingent** — a stochastic realisation of the same
bottleneck — whereas *survival itself* is lawfully governed by affinity diversity.
The contrast is informative: diversity sets the **probability of passing** the
bottleneck (lawful, portfolio effect); the **particular path** taken by survivors
is contingent.

## 5. Limitations

- Mechanism quantified on 8 seeds/condition over 4 000 ticks (0 extinctions in this
  window: creux depth is a proxy; the extinction gradient itself is the 16 000-tick
  result, §3.1).
- `crash_async` is structurally 0 for k=1 (single affinity); the real signal is the
  joint depth + reservoir-count + measured asynchrony in k=4.
- The behavioural origin of reservoir asynchrony (do lineages explore differently?)
  is not isolated and would require policy/representation introspection.
- One regime (`coordination_collective`, 4 biomes); generality across topologies is
  untested.

## 6. Conclusion

Affinity diversity causally buffers population survival in AetherLife, monotonically
and mechanistically, through an emergent portfolio effect of asynchronous
affinity×biome reservoirs. The result instantiates a known ecological insurance
principle without encoding it, and stands as the program's most robust causal
finding.

## References

- Yachi, S. & Loreau, M. (1999). *Biodiversity and ecosystem productivity in a
  fluctuating environment: the insurance hypothesis.* PNAS 96(4): 1463–1468.
- Companion findings (this repository): C2 causal intervention
  (`2026-06-01-finding-v8c3-c2-affinity-diversity-causal.md`), mechanism §8,
  mobility contingency (`2026-06-02-finding-v8c3-h2-policy-refuted-mobility-contingent.md`).
