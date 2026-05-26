"""DiscoveriesDetector — détecte des patterns dans les métriques d'un run.

Stricte exigence : **langage probabiliste**.

Une "discovery" n'est jamais une affirmation absolue. C'est un pattern
détecté avec :
    - un nom court (slug)
    - un niveau de confiance ∈ [0, 1]
    - une description courte ("Le pattern suggère...")
    - les preuves observables (valeurs métriques)
    - les pistes de validation ultérieure (ablation, multi-seed, etc.)

Catégories :
    - regime          : monoculture, coexistence, queue longue
    - extinction      : crash démographique, extinction terminale
    - language        : dialectes, vocabulaire actif, monopole token
    - cognition       : héritage cognitif observable
    - selection       : élimination des lignées faibles
    - instability     : divergence numérique, oscillation loss

Aucune discovery n'est "vraie". Toute discovery est un signal à VÉRIFIER
(multi-seed, ablation, hypothèse comportementale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiscoveryCategory(str, Enum):
    REGIME = "regime"
    EXTINCTION = "extinction"
    LANGUAGE = "language"
    COGNITION = "cognition"
    SELECTION = "selection"
    INSTABILITY = "instability"
    COOPERATION = "cooperation"


@dataclass(frozen=True)
class Discovery:
    """Pattern détecté avec preuves probabilistes.

    Champs :
        slug         : identifiant court (e.g. "language_dialects_emerging")
        category     : DiscoveryCategory
        confidence   : score [0, 1] basé sur les seuils franchis
        headline     : phrase probabiliste, ne JAMAIS affirmer
        evidence     : dict des valeurs métriques observées
        validation   : list[str] de pistes de validation ultérieure
    """
    slug: str
    category: DiscoveryCategory
    confidence: float
    headline: str
    evidence: dict[str, Any] = field(default_factory=dict)
    validation: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence doit être dans [0, 1] (got {self.confidence})"
            )


class DiscoveriesDetector:
    """Détecteur de patterns dans un report JSON d'un run AetherLife.

    Usage :
        detector = DiscoveriesDetector(report_dict)
        discoveries = detector.detect_all()
    """

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report

    # ─── Accesseurs sûrs (renvoient None si manquant) ──────────────────

    def _get(self, *path) -> Any:
        cur: Any = self.report
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
                cur = cur[k]
            else:
                return None
        return cur

    # ─── Détecteurs par catégorie ──────────────────────────────────────

    def detect_regime(self) -> list[Discovery]:
        """Régimes démographiques : monoculture, coexistence, queue longue."""
        n_lineages = self._get("criterion_3_selection", "n_lineages_final")
        dom_pct = self._get("final_state", "top_lineages", 0, "pct")
        n_alive = self._get("final_state", "n_alive")
        if n_lineages is None or dom_pct is None:
            return []
        out: list[Discovery] = []
        if n_alive == 0:
            return out  # géré dans detect_extinction
        if dom_pct >= 70.0 and n_lineages <= 2:
            out.append(Discovery(
                slug="regime_monoculture",
                category=DiscoveryCategory.REGIME,
                confidence=min(1.0, (dom_pct - 70) / 30 + 0.6),
                headline=(
                    f"Pattern suggère un régime de monoculture : "
                    f"{n_lineages} lignées vivantes, dominance "
                    f"{dom_pct:.1f} % par la lignée principale."
                ),
                evidence={"n_lineages_final": n_lineages, "dominant_pct": dom_pct},
                validation=[
                    "Run multi-seed pour confirmer la dominance reproductible",
                    "Vérifier la viabilité du régime sur 100k+ ticks",
                ],
            ))
        elif dom_pct >= 70.0 and n_lineages >= 5:
            out.append(Discovery(
                slug="regime_long_tail",
                category=DiscoveryCategory.REGIME,
                confidence=0.7,
                headline=(
                    f"Pattern suggère un régime en queue longue : "
                    f"1 lignée dominante ({dom_pct:.1f} %) + "
                    f"{n_lineages - 1} lignées marginales préservées."
                ),
                evidence={"n_lineages_final": n_lineages, "dominant_pct": dom_pct},
                validation=[
                    "Inspecter la composition des lignées marginales",
                    "Mesurer si les marginales contribuent à la diversité linguistique",
                ],
            ))
        elif dom_pct < 50.0 and n_lineages >= 3:
            out.append(Discovery(
                slug="regime_coexistence",
                category=DiscoveryCategory.REGIME,
                confidence=0.8,
                headline=(
                    f"Pattern suggère une coexistence équilibrée : "
                    f"{n_lineages} lignées vivantes, aucune dominance > 50 % "
                    f"(max {dom_pct:.1f} %)."
                ),
                evidence={"n_lineages_final": n_lineages, "dominant_pct": dom_pct},
                validation=[
                    "Mesurer la stabilité de l'équilibre sur 100k ticks",
                    "Vérifier si le pattern résiste à la mortalité accidentelle",
                ],
            ))
        return out

    def detect_extinction(self) -> list[Discovery]:
        """Extinction terminale ou crash démographique."""
        n_alive = self._get("final_state", "n_alive")
        alive_curve = self._get("curves", "alive") or []
        if n_alive == 0:
            return [Discovery(
                slug="extinction_terminal",
                category=DiscoveryCategory.EXTINCTION,
                confidence=1.0,
                headline=(
                    "Extinction terminale observée : 0 vivants en fin de run."
                ),
                evidence={
                    "n_alive_final": 0,
                    "n_deaths": self._get("final_state", "n_deaths"),
                    "n_births_total": self._get("final_state", "n_births_total"),
                },
                validation=[
                    "Identifier le tick exact d'effondrement",
                    "Reproduire avec un autre seed",
                    "Tester si extinction provient d'instabilité numérique OU "
                    "écologique (loss curve vs alive curve)",
                ],
            )]
        # Crash partiel : chute > 50 % entre deux snapshots adjacents
        out: list[Discovery] = []
        if len(alive_curve) >= 2:
            for i in range(1, len(alive_curve)):
                t_prev, n_prev = alive_curve[i - 1]
                t_cur, n_cur = alive_curve[i]
                if n_prev > 0 and n_cur / n_prev < 0.5:
                    out.append(Discovery(
                        slug="population_crash",
                        category=DiscoveryCategory.EXTINCTION,
                        confidence=0.7,
                        headline=(
                            f"Pattern suggère un crash démographique entre "
                            f"t={t_prev} ({n_prev} vivants) et t={t_cur} "
                            f"({n_cur} vivants) — réduction > 50 %."
                        ),
                        evidence={
                            "t_before": t_prev, "n_before": n_prev,
                            "t_after": t_cur, "n_after": n_cur,
                            "drop_ratio": (n_prev - n_cur) / max(n_prev, 1),
                        },
                        validation=[
                            "Hypothèse 1 : winter sévère + métabolisme élevé",
                            "Hypothèse 2 : exhaustion ressource locale",
                            "Hypothèse 3 : cascade reproductive (max_pop saturé)",
                        ],
                    ))
                    break  # premier crash seulement
        return out

    def detect_language(self) -> list[Discovery]:
        """Patterns linguistiques émergents : dialectes, vocab actif."""
        lang = self._get("language_metrics_v8b2") or {}
        if not lang:
            return []
        out: list[Discovery] = []
        conc = lang.get("mean_token_lineage_concentration", 0)
        l2 = lang.get("mean_inter_lineage_distance", 0)
        entropy_ratio = lang.get("entropy_ratio", 0)
        total_voc = lang.get("total_vocalize_count", 0)
        # Dialectes : concentration + distance
        if conc >= 0.7 and l2 >= 1.0:
            out.append(Discovery(
                slug="language_dialects_emerging",
                category=DiscoveryCategory.LANGUAGE,
                confidence=min(1.0, conc),
                headline=(
                    f"Pattern suggère l'émergence de dialectes par lignée : "
                    f"chaque token est utilisé à {conc:.1%} par une seule "
                    f"lignée, distance L2 inter-vocabs = {l2:.2f}."
                ),
                evidence={
                    "concentration_per_lineage": conc,
                    "L2_inter_lineage": l2,
                    "entropy_ratio": entropy_ratio,
                },
                validation=[
                    "Confirmation multi-seed (≥3 seeds avec concentration ≥ 70 %)",
                    "Vérifier corrélation token → contexte (food, danger, etc.)",
                    "Test causal : si on retire la pression coût/survie, "
                    "le dialecte disparaît-il ?",
                ],
            ))
        # Vocabulaire actif vs spam vs monopole
        if 0.5 < entropy_ratio < 0.95 and total_voc > 1000:
            out.append(Discovery(
                slug="language_active_vocabulary",
                category=DiscoveryCategory.LANGUAGE,
                confidence=0.7,
                headline=(
                    f"Pattern suggère un vocabulaire actif et structuré "
                    f"(entropy ratio {entropy_ratio:.1%}, ni monopole ni "
                    f"chaos uniforme). Total vocalize : {total_voc}."
                ),
                evidence={
                    "entropy_ratio": entropy_ratio,
                    "total_vocalize": total_voc,
                    "per_token_usage": lang.get("per_token_usage_top", {}),
                },
                validation=[
                    "Mesurer la distribution par lignée (vs globale)",
                    "Comparer à un baseline 'vocalize aléatoire'",
                ],
            ))
        elif entropy_ratio >= 0.95:
            out.append(Discovery(
                slug="language_uniform_token_usage",
                category=DiscoveryCategory.LANGUAGE,
                confidence=0.6,
                headline=(
                    f"Pattern suggère une distribution de tokens quasi-uniforme "
                    f"(entropy ratio {entropy_ratio:.1%}). "
                    f"Compatible avec usage aléatoire OU lignées multiples "
                    f"avec vocabularies équilibrés."
                ),
                evidence={"entropy_ratio": entropy_ratio},
                validation=[
                    "Désambiguer : décomposer entropy par lignée",
                    "Si entropy_par_lineage << log(N) : structure cachée",
                ],
            ))
        return out

    def detect_cognition(self) -> list[Discovery]:
        """Héritage cognitif : Q4 vs Q1 lifespan."""
        inh = self._get("criterion_1_inheritance", "lifespan_by_birth_quartile") or {}
        if not inh:
            return []
        q1 = inh.get("Q1_early", {}).get("mean")
        q4 = inh.get("Q4_late", {}).get("mean")
        if q1 is None or q4 is None:
            return []
        delta = q4 - q1
        out: list[Discovery] = []
        if delta > 100:
            out.append(Discovery(
                slug="cognition_inheritance_observable",
                category=DiscoveryCategory.COGNITION,
                confidence=min(1.0, delta / 1000),
                headline=(
                    f"Pattern suggère un héritage cognitif observable : "
                    f"les agents nés tard vivent en moyenne "
                    f"{delta:+.0f} ticks de plus que les fondateurs "
                    f"(Q1={q1:.0f}, Q4={q4:.0f})."
                ),
                evidence={"q1_mean": q1, "q4_mean": q4, "delta": delta},
                validation=[
                    "Test isolé gen0 vs genN (run isolé sans repro)",
                    "Mesurer la corrélation entre lifespan et generation",
                    "Reproduire sur ≥3 seeds",
                ],
            ))
        elif delta < -200:
            out.append(Discovery(
                slug="cognition_inheritance_apparent_negative",
                category=DiscoveryCategory.COGNITION,
                confidence=0.5,
                headline=(
                    f"Pattern apparent : Q4 < Q1 ({delta:+.0f} ticks). "
                    f"ATTENTION : artefact statistique probable — les agents "
                    f"nés tard n'ont pas eu le temps de mourir naturellement."
                ),
                evidence={"q1_mean": q1, "q4_mean": q4, "delta": delta},
                validation=[
                    "Limiter l'analyse aux agents qui ont VRAIMENT atteint "
                    "leur fin de vie (ex. Q1 vs Q2)",
                    "Pas conclure à 'pas d'héritage' sans run plus long",
                ],
            ))
        return out

    def detect_selection(self) -> list[Discovery]:
        """Sélection cognitive : nb lignées éteintes."""
        init = self._get("criterion_3_selection", "n_lineages_initial")
        fin = self._get("criterion_3_selection", "n_lineages_final")
        if init is None or fin is None:
            return []
        if init == 0:
            return []
        reduction = (init - fin) / init
        out: list[Discovery] = []
        if reduction >= 0.5:
            out.append(Discovery(
                slug="selection_strong",
                category=DiscoveryCategory.SELECTION,
                confidence=reduction,
                headline=(
                    f"Pattern suggère une sélection forte au niveau lignée : "
                    f"{init - fin}/{init} lignées initiales se sont éteintes "
                    f"({reduction:.0%} de réduction)."
                ),
                evidence={
                    "n_lineages_initial": init,
                    "n_lineages_final": fin,
                    "reduction_ratio": reduction,
                },
                validation=[
                    "Identifier QUELS critères différencient les lignées "
                    "survivantes des éteintes (traits, vocab, affinity)",
                    "Tester si l'extinction est cognitive (cerveau faible) "
                    "OU démographique (chance de survie)",
                ],
            ))
        return out

    def detect_causality(self) -> list[Discovery]:
        """V8-B2.2 — Patterns de causalité comportementale du langage.

        Sources : report["language_causality_v8b2_2"] = {
            listener_shift_mean, listener_shift_max, listener_shift_per_token,
            context_consistency_mean, context_consistency_per_token,
            n_emissions_total, verdict
        }

        Patterns détectables :
          - causality_signal_present : shift > 0.03 ET context > 0.40
          - causality_signal_strong  : shift > 0.10 OU context > 0.70
          - causality_decorative_refuted : shift > 0 ET context >> baseline
          - causality_context_specialization : context_per_token homogène
            et au-dessus baseline
        """
        cause = self._get("language_causality_v8b2_2") or {}
        if not cause:
            return []
        out: list[Discovery] = []
        shift_mean = cause.get("listener_shift_mean", 0)
        shift_max = cause.get("listener_shift_max", 0)
        cons_mean = cause.get("context_consistency_mean", 0)
        n_emissions = cause.get("n_emissions_total", 0)
        per_token_shift = cause.get("listener_shift_per_token", {}) or {}
        per_token_cons = cause.get("context_consistency_per_token", {}) or {}

        if n_emissions < 1000:
            # Pas assez d'émissions pour conclure
            return []

        # Pattern 1 : signal causal présent (faible mais réel)
        if shift_mean > 0.03 and cons_mean > 0.30:
            out.append(Discovery(
                slug="causality_signal_present",
                category=DiscoveryCategory.LANGUAGE,
                confidence=min(1.0, shift_mean * 5 + cons_mean / 2),
                headline=(
                    f"Pattern compatible avec signal causal modéré : "
                    f"KL listener shift = {shift_mean:.3f} mean, "
                    f"context consistency = {cons_mean:.0%}. "
                    f"Le canal n'est pas décoratif (signal > baseline)."
                ),
                evidence={
                    "listener_shift_mean": shift_mean,
                    "listener_shift_max": shift_max,
                    "context_consistency_mean": cons_mean,
                    "n_emissions_total": n_emissions,
                },
                validation=[
                    "Multi-seed pour confirmer signal robuste",
                    "Run plus long : signal croît avec apprentissage ?",
                    "Test d'ablation interventionnelle (V8-B2.3) : "
                    "couper canal → comportement change ?",
                ],
            ))

        # Pattern 2 : signal causal FORT (au moins un token > 0.10)
        if shift_max > 0.10:
            top_tok = max(
                per_token_shift.items(), key=lambda x: float(x[1]),
                default=("?", 0),
            )
            out.append(Discovery(
                slug="causality_signal_strong",
                category=DiscoveryCategory.LANGUAGE,
                confidence=min(1.0, shift_max),
                headline=(
                    f"Pattern compatible avec signal causal FORT : "
                    f"le token {top_tok[0]} produit un shift KL de "
                    f"{shift_max:.3f} (seuil 'fort' = 0.10). "
                    f"Au moins un symbole modifie statistiquement le "
                    f"comportement des auditeurs."
                ),
                evidence={
                    "shift_max": shift_max,
                    "top_token": top_tok[0],
                    "per_token_shift": per_token_shift,
                },
                validation=[
                    "Quel contexte d'émission est associé à ce token ?",
                    "Ablation sélective : si on désactive ce token "
                    "seulement, l'effet disparaît-il ?",
                ],
            ))

        # Pattern 3 : spécialisation contextuelle ×N baseline
        # Avec 4 buckets contexte par dimension × 4 dim ≈ 72 clusters
        baseline_random = 1.0 / 72.0  # ≈ 1.4%
        ratio = cons_mean / baseline_random if baseline_random > 0 else 0
        if ratio > 10:
            out.append(Discovery(
                slug="causality_context_specialization",
                category=DiscoveryCategory.LANGUAGE,
                confidence=min(1.0, (ratio - 10) / 50),
                headline=(
                    f"Pattern de spécialisation contextuelle massive : "
                    f"les tokens sont émis dans le cluster contextuel "
                    f"majoritaire à {cons_mean:.0%}, soit ×{ratio:.0f} le "
                    f"baseline aléatoire (~{baseline_random:.1%}). "
                    f"Les tokens ne sont PAS aléatoires."
                ),
                evidence={
                    "context_consistency_mean": cons_mean,
                    "baseline_random_estimate": baseline_random,
                    "ratio_to_baseline": ratio,
                    "per_token_consistency": per_token_cons,
                },
                validation=[
                    "Identifier les contextes majoritaires par token "
                    "(quel contexte → quel token ?)",
                    "Test croisé inter-lignées : un token corrèle-t-il "
                    "au même contexte dans différentes lignées ?",
                ],
            ))

        return out

    def detect_cooperation(self) -> list[Discovery]:
        """V8-C3 — Patterns d'émergence coopérative.

        Sources :
          - report["cooperative_v8c3"] : compteurs bruts (successes, failures)
          - report["cooperative_metrics_v8c3"] : 4 métriques observationnelles
            (clustering, delay, token entropy, success chains)

        Patterns détectables :
          - cooperation_apprenable : ≥ 50 succès + clustering Q4 > Q1
            → la mécanique est apprise (organisation spatiale corrélée
              au succès, pas du pur hasard)
          - cooperation_protocol_emergent : token dominant pre-success
            > 0.5 ET delay_trend < 0 → compression émergente du protocole
            (un token devient privilégié + temps de réaction diminue)
          - cooperation_cascade_attractor : n_cascade_successes /
            n_successes > 0.2 → boucle de renforcement détectée

        Tous ces patterns sont mécaniques, observables, falsifiables
        et NON sémantiques.
        """
        coop = self._get("cooperative_v8c3") or {}
        if not coop or not coop.get("enabled"):
            return []
        n_success = coop.get("gather_successes_total", 0)
        n_fail = coop.get("gather_failures_total", 0)
        if n_success == 0 and n_fail == 0:
            return []

        out: list[Discovery] = []
        metrics = self._get("cooperative_metrics_v8c3") or {}

        # ─── Pattern 1 : cooperation_apprenable ─────────────────────────
        cl = metrics.get("clustering_pre_success", {}) or {}
        cl_trend = cl.get("trend_q4_minus_q1", 0.0) or 0.0
        cl_mean = cl.get("mean_neighbors_r3", 0.0) or 0.0
        if n_success >= 50 and cl_trend > 0.0:
            # Confiance proportionnelle au signal : plus de successes + trend
            # positif plus marqué → confiance plus haute
            conf = min(1.0, 0.4 + (n_success / 500) + (cl_trend / 5.0))
            out.append(Discovery(
                slug="cooperation_apprenable",
                category=DiscoveryCategory.COOPERATION,
                confidence=conf,
                headline=(
                    f"Pattern suggère que la mécanique coopérative est "
                    f"APPRISE : {n_success} succès gather observés, "
                    f"clustering Q4-Q1 = {cl_trend:+.2f} (les agents "
                    f"convergent davantage autour des spots au fil du run). "
                    f"Densité moyenne au moment du succès : "
                    f"{cl_mean:.2f} voisins dans r=3."
                ),
                evidence={
                    "gather_successes_total": n_success,
                    "gather_failures_total": n_fail,
                    "clustering_trend_q4_minus_q1": cl_trend,
                    "clustering_mean_neighbors_r3": cl_mean,
                },
                validation=[
                    "Multi-seed pour confirmer la trend de clustering",
                    "Run plus long : clustering continue-t-il de monter ?",
                    "Comparer baseline (agents random sur même map)",
                ],
            ))

        # ─── Pattern 2 : cooperation_protocol_emergent ──────────────────
        tk = metrics.get("token_entropy_pre_success", {}) or {}
        dl = metrics.get("vocalize_to_gather_delay", {}) or {}
        dominant_share = tk.get("dominant_share", 0.0) or 0.0
        dominant_token = tk.get("dominant_token")
        delay_trend = dl.get("trend_q4_minus_q1", 0.0) or 0.0
        delay_mean = dl.get("mean_min_delay")
        if (
            n_success >= 30
            and dominant_share > 0.5
            and delay_trend < 0.0
        ):
            conf = min(
                1.0,
                0.5 + (dominant_share - 0.5) + abs(delay_trend) / 10.0,
            )
            out.append(Discovery(
                slug="cooperation_protocol_emergent",
                category=DiscoveryCategory.COOPERATION,
                confidence=conf,
                headline=(
                    f"Pattern compatible avec un PROTOCOLE COOPÉRATIF "
                    f"émergent : le token {dominant_token} concentre "
                    f"{dominant_share:.0%} des vocalisations dans la "
                    f"fenêtre 5 ticks AVANT chaque succès, et le délai "
                    f"vocalize→succès baisse "
                    f"(Q4-Q1={delay_trend:+.2f}, mean={delay_mean}). "
                    f"Compatible avec une compression émergente du signal "
                    f"coopératif."
                ),
                evidence={
                    "dominant_token": dominant_token,
                    "dominant_share_pre_success": dominant_share,
                    "delay_trend_q4_minus_q1": delay_trend,
                    "delay_mean_min": delay_mean,
                    "token_distribution_pre_success": (
                        tk.get("distribution", {})
                    ),
                },
                validation=[
                    "Multi-seed : le même token dominant émerge-t-il ?",
                    "Ablation sélective : désactiver ce token spécifique → "
                    "les succès chutent-ils ?",
                    "Comparer distribution token pre-success vs post-success "
                    "vs global : la spécialisation est-elle exclusive ?",
                ],
            ))

        # ─── Pattern 3 : cooperation_cascade_attractor ──────────────────
        ch = metrics.get("success_chains", {}) or {}
        n_cascade = ch.get("n_cascade_successes", 0) or 0
        max_chain = ch.get("max_chain_len", 0) or 0
        n_chains = ch.get("n_chains", 0) or 0
        cascade_ratio = n_cascade / max(n_success, 1)
        if n_success >= 30 and cascade_ratio > 0.2:
            conf = min(1.0, 0.4 + cascade_ratio + (max_chain / 50.0))
            out.append(Discovery(
                slug="cooperation_cascade_attractor",
                category=DiscoveryCategory.COOPERATION,
                confidence=conf,
                headline=(
                    f"Pattern suggère un ATTRACTEUR coopératif : "
                    f"{n_cascade}/{n_success} succès ({cascade_ratio:.0%}) "
                    f"se produisent dans des cascades de ≥3 succès. "
                    f"Chaîne la plus longue : {max_chain}. Compatible avec "
                    f"des boucles de renforcement (un succès → "
                    f"regroupement → autre succès)."
                ),
                evidence={
                    "n_cascade_successes": n_cascade,
                    "n_successes_total": n_success,
                    "cascade_ratio": cascade_ratio,
                    "max_chain_len": max_chain,
                    "n_chains": n_chains,
                    "n_isolated_successes": ch.get("n_isolated_successes", 0),
                },
                validation=[
                    "Inspecter manuellement une cascade : que font les "
                    "agents juste après un succès ?",
                    "Comparer cascade_ratio sur le 1er vs 4e quartile du run",
                    "Si bonus_energy baissé, la cascade disparaît-elle ?",
                ],
            ))

        # ─── Cas "diagnostic" : la mécanique existe mais aucun pattern ──
        # Si on a des succès mais aucun des 3 patterns ne se déclenche,
        # on émet une découverte "neutre" pour ne pas effacer le signal
        # mais SANS surinterpréter.
        if n_success > 0 and not out:
            out.append(Discovery(
                slug="cooperation_mechanic_active_no_pattern",
                category=DiscoveryCategory.COOPERATION,
                confidence=0.3,
                headline=(
                    f"La mécanique coopérative existe ({n_success} succès, "
                    f"{n_fail} échecs) mais aucun pattern de proto-"
                    f"coordination n'est encore détectable (clustering trend, "
                    f"protocole, cascade tous sous seuils). Compatible avec "
                    f"une mécanique non encore maîtrisée par les agents."
                ),
                evidence={
                    "gather_successes_total": n_success,
                    "gather_failures_total": n_fail,
                    "success_rate": n_success / max(n_success + n_fail, 1),
                },
                validation=[
                    "Run plus long pour laisser l'apprentissage prendre",
                    "Augmenter bonus_energy ou spawn_lambda pour amplifier signal",
                    "Vérifier que le canal gather_view est bien visible aux agents",
                ],
            ))

        return out

    def detect_instability(self) -> list[Discovery]:
        """Instabilité numérique DQN."""
        loss_curve = self._get("curves", "loss") or []
        if not loss_curve:
            return []
        max_loss = max((v for _, v in loss_curve), default=0)
        out: list[Discovery] = []
        if max_loss > 100:
            out.append(Discovery(
                slug="instability_dqn_divergence",
                category=DiscoveryCategory.INSTABILITY,
                confidence=0.8,
                headline=(
                    f"Pattern suggère une divergence numérique du DQN : "
                    f"loss max observée = {max_loss:.1f} (> 100). "
                    f"Probable explosion des Q-values."
                ),
                evidence={"max_loss": max_loss, "n_loss_points": len(loss_curve)},
                validation=[
                    "Resserrer gradient clipping",
                    "Baisser learning rate",
                    "Ajouter Huber loss",
                ],
            ))
        elif max_loss > 10:
            out.append(Discovery(
                slug="instability_loss_spikes",
                category=DiscoveryCategory.INSTABILITY,
                confidence=0.5,
                headline=(
                    f"Pattern : pics de loss observés (max {max_loss:.1f}) "
                    f"mais pas de divergence terminale. Acceptable."
                ),
                evidence={"max_loss": max_loss},
                validation=["Vérifier que ces pics ne provoquent pas d'extinction"],
            ))
        return out

    def detect_all(self) -> list[Discovery]:
        """Lance tous les détecteurs et concatène les résultats."""
        out: list[Discovery] = []
        for fn in (
            self.detect_regime, self.detect_extinction,
            self.detect_language, self.detect_causality,
            self.detect_cognition,
            self.detect_selection, self.detect_instability,
            self.detect_cooperation,
        ):
            try:
                out.extend(fn())
            except Exception:
                # Fail-safe : un détecteur défaillant ne casse pas le rapport
                continue
        return out
