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
            self.detect_language, self.detect_cognition,
            self.detect_selection, self.detect_instability,
        ):
            try:
                out.extend(fn())
            except Exception:
                # Fail-safe : un détecteur défaillant ne casse pas le rapport
                continue
        return out
