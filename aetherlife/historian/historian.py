"""Historian — orchestrateur du reporting d'un run AetherLife.

Usage typique :

    from aetherlife.historian import Historian

    historian = Historian.from_report_path("results/v8b2_30k/overnight_v8b1_seed42.json")
    historian.write_all("reports/v8b2_30k")

Génère :
    summary.md, scientific_report.md, public_article.md, discoveries.md,
    lineages.md, dialects.md, metrics.json, events.jsonl, charts.csv
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from aetherlife.historian.discoveries import (
    DiscoveriesDetector, Discovery, DiscoveryCategory,
)


class Historian:
    """Observer/reporter d'un run AetherLife. Aucune influence sur les agents."""

    def __init__(self, report: dict[str, Any], run_id: str | None = None) -> None:
        self.report = report
        self.run_id = run_id or self._derive_run_id()
        self.discoveries = DiscoveriesDetector(report).detect_all()
        self.generated_at = datetime.now(timezone.utc).isoformat()

    # ─── Loaders ───────────────────────────────────────────────────────

    @classmethod
    def from_report_path(cls, path: str, run_id: str | None = None) -> "Historian":
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        if run_id is None:
            run_id = os.path.splitext(os.path.basename(path))[0]
        return cls(report=report, run_id=run_id)

    @classmethod
    def from_report(cls, report: dict[str, Any], run_id: str | None = None) -> "Historian":
        return cls(report=report, run_id=run_id)

    def _derive_run_id(self) -> str:
        cfg = self.report.get("config", {})
        seed = cfg.get("seed", "noseed")
        n_ticks = cfg.get("n_ticks", "noticks")
        return f"run_seed{seed}_t{n_ticks}"

    # ─── Helpers d'accès ───────────────────────────────────────────────

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

    # ─── Public API : write_all ────────────────────────────────────────

    def write_all(self, out_dir: str) -> dict[str, str]:
        """Génère tous les fichiers dans `out_dir`. Retourne {nom: chemin}."""
        os.makedirs(out_dir, exist_ok=True)
        files: dict[str, str] = {}
        for name, content in [
            ("summary.md", self.render_summary()),
            ("scientific_report.md", self.render_scientific_report()),
            ("public_article.md", self.render_public_article()),
            ("discoveries.md", self.render_discoveries()),
            ("lineages.md", self.render_lineages()),
            ("dialects.md", self.render_dialects()),
        ]:
            path = os.path.join(out_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            files[name] = path

        path = os.path.join(out_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._build_metrics(), f, indent=2, default=str)
        files["metrics.json"] = path

        path = os.path.join(out_dir, "events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for line in self._build_events():
                f.write(json.dumps(line, default=str) + "\n")
        files["events.jsonl"] = path

        path = os.path.join(out_dir, "charts.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            self._write_charts_csv(writer)
        files["charts.csv"] = path
        return files

    # ─── Builders structurés ───────────────────────────────────────────

    def _build_metrics(self) -> dict[str, Any]:
        """metrics.json — données brutes structurées (re-utilisables)."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "protocol": {
                "seed": self._get("config", "seed"),
                "ticks": self._get("config", "n_ticks"),
                "device": self._get("config", "device"),
                "vision_radius": self._get("config", "vision_radius"),
            },
            "runtime": self._get("runtime") or {},
            "ecology": {
                "n_alive_final": self._get("final_state", "n_alive"),
                "n_births_total": self._get("final_state", "n_births_total"),
                "n_deaths": self._get("final_state", "n_deaths"),
                "n_lineages_initial": self._get("criterion_3_selection", "n_lineages_initial"),
                "n_lineages_final": self._get("criterion_3_selection", "n_lineages_final"),
                "dominance_pct": self._get("final_state", "top_lineages", 0, "pct"),
                "affinity_distribution": self._get("final_state", "affinity_distribution"),
            },
            "cognition": {
                "kl_inter_lineages": self._get("criterion_2_divergence", "final_kl_mean"),
                "total_brain_steps": self._get("criterion_4_memory", "total_brain_steps"),
                "lifespan_by_quartile": self._get("criterion_1_inheritance", "lifespan_by_birth_quartile"),
            },
            "language": self._get("language_metrics_v8b2") or {},
            "discoveries": [
                {
                    "slug": d.slug,
                    "category": d.category.value,
                    "confidence": d.confidence,
                    "headline": d.headline,
                    "evidence": d.evidence,
                    "validation": d.validation,
                }
                for d in self.discoveries
            ],
        }

    def _build_events(self) -> list[dict[str, Any]]:
        """events.jsonl — timeline d'événements significatifs.

        Reconstruit depuis les curves (alive/lineages/loss/divergence) car on
        n'a pas accès au log brut par tick. Inclut aussi les discoveries.
        """
        events: list[dict[str, Any]] = []
        for t, n in self._get("curves", "alive") or []:
            events.append({"t": t, "type": "snapshot_alive", "n_alive": n})
        for t, n in self._get("curves", "lineages") or []:
            events.append({"t": t, "type": "snapshot_lineages", "n_lineages": n})
        for t, kl in self._get("curves", "divergence") or []:
            events.append({"t": t, "type": "snapshot_kl", "kl_inter_lineage": kl})
        for t, loss in self._get("curves", "loss") or []:
            events.append({"t": t, "type": "snapshot_loss", "mean_loss": loss})
        for d in self.discoveries:
            events.append({
                "t": None, "type": "discovery",
                "slug": d.slug, "category": d.category.value,
                "confidence": d.confidence, "headline": d.headline,
            })
        # tri par tick (None à la fin)
        events.sort(key=lambda e: (e.get("t") is None, e.get("t") or 0))
        return events

    def _write_charts_csv(self, writer) -> None:
        """charts.csv — colonnes : tick, alive, lineages, loss, kl.

        Jointure des 4 curves alignées par tick (peut avoir des lignes
        partielles si snap_every diffère, on remplit avec '').
        """
        alive = dict(self._get("curves", "alive") or [])
        lineages = dict(self._get("curves", "lineages") or [])
        loss = dict(self._get("curves", "loss") or [])
        kl = dict(self._get("curves", "divergence") or [])
        ticks = sorted(set(alive) | set(lineages) | set(loss) | set(kl))
        writer.writerow(["tick", "alive", "n_lineages", "mean_loss", "kl_inter_lineages"])
        for t in ticks:
            writer.writerow([
                t,
                alive.get(t, ""),
                lineages.get(t, ""),
                loss.get(t, ""),
                kl.get(t, ""),
            ])

    # ─── Markdown renderers ────────────────────────────────────────────

    def _h1(self, title: str) -> str:
        return f"# {title}\n\n"

    def _h2(self, title: str) -> str:
        return f"\n## {title}\n\n"

    def _kv_table(self, rows: list[tuple[str, Any]]) -> str:
        lines = ["| Métrique | Valeur |", "|---|---|"]
        for k, v in rows:
            if v is None:
                v = "non observable"
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines) + "\n"

    def render_summary(self) -> str:
        """summary.md — résumé 1 page court et factuel."""
        s = self._h1(f"AetherLife — Résumé du run `{self.run_id}`")
        s += f"_Généré : {self.generated_at}_\n\n"
        s += self._h2("Protocole")
        s += self._kv_table([
            ("Seed", self._get("config", "seed")),
            ("Ticks", self._get("config", "n_ticks")),
            ("Device", self._get("config", "device")),
            ("Durée", f"{self._get('runtime', 'duration_s'):.1f} s"
                if self._get('runtime', 'duration_s') is not None else None),
        ])
        s += self._h2("Écologie finale")
        s += self._kv_table([
            ("Vivants", self._get("final_state", "n_alive")),
            ("Naissances totales", self._get("final_state", "n_births_total")),
            ("Morts totales", self._get("final_state", "n_deaths")),
            ("Lignées vivantes", self._get("criterion_3_selection", "n_lineages_final")),
            ("Dominance top", f"{self._get('final_state', 'top_lineages', 0, 'pct'):.1f} %"
                if self._get('final_state', 'top_lineages', 0, 'pct') is not None else None),
        ])
        lang = self._get("language_metrics_v8b2") or {}
        if lang:
            s += self._h2("Langage (si activé)")
            s += self._kv_table([
                ("Total vocalize", lang.get("total_vocalize_count")),
                ("Tokens / 1000 ticks", f"{lang.get('tokens_per_1000_ticks', 0):.0f}"),
                ("Entropy ratio", f"{lang.get('entropy_ratio', 0):.1%}"),
                ("Concentration par lignée", f"{lang.get('mean_token_lineage_concentration', 0):.1%}"),
                ("Distance L2 inter-vocabs", f"{lang.get('mean_inter_lineage_distance', 0):.2f}"),
            ])
        s += self._h2("Découvertes détectées (probabilistes)")
        if not self.discoveries:
            s += "_Aucun pattern significatif détecté._\n"
        else:
            for d in self.discoveries:
                s += f"- **{d.slug}** (conf={d.confidence:.2f}) : {d.headline}\n"
        return s

    def render_scientific_report(self) -> str:
        """scientific_report.md — version rigoureuse, avec limites."""
        s = self._h1(f"Rapport scientifique : run `{self.run_id}`")
        s += f"_Généré : {self.generated_at}_\n\n"

        s += self._h2("1. Protocole expérimental")
        s += "Run réalisé avec les paramètres ci-dessous. Reproductibilité "
        s += "via le seed indiqué.\n\n"
        s += self._kv_table([
            ("Seed", self._get("config", "seed")),
            ("Nombre de ticks", self._get("config", "n_ticks")),
            ("Device d'inférence", self._get("config", "device")),
            ("Vision radius", self._get("config", "vision_radius")),
            ("Dim observation", self._get("config", "obs_dim")),
            ("Durée totale (s)", self._get("runtime", "duration_s")),
            ("Ticks / seconde", self._get("runtime", "ticks_per_sec")),
        ])

        s += self._h2("2. Métriques écologiques")
        s += self._kv_table([
            ("Vivants en fin de run", self._get("final_state", "n_alive")),
            ("Naissances cumulées", self._get("final_state", "n_births_total")),
            ("Morts cumulées", self._get("final_state", "n_deaths")),
            ("Lignées initiales", self._get("criterion_3_selection", "n_lineages_initial")),
            ("Lignées finales", self._get("criterion_3_selection", "n_lineages_final")),
            ("Dominance top (%)", self._get("final_state", "top_lineages", 0, "pct")),
            ("Affinities représentées", self._get("final_state", "n_affinities_alive")),
        ])

        s += self._h2("3. Cognition et héritage")
        q = self._get("criterion_1_inheritance", "lifespan_by_birth_quartile") or {}
        rows = []
        for k in ("Q1_early", "Q2", "Q3", "Q4_late"):
            v = q.get(k, {})
            if v:
                rows.append((
                    f"{k} (n={v.get('n')}) mean_lifespan",
                    f"{v.get('mean', 0):.0f} (median {v.get('median', 0):.0f})",
                ))
        rows.append(("KL inter-lignées", self._get("criterion_2_divergence", "final_kl_mean")))
        rows.append(("Brain steps cumulés", self._get("criterion_4_memory", "total_brain_steps")))
        s += self._kv_table(rows)

        lang = self._get("language_metrics_v8b2") or {}
        if lang:
            s += self._h2("4. Langage émergent (V8-B2+)")
            s += self._kv_table([
                ("Vocalize total", lang.get("total_vocalize_count")),
                ("Tokens / 1000 ticks", lang.get("tokens_per_1000_ticks")),
                ("Energy cost cumulé", lang.get("vocalize_energy_cost_total")),
                ("Usage entropy moyenne", lang.get("mean_usage_entropy")),
                ("Entropy max possible", lang.get("max_possible_entropy")),
                ("Entropy ratio", lang.get("entropy_ratio")),
                ("Concentration par lignée (moy)", lang.get("mean_token_lineage_concentration")),
                ("Distance L2 inter-vocabs (moy)", lang.get("mean_inter_lineage_distance")),
            ])

        s += self._h2("5. Patterns détectés (probabilistes)")
        s += "Les patterns ci-dessous sont des **hypothèses prudentes** "
        s += "déduites des métriques. Aucun n'est une affirmation définitive. "
        s += "Chaque pattern liste des pistes de validation ultérieure.\n\n"
        if not self.discoveries:
            s += "_Aucun pattern significatif détecté._\n"
        else:
            for d in self.discoveries:
                s += f"### {d.slug} (catégorie : {d.category.value}, "
                s += f"confiance : {d.confidence:.2f})\n\n"
                s += f"> {d.headline}\n\n"
                if d.evidence:
                    s += "**Preuves observées** :\n\n"
                    for k, v in d.evidence.items():
                        s += f"- `{k}` = {v}\n"
                    s += "\n"
                if d.validation:
                    s += "**Pistes de validation** :\n\n"
                    for v in d.validation:
                        s += f"- {v}\n"
                    s += "\n"

        s += self._h2("6. Limites méthodologiques")
        s += (
            "- **Mono-seed** : ce run est UN tirage. Toute conclusion nécessite "
            "une validation multi-seed (≥3 seeds avec variance bornée).\n"
            "- **Pas d'ablation** : les patterns observés peuvent venir de la "
            "config (biomes, affinity, vocab), pas d'une vraie propriété du "
            "système. Tester en désactivant chaque mécanisme.\n"
            "- **Pas de causalité comportementale** : 'concentration 99 %' "
            "n'implique pas 'les agents communiquent'. Il faudrait mesurer "
            "le changement de comportement de l'auditeur APRÈS écoute.\n"
            "- **Quartile bias** : Q4 = nés tardivement, donc lifespan "
            "tronqué par fin de run. Q1 vs Q2 plus fiable que Q1 vs Q4.\n"
            "- **Le DQN n'est pas la cognition humaine** : ne pas anthropomorphiser.\n"
        )

        s += self._h2("7. Pistes d'impact potentiel")
        s += (
            "- Validation multi-seed (≥5 seeds) du finding principal.\n"
            "- Mesure de causalité comportementale (token → action listener).\n"
            "- Run très long (100k+ ticks) pour vérifier la stabilité asymptotique.\n"
            "- Test d'ablation : désactiver biomes, désactiver vocab, désactiver "
            "respawn — quelle propriété reste ?\n"
            "- Étude des dialectes : mots dominants, contextes corrélés, "
            "topographie des emissions.\n"
        )
        return s

    def render_public_article(self) -> str:
        """public_article.md — article blog/newsletter accessible."""
        s = self._h1("AetherLife — Une civilisation artificielle apprend à parler")
        s += "_Note publique du run "
        s += f"`{self.run_id}` — généré le {self.generated_at[:10]}_\n\n"
        s += (
            "Dans le projet AetherLife, des agents artificiels vivent dans un "
            "monde simulé : ils mangent, se reproduisent, héritent d'un cerveau "
            "neuronal et de quelques 'biais' comportementaux. **Personne ne leur "
            "dit comment vivre.** Tout émerge.\n\n"
        )
        n_alive = self._get("final_state", "n_alive")
        n_births = self._get("final_state", "n_births_total")
        n_lineages = self._get("criterion_3_selection", "n_lineages_final")
        s += "## Que s'est-il passé dans ce run ?\n\n"
        n_ticks = self._get("config", "n_ticks")
        if n_alive is not None and n_births is not None:
            s += (
                f"Sur **{n_ticks:,} ticks** de simulation, **{n_births} naissances** "
                f"ont eu lieu, **{n_alive} agents sont vivants** à la fin. "
            )
            if n_lineages:
                s += (
                    f"**{n_lineages} lignées familiales** coexistent encore — "
                    f"chacune partage le même cerveau neuronal hérité de son "
                    f"fondateur, avec quelques mutations.\n\n"
                )

        lang = self._get("language_metrics_v8b2") or {}
        if lang and lang.get("total_vocalize_count", 0) > 0:
            s += "## Ils ont commencé à parler\n\n"
            conc = lang.get("mean_token_lineage_concentration", 0)
            l2 = lang.get("mean_inter_lineage_distance", 0)
            total = lang.get("total_vocalize_count", 0)
            s += (
                f"Au cours de ce run, les agents ont émis **{total:,} signaux "
                f"vocalisés** (chacun coûtant de l'énergie). Plus intéressant : "
            )
            if conc >= 0.7:
                s += (
                    f"chaque type de signal est utilisé à **{conc:.0%} par "
                    f"une seule lignée**. Autrement dit, **chaque famille a "
                    f"développé son propre dialecte**.\n\n"
                )
            else:
                s += (
                    f"la concentration par lignée est de {conc:.0%} — ils "
                    f"parlent, mais sans dialecte clair encore.\n\n"
                )
            if l2 >= 1.0:
                s += (
                    f"La distance L2 entre les vocabulaires des familles est "
                    f"de **{l2:.2f}**. Ce nombre mesure à quel point les "
                    f"'mots' (vecteurs numériques) de deux familles divergent. "
                    f"Plus c'est élevé, plus elles 'parlent différemment'.\n\n"
                )

        s += "## Important : ce sont des hypothèses, pas des certitudes\n\n"
        s += (
            "Nous ne savons pas ce que ces signaux signifient pour les agents. "
            "Aucun token n'est étiqueté 'nourriture' ou 'danger'. Nous observons "
            "uniquement des **corrélations probabilistes** : un signal apparaît "
            "souvent quand la nourriture est proche ; un autre, quand un agent "
            "rentre vers son territoire. Le sens reste à vérifier.\n\n"
        )
        s += "## Ce qu'on a appris (et ce qu'il reste à valider)\n\n"
        for d in self.discoveries[:5]:
            s += f"- {d.headline}\n"
        if not self.discoveries:
            s += "_Pas de pattern significatif sur ce run._\n"
        s += "\n"
        s += "## La suite\n\n"
        s += (
            "- Reproduire ces observations sur plusieurs simulations indépendantes.\n"
            "- Mesurer si l'écoute d'un signal modifie réellement le comportement de "
            "l'auditeur dans les ticks suivants.\n"
            "- Voir si certains dialectes survivent, fusionnent ou disparaissent.\n\n"
            "AetherLife n'est pas une intelligence artificielle généraliste. "
            "C'est un **laboratoire d'évolution cognitive** où l'on peut "
            "observer, pour la première fois, ce qui pourrait ressembler à "
            "l'apparition spontanée d'une culture partagée.\n"
        )
        return s

    def render_discoveries(self) -> str:
        s = self._h1(f"Découvertes détectées — `{self.run_id}`")
        s += (
            "_Les 'découvertes' ci-dessous sont des **hypothèses prudentes** "
            "déduites de patterns observés dans les métriques. Aucune ne doit "
            "être interprétée comme une affirmation absolue. Chaque hypothèse "
            "liste des pistes de validation ultérieure._\n\n"
        )
        if not self.discoveries:
            s += "Aucun pattern significatif détecté.\n"
            return s
        for d in self.discoveries:
            s += f"## `{d.slug}` ({d.category.value})\n\n"
            s += f"**Confiance détectée** : {d.confidence:.2f} / 1.00\n\n"
            s += f"> {d.headline}\n\n"
            if d.evidence:
                s += "**Preuves métriques** :\n\n"
                for k, v in d.evidence.items():
                    s += f"- `{k}` = `{v}`\n"
                s += "\n"
            if d.validation:
                s += "**Pistes de validation** :\n\n"
                for v in d.validation:
                    s += f"- {v}\n"
                s += "\n"
        return s

    def render_lineages(self) -> str:
        s = self._h1(f"Vie des lignées — `{self.run_id}`")
        s += "_Suivi démographique et généalogique des familles d'agents._\n\n"

        s += self._h2("Distribution finale")
        top = self._get("final_state", "top_lineages") or []
        if top:
            s += "Top lignées vivantes en fin de run :\n\n"
            s += "| Root ID | Vivants | % |\n|---|---|---|\n"
            for lin in top:
                s += (
                    f"| {lin.get('root_id')} | {lin.get('alive')} | "
                    f"{lin.get('pct', 0):.1f} % |\n"
                )
            s += "\n"
        else:
            s += "_Aucune donnée de lignée vivante (extinction possible)._\n\n"

        aff = self._get("final_state", "affinity_distribution") or {}
        if aff:
            s += self._h2("Distribution par affinity (biome préféré)")
            s += "| Affinity | Vivants |\n|---|---|\n"
            for k, v in aff.items():
                names = {"0": "PLAIN", "1": "FOREST", "2": "DESERT", "3": "TUNDRA"}
                s += f"| {names.get(str(k), str(k))} | {v} |\n"
            s += "\n"

        lineages_curve = self._get("curves", "lineages") or []
        if lineages_curve:
            s += self._h2("Évolution du nombre de lignées")
            s += "| Tick | Lignées vivantes |\n|---|---|\n"
            for t, n in lineages_curve:
                s += f"| {t} | {n} |\n"
            s += "\n"

        s += self._h2("Lecture")
        n_init = self._get("criterion_3_selection", "n_lineages_initial")
        n_fin = self._get("criterion_3_selection", "n_lineages_final")
        if n_init is not None and n_fin is not None:
            s += (
                f"De {n_init} lignées initiales, "
                f"{n_fin} sont vivantes en fin de run. "
            )
            if n_init > 0:
                reduction = (n_init - n_fin) / n_init
                s += f"Réduction : {reduction:.0%}. "
            s += (
                "Le système de **seed bank** (cerveau archivé par affinity) "
                "permet la résurrection partielle de lignées éteintes après "
                "un délai, ce qui crée la 'queue longue' observée.\n"
            )
        return s

    def render_dialects(self) -> str:
        s = self._h1(f"Analyse linguistique — `{self.run_id}`")
        lang = self._get("language_metrics_v8b2") or {}
        if not lang:
            s += (
                "_Le langage n'était pas activé dans ce run (régime ≠ language). "
                "Pour activer : passer `--regime language` au bench._\n"
            )
            return s

        s += "_Émergence du langage par sélection naturelle pure : "
        s += "pas de reward direct, juste un coût énergétique par "
        s += "vocalisation. Les lignées qui utilisent des tokens 'utiles' "
        s += "survivent ; les autres s'éteignent._\n\n"

        s += self._h2("Métriques d'usage")
        s += self._kv_table([
            ("Total vocalize", lang.get("total_vocalize_count")),
            ("Vocalize / 1000 ticks", f"{lang.get('tokens_per_1000_ticks', 0):.0f}"),
            ("Energy cost cumulé", f"{lang.get('vocalize_energy_cost_total', 0):.1f}"),
        ])

        s += self._h2("Structure du vocabulaire")
        per_tok = lang.get("per_token_usage_top", {})
        if per_tok:
            s += "| Token | Usage total |\n|---|---|\n"
            for tok_id, count in sorted(per_tok.items(), key=lambda x: -int(x[1])):
                s += f"| `{tok_id}` | {count} |\n"
            s += "\n"

        s += self._h2("Émergence des dialectes (signal probabiliste)")
        s += self._kv_table([
            ("Entropy moyenne", f"{lang.get('mean_usage_entropy', 0):.3f}"),
            ("Entropy max possible", f"{lang.get('max_possible_entropy', 0):.3f}"),
            ("Entropy ratio", f"{lang.get('entropy_ratio', 0):.1%}"),
            ("Concentration par lignée (moy)",
             f"{lang.get('mean_token_lineage_concentration', 0):.1%}"),
            ("Distance L2 inter-vocabs",
             f"{lang.get('mean_inter_lineage_distance', 0):.2f}"),
        ])
        s += "\n"
        s += "**Lecture probabiliste** :\n\n"
        conc = lang.get("mean_token_lineage_concentration", 0)
        l2 = lang.get("mean_inter_lineage_distance", 0)
        entropy = lang.get("entropy_ratio", 0)
        if conc >= 0.7:
            s += (
                f"- Le pattern de concentration ({conc:.1%}) suggère que "
                "**chaque token est utilisé majoritairement par une seule lignée**. "
                "C'est une signature compatible avec l'émergence de dialectes.\n"
            )
        else:
            s += (
                f"- Concentration {conc:.1%} : pas de dialecte stable détectable.\n"
            )
        if l2 >= 1.0:
            s += (
                f"- Distance L2 ({l2:.2f}) entre vocabularies est élevée — "
                "les 'mots' (embeddings) divergent entre lignées.\n"
            )
        if 0.5 < entropy < 0.95:
            s += (
                f"- Entropy ratio {entropy:.1%} : vocabulaire structuré "
                "(ni monopole ni chaos uniforme).\n"
            )
        s += "\n"
        s += "**À valider** :\n\n"
        s += "- Concentration par lignée robuste sur ≥ 3 seeds différents.\n"
        s += "- Corrélation token → contexte (food visible, low energy, etc.).\n"
        s += "- Causalité : l'écoute d'un token modifie-t-elle le comportement ?\n"
        s += "- Stabilité long terme : les dialectes survivent-ils à 100k ticks ?\n"
        return s
