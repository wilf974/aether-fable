"""prereg — gestion d'un préenregistrement AetherLife (V2.5).

Trois sous-commandes :

    plan    : affiche le plan de lancement reproductible (condition × seed)
    audit   : agrège les runs collectés et confronte aux critères figés -> verdict
    lock    : verrouille un spec (renseigne locked_at) avant collecte

Le spec est un JSON produit/relu par aetherlife.analysis.prereg.PreregSpec.

Exemples :
    python scripts/prereg.py plan  docs/preregistrations/c2_N30.json
    python scripts/prereg.py lock  docs/preregistrations/c2_N30.json
    python scripts/prereg.py audit docs/preregistrations/c2_N30.json --runs results/c2_repli_N30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aetherlife.analysis.aggregate import collect_runs, get_path
from aetherlife.analysis.prereg import PreregSpec, audit


def _runs_by_condition(spec: PreregSpec, runs_root: Path) -> dict[str, list[dict]]:
    """Range les runs par condition via leur sous-dossier {label}/ dans le chemin.

    Convention launch_plan : results/<prereg_id>/<label>/seed<s>/.
    On classe chaque run selon le label de condition présent dans son out_dir,
    via le champ config.regime/seed si dispo, sinon par appartenance de dossier.
    """
    by: dict[str, list[dict]] = {c.label: [] for c in spec.conditions}
    for cond in spec.conditions:
        cond_dir = runs_root / cond.label
        if cond_dir.is_dir():
            by[cond.label] = collect_runs(cond_dir)
    return by


def cmd_plan(spec: PreregSpec, args) -> int:
    plan = spec.launch_plan(out_root=args.out_root)
    print(f"# Plan de lancement — {spec.prereg_id}  "
          f"({len(spec.conditions)} conditions × {len(spec.seeds)} seeds = {len(plan)} runs)")
    if spec.locked_at:
        print(f"# VERROUILLÉ : {spec.locked_at}")
    else:
        print("# NON VERROUILLÉ — exécuter `lock` avant collecte")
    for entry in plan:
        print(" ".join(entry["command"]))
    return 0


def cmd_lock(spec: PreregSpec, args) -> int:
    if spec.locked_at:
        print(f"Déjà verrouillé le {spec.locked_at} — aucune action.", file=sys.stderr)
        return 0
    locked = spec.locked()
    locked.save(args.spec)
    print(f"Verrouillé le {locked.locked_at} -> {args.spec}")
    return 0


def cmd_audit(spec: PreregSpec, args) -> int:
    runs_root = Path(args.runs)
    if not runs_root.is_dir():
        print(f"dossier de runs introuvable : {runs_root}", file=sys.stderr)
        return 1
    by_cond = _runs_by_condition(spec, runs_root)
    result = audit(spec, by_cond, confidence=args.confidence)

    print(f"=== AUDIT — {spec.prereg_id} ===")
    if spec.locked_at:
        print(f"spec verrouillé : {spec.locked_at}")
    else:
        print("⚠ spec NON verrouillé — audit exploratoire (résultat non confirmatoire)")
    print(f"hypothèse : {spec.hypothesis}\n")
    print(f"{'condition':<10} {'n':>3}  effectifs collectés")
    for c in spec.conditions:
        print(f"  {c.label:<8} {len(by_cond.get(c.label, [])):>3}")
    print(f"\n{'critère':<22} {'condition':<8} {'attendu':>10} {'observé':>10} {'n':>3}  ok")
    for cr in result.criteria:
        obs = "—" if cr.observed is None else f"{cr.observed:.4g}"
        print(f"  {cr.name:<20} {str(cr.condition_label):<8} {cr.expected:>10} "
              f"{obs:>10} {cr.n:>3}  {'✓' if cr.passed else '✗'}")
    print(f"\nVERDICT : {result.verdict.value}  ({result.n_pass}/{result.n_total} critères)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="affiche les commandes reproductibles")
    p_plan.add_argument("spec", type=Path)
    p_plan.add_argument("--out-root", default="results")

    p_lock = sub.add_parser("lock", help="verrouille le spec (locked_at)")
    p_lock.add_argument("spec", type=Path)

    p_audit = sub.add_parser("audit", help="agrège les runs et rend le verdict")
    p_audit.add_argument("spec", type=Path)
    p_audit.add_argument("--runs", required=True, help="dossier racine des runs collectés")
    p_audit.add_argument("--confidence", type=float, default=0.95)

    args = ap.parse_args(argv)
    if not args.spec.exists():
        print(f"spec introuvable : {args.spec}", file=sys.stderr)
        return 1
    spec = PreregSpec.load(args.spec)

    return {"plan": cmd_plan, "lock": cmd_lock, "audit": cmd_audit}[args.cmd](spec, args)


if __name__ == "__main__":
    sys.exit(main())
