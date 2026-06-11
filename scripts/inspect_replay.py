"""inspect_replay — inspecteur tick-par-tick des events v8 (schema 2).

Lit un dossier de run (``events.jsonl`` + ``meta.json`` optionnel) et permet
d'inspecter l'état du monde sans GUI : résumé d'un tick, trajectoire d'un
agent, recherche d'événements (extinction, pic vocal), et métriques d'écologie
agrégées sur le run.

Usage :
    python scripts/inspect_replay.py RUN_DIR --summary
    python scripts/inspect_replay.py RUN_DIR --tick 15000
    python scripts/inspect_replay.py RUN_DIR --agent 7
    python scripts/inspect_replay.py RUN_DIR --find extinction
    python scripts/inspect_replay.py RUN_DIR --ecology

RUN_DIR peut aussi être directement un chemin vers events.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from aetherlife.metrics.ecology import EcologyTracker
from aetherlife.viz.v8_replay import iter_events, load_meta


def resolve_paths(run: Path) -> tuple[Path, Path | None]:
    """Retourne (events_path, meta_path|None) depuis un dossier ou un .jsonl."""
    if run.is_dir():
        events = run / "events.jsonl"
        meta = run / "meta.json"
        return events, (meta if meta.exists() else None)
    return run, (run.parent / "meta.json" if (run.parent / "meta.json").exists() else None)


def load_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        raise FileNotFoundError(f"events.jsonl introuvable : {events_path}")
    return list(iter_events(str(events_path)))


def is_snapshot(ev: dict) -> bool:
    """Distingue les events « état du monde » (schema 2) des events typés legacy."""
    return "agents" in ev and "type" not in ev


def summarize(events: list[dict], meta: dict | None) -> list[str]:
    snaps = [e for e in events if is_snapshot(e)]
    lines = ["=== RÉSUMÉ DU RUN ==="]
    if meta:
        lines.append(
            f"grille {meta.get('rows','?')}×{meta.get('cols','?')}  "
            f"regime={meta.get('regime','?')}  seed={meta.get('seed','?')}  "
            f"n_tokens={meta.get('n_tokens','?')}  schema=v{meta.get('schema_version','?')}"
        )
    if not snaps:
        lines.append("(aucun snapshot d'état — fichier d'events legacy ?)")
        return lines
    ticks = [e["t"] for e in snaps]
    alive = [e.get("n_alive", len(e["agents"])) for e in snaps]
    lines.append(f"snapshots={len(snaps)}  ticks {min(ticks)}..{max(ticks)}")
    lines.append(f"population : début={alive[0]}  fin={alive[-1]}  "
                 f"min={min(alive)}  max={max(alive)}")
    if any("n_lin" in e for e in snaps):
        lins = [e.get("n_lin", 0) for e in snaps]
        lines.append(f"lignées : début={lins[0]}  fin={lins[-1]}  max={max(lins)}")
    n_vocal = sum(len(e.get("vocal", {})) for e in snaps)
    lines.append(f"vocalisations totales (snapshots) : {n_vocal}")
    return lines


def show_tick(events: list[dict], t: int) -> list[str]:
    snaps = [e for e in events if is_snapshot(e)]
    match = min(snaps, key=lambda e: abs(e["t"] - t), default=None)
    if match is None:
        return [f"aucun snapshot proche de t={t}"]
    lines = [f"=== TICK {match['t']} (demandé {t}) ==="]
    lines.append(f"n_alive={match.get('n_alive','?')}  "
                 f"n_lin={match.get('n_lin','?')}  season={match.get('season','?')}")
    agents = match.get("agents", [])
    lines.append(f"agents ({len(agents)}) :")
    for a in agents[:40]:
        lines.append(f"  id={a['id']:<4} lin={a['lin']:<4} pos=({a['r']:>3},{a['c']:>3}) "
                     f"e={a['e']:<5} er={a.get('er','?'):<5} age={a['age']:<5} aff={a.get('aff','?')}")
    if len(agents) > 40:
        lines.append(f"  … +{len(agents) - 40} autres")
    vocal = match.get("vocal", {})
    if vocal:
        lines.append(f"vocal : {vocal}")
    spots = match.get("spots", [])
    if spots:
        lines.append(f"spots actifs : {len(spots)}  "
                     f"(max occupants={max((s.get('n',0) for s in spots), default=0)})")
    return lines


def trace_agent(events: list[dict], agent_id: int) -> list[str]:
    lines = [f"=== TRAJECTOIRE AGENT {agent_id} ==="]
    seen = False
    for e in events:
        if not is_snapshot(e):
            continue
        for a in e.get("agents", []):
            if a["id"] == agent_id:
                seen = True
                vocal = e.get("vocal", {}).get(str(agent_id))
                vtxt = f"  vocal={vocal}" if vocal is not None else ""
                lines.append(f"  t={e['t']:<7} pos=({a['r']:>3},{a['c']:>3}) "
                             f"e={a['e']:<5} age={a['age']:<5} lin={a['lin']} aff={a.get('aff','?')}{vtxt}")
    if not seen:
        lines.append(f"  agent {agent_id} absent des snapshots")
    return lines


def find_events(events: list[dict], query: str) -> list[str]:
    snaps = [e for e in events if is_snapshot(e)]
    q = query.lower()
    lines = [f"=== RECHERCHE : {query} ==="]
    if q in ("extinction", "collapse", "effondrement"):
        prev = None
        for e in snaps:
            n = e.get("n_alive", len(e.get("agents", [])))
            if n == 0:
                lines.append(f"  EXTINCTION à t={e['t']}")
                break
            if prev is not None and prev > 0 and n <= prev * 0.5:
                lines.append(f"  chute brutale t={e['t']} : {prev} → {n} (-{100*(prev-n)//prev}%)")
            prev = n
    elif q in ("vocal", "vocal_peak", "langage"):
        ranked = sorted(snaps, key=lambda e: len(e.get("vocal", {})), reverse=True)
        for e in ranked[:5]:
            lines.append(f"  t={e['t']} : {len(e.get('vocal', {}))} vocalisations")
    else:
        lines.append(f"  requête inconnue. Essayez : extinction, vocal")
    if len(lines) == 1:
        lines.append("  (rien trouvé)")
    return lines


def ecology_block(events: list[dict], meta: dict | None) -> list[str]:
    rows = (meta or {}).get("rows", 64)
    cols = (meta or {}).get("cols", 64)
    naff = (meta or {}).get("n_initial_affinities", 4)
    tr = EcologyTracker(rows=rows, cols=cols, n_affinities=max(naff, 1))
    for e in events:
        if is_snapshot(e):
            tr.observe_event(e)
    block = tr.finalize()
    lines = ["=== ÉCOLOGIE (agrégé sur le run) ==="]
    lines.append(f"observations         : {block['n_observations']}")
    lines.append(f"effectifs / affinité : {block['affinity_counts']}")
    lines.append(f"Shannon H'           : {block['shannon_diversity']:.4f}  "
                 f"(évenness {block['shannon_evenness']:.3f})")
    lines.append(f"dominance Simpson λ  : {block['simpson_dominance']:.4f}")
    lines.append(f"recouvrement niche   : moyen={block['mean_niche_overlap']:.4f}  "
                 f"paires={block['niche_overlap']}")
    bif = block["alive_bifurcation"]
    if bif:
        verdict = "OUI" if bif["changed"] else "non"
        lines.append(f"bifurcation alive(t) : {verdict}  "
                     f"(score={bif['score']}, split#{bif['index']}, "
                     f"{bif['mean_before']}→{bif['mean_after']})")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="dossier de run ou chemin events.jsonl")
    ap.add_argument("--summary", action="store_true", help="résumé global (défaut)")
    ap.add_argument("--tick", type=int, help="état du monde au tick le plus proche")
    ap.add_argument("--agent", type=int, help="trajectoire d'un agent")
    ap.add_argument("--find", type=str, help="recherche : extinction | vocal")
    ap.add_argument("--ecology", action="store_true", help="métriques d'écologie agrégées")
    args = ap.parse_args(argv)

    events_path, meta_path = resolve_paths(args.run)
    try:
        events = load_events(events_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    meta = load_meta(str(meta_path)) if meta_path else None

    out: list[str] = []
    did_specific = False
    if args.tick is not None:
        out += show_tick(events, args.tick); did_specific = True
    if args.agent is not None:
        out += trace_agent(events, args.agent); did_specific = True
    if args.find:
        out += find_events(events, args.find); did_specific = True
    if args.ecology:
        out += ecology_block(events, meta); did_specific = True
    if args.summary or not did_specific:
        out = summarize(events, meta) + (["", *out] if out else [])

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
