"""Command-line entry point: python -m tabet {validate,build,history}."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from .ingest import IngestError, load_all
from .store import MatchStore, parse_cutoff
from .teams import UnknownTeamError
from .validate import has_errors, validate


def _load_and_validate(res: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    try:
        matches, odds = load_all(res)
    except IngestError as e:
        print(f"INGEST FAILED\n{e}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame(), False
    issues = validate(matches, odds)
    seasons = matches.groupby("season")["status"].value_counts().unstack(fill_value=0)
    print(f"Loaded {len(matches)} matches from {matches['source_file'].nunique()} file(s)")
    print(seasons.to_string())
    print()
    for i in issues:
        print(i)
    ok = not has_errors(issues)
    n_err = sum(i.level == "error" for i in issues)
    n_warn = len(issues) - n_err
    print(f"\n{'PASS' if ok else 'FAIL'}: {n_err} error(s), {n_warn} warning(s)")
    return matches, odds, ok


def cmd_validate(args: argparse.Namespace) -> int:
    _, _, ok = _load_and_validate(args.res)
    return 0 if ok else 1


def cmd_build(args: argparse.Namespace) -> int:
    matches, odds, ok = _load_and_validate(args.res)
    if not ok:
        print("Not writing outputs: fix the errors above first.", file=sys.stderr)
        return 1
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out / "matches.csv", index=False)
    odds.to_csv(out / "odds.csv", index=False)
    manifest = {
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_matches": len(matches),
        "sources": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path(args.res).glob("*.csv"))
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out / 'matches.csv'}, {out / 'odds.csv'}, {out / 'manifest.json'}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    try:
        cutoff = parse_cutoff(args.cutoff)
        hist = MatchStore.from_dir(args.res).get_history(args.team, cutoff, last=args.last)
    except (ValueError, UnknownTeamError) as e:
        print(e, file=sys.stderr)
        return 2
    if hist.empty:
        print(f"No completed matches for {args.team} before {cutoff}")
        return 0
    show = hist.assign(kickoff_utc=hist["kickoff_utc"].dt.strftime("%Y-%m-%d %H:%M"))
    cols = ["kickoff_utc", "season", "venue", "opponent_id", "goals_for", "goals_against",
            "outcome", "shots_for", "shots_against", "sot_for", "sot_against"]
    print(f"{hist['team_id'].iat[0]} — results known before {cutoff:%Y-%m-%d %H:%M}Z")
    print(show[cols].to_string(index=False))
    pts = hist["points"]
    print(f"\n{len(hist)} match(es): {pts.sum()} pts, {pts.mean():.2f} per game, "
          f"GF {hist['goals_for'].sum()} GA {hist['goals_against'].sum()}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from .forecast import forecast, save
    fc = forecast(gameweeks=args.gameweeks, refresh=True, lineups=not args.no_lineups)
    print(f"saved {save(fc)}")
    return 0


def cmd_log_forecasts(args: argparse.Namespace) -> int:
    from .prospective import log_due
    recs = log_due(horizon_h=args.horizon, every_h=args.every, dry_run=args.dry_run)
    if args.dry_run:
        print(f"{len(recs)} fixture(s) due:", [f"{r['home_id']} v {r['away_id']}" for r in recs])
    else:
        print(f"logged {len(recs)} fixture(s)")
    return 0


def cmd_score_forecasts(args: argparse.Namespace) -> int:
    from .prospective import decomposition, robustness, score, summarise
    s = score()
    if s.empty:
        print("nothing to score yet")
        return 0
    print(summarise(s, args.horizon).round(4).to_string(index=False))
    if "superseded" in s:
        print(f"superseded records (kickoff moved after logging): {s.loc[s.superseded, 'fpl_fixture'].nunique()}")
    d = decomposition(s, args.horizon)
    if len(d):
        print("\nXI-feature decomposition (secondary, amendment A2):")
        print(d.round(4).to_string(index=False))
    rb = robustness(s, "pE", args.horizon)
    if rb:
        print("\npE - p0 robustness (amendment A3):", {k: v for k, v in rb.items() if k != "leave_one_gameweek_out"})
        print("  leave one gameweek out:", {g: round(v, 4) for g, v in rb["leave_one_gameweek_out"].items()})
    return 0


def cmd_check_sources(args: argparse.Namespace) -> int:
    """Fetch each live source once and report whether it parsed (no logging)."""
    import json as _json
    from .fpl import fetch_live_season
    from .lineup_sources import fetch, fpl_players_now, parse_ffs, parse_rotowire
    from .prospective import RESULTS_URL
    import urllib.request
    from .fpl import _ssl_context
    stamp = "check_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%MZ")
    ok = True
    try:
        boot = _json.loads((fetch_live_season("2026-27", refresh=True) / "bootstrap.json").read_text())
        fp = fpl_players_now(boot)
        print(f"FPL API: ok, {len(fp)} players")
    except Exception as e:
        print(f"FPL API: FAILED {e}")
        return 1
    for src, parse in (("ffs", parse_ffs), ("rotowire", lambda pg: parse_rotowire(pg, fp))):
        try:
            df = parse(fetch(src, stamp))
            print(f"{src}: ok, {len(df)} players for {df['team_id'].nunique() if len(df) else 0} teams")
            ok &= len(df) > 0
        except Exception as e:
            print(f"{src}: FAILED {e}")
            ok = False
    try:
        req = urllib.request.Request(RESULTS_URL, headers={"User-Agent": "tabet-research/0.1"})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            print(f"football-data: ok, {len(r.read().splitlines()) - 1} rows")
    except Exception as e:
        print(f"football-data: FAILED {e}")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tabet", description="TaBet data pipeline")
    p.add_argument("--res", type=Path, default=Path("res"), help="directory of season CSVs")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="load and validate source data").set_defaults(func=cmd_validate)

    b = sub.add_parser("build", help="validate, then write normalised tables")
    b.add_argument("--out", type=Path, default=Path("data/processed"))
    b.set_defaults(func=cmd_build)

    h = sub.add_parser("history", help="a team's results known before a cutoff")
    h.add_argument("team")
    h.add_argument("--cutoff", required=True, help="ISO time with timezone, e.g. 2026-01-01T12:00Z")
    h.add_argument("--last", type=int, default=None)
    h.set_defaults(func=cmd_history)

    f = sub.add_parser("forecast", help="forecast upcoming fixtures and save the result")
    f.add_argument("--gameweeks", type=int, default=4)
    f.add_argument("--no-lineups", action="store_true")
    f.set_defaults(func=cmd_forecast)

    lg = sub.add_parser("log-forecasts", help="log forecasts for fixtures due at the fixed horizon (run hourly)")
    lg.add_argument("--horizon", type=float, default=24.0, help="hours before kickoff")
    lg.add_argument("--every", type=float, default=2.0, help="hours between scheduled runs")
    lg.add_argument("--dry-run", action="store_true")
    lg.set_defaults(func=cmd_log_forecasts)

    sub.add_parser("check-sources", help="check each live data source is reachable and parses").set_defaults(
        func=cmd_check_sources)

    sc = sub.add_parser("score-forecasts", help="score logged forecasts against confirmed XIs and results")
    sc.add_argument("--horizon", type=float, default=24.0)
    sc.set_defaults(func=cmd_score_forecasts)

    args = p.parse_args(argv)
    return args.func(args)
