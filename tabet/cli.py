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
    import pandas as pd

    from .prospective import log_due
    if args.now and not args.dry_run:
        raise SystemExit("--now is only allowed with --dry-run (a real record's cutoff is the run time)")
    now = pd.Timestamp(args.now, tz="UTC") if args.now else None
    from .prospective import LOG_HORIZONS
    horizons = ((args.horizon, args.window),) if args.horizon is not None or args.window is not None else LOG_HORIZONS
    if len(horizons) == 1 and None in horizons[0]:
        raise SystemExit("--horizon and --window go together")
    recs = log_due(now=now, horizons=horizons, dry_run=args.dry_run)
    if args.dry_run:
        role = {24.0: "primary", 26.0: "fallback"}
        print(f"{len(recs)} record(s) due:", [f"{r['fpl_fixture']} {r['home_id']} v {r['away_id']} "
                                              f"(kickoff {pd.Timestamp(r['kickoff_utc']):%Y-%m-%d %H:%M}, "
                                              f"{role.get(r['horizon_h'], r['horizon_h'])})" for r in recs])
    else:
        print(f"logged {len(recs)} fixture(s)")
        _archive_odds(now or pd.Timestamp.now(tz="UTC"))
    return 0


def _archive_odds(now, out_dir: Path | None = None, force: bool = False) -> dict:
    """Amendment A6: archive pre-match odds on every logging run. Benchmark only; runs after the
    forecasts are logged and can never stop or change them."""
    from . import odds_archive
    from .forecast import upcoming_fixtures
    try:
        fx = upcoming_fixtures(refresh=False)
    except Exception as e:
        print(f"odds archive: no FPL fixture list ({e}); rows are kept unmatched")
        fx = None
    try:
        r = odds_archive.archive(now, fx, out_dir=out_dir or odds_archive.ODDS_DIR, force=force)
    except Exception as e:
        print(f"odds archive failed ({e})")
        return {"status": "error", "error": str(e)}
    if r["status"] == "ok":
        print(f"odds archive: {r['n_rows']} rows ({r['n_e0']} E0, {r['n_e0_upcoming']} upcoming E0 fixtures); "
              f"{'new copy saved ' + r['raw_file'] if r['changed'] else 'unchanged, nothing saved'}; "
              f"football-data Last-Modified {r['source_last_modified'] or 'n/a'}")
        if r["unmatched"]:
            print(f"odds archive: {len(r['unmatched'])} E0 row(s) not matched to an FPL fixture (kept): {r['unmatched']}")
    else:
        print(f"odds archive: {r['status']} ({r.get('reason') or r.get('error')})")
    return r


def cmd_archive_odds(args: argparse.Namespace) -> int:
    r = _archive_odds(pd.Timestamp.now(tz="UTC"), out_dir=args.dir, force=True)
    return 0 if r["status"] == "ok" else 1


def cmd_score_forecasts(args: argparse.Namespace) -> int:
    from .prospective import (alpha_report, by_hours_before, by_record_role, decomposition, robustness, score,
                              select_records, summarise)
    s = score()
    if s.empty:
        print("nothing to score yet")
        return 0
    print(summarise(s, args.horizon).round(4).to_string(index=False))
    sel = select_records(s).drop_duplicates("fpl_fixture")
    print(f"scored fixtures: {len(sel)}; from fallback records (26-32 h): {(sel['horizon_h'] != 24.0).mean():.0%}")
    if "superseded" in s:
        print(f"superseded records (kickoff moved after logging): {s.loc[s.superseded, 'fpl_fixture'].nunique()}")
    d = decomposition(s, args.horizon)
    if len(d):
        print("\nXI-feature decomposition (secondary, amendment A2):")
        print(d.round(4).to_string(index=False))
    bh = by_hours_before(s, primary_horizon=args.horizon)
    if len(bh):
        print("\nBy hours before kickoff (amendment A4):")
        print(bh.round(4).to_string(index=False))
    rb = robustness(s, "pE", args.horizon)
    if rb:
        print("\npE - p0 robustness (amendment A3):", {k: v for k, v in rb.items() if k != "leave_one_gameweek_out"})
        print("  leave one gameweek out:", {g: round(v, 4) for g, v in rb["leave_one_gameweek_out"].items()})
    rr = by_record_role(s)
    if len(rr):
        print("\nBy record type and lineup-source age (amendment A5):")
        print(rr.round(4).to_string(index=False))
    bp = robustness(s, "pBP", args.horizon)
    if bp:
        n_rec = select_records(s).drop_duplicates("fpl_fixture").shape[0]
        print(f"\npBP - p0 (secondary, amendment A5; {n_rec - bp['n']} scored fixture(s) without pBP):",
              {k: v for k, v in bp.items() if k != "leave_one_gameweek_out"})
    print("\nalpha (amendment A5):", alpha_report(s))
    _print_benchmark(s)
    return 0


def _print_benchmark(s: pd.DataFrame) -> None:
    """Amendment A6: TaBet vs bookmaker odds on identical fixtures. Benchmark only: printed and saved,
    never used for alpha or any model choice."""
    from . import market_benchmark
    try:
        b = market_benchmark.benchmark(s)
    except Exception as e:
        print(f"\nmarket benchmark failed ({e})")
        return
    if b.empty:
        print("\nmarket benchmark (amendment A6): no scored fixtures")
        return
    p = market_benchmark.save(b)
    print(f"\nMarket benchmark (amendment A6, benchmark only; TaBet - market log-loss, negative = TaBet better; "
          f"p_alpha uses alpha = {b['alpha'].iloc[0]:.2f}, provisional until gameweek 19 is complete). Saved {p}:")
    show = [c for c in ("snapshot", "candidate", "n", "coverage", "log_loss_tabet", "log_loss_market",
                        "tabet_minus_market", "se_paired", "se_gameweek_clustered", "agree_rate", "tabet_only_right",
                        "market_only_right", "both_wrong", "n_later", "median_hours_after_cutoff") if c in b]
    print(b[show].round(4).to_string(index=False))


def cmd_heartbeat(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from .prospective import LOG, heartbeat, heartbeat_markdown, results_behind
    now = pd.Timestamp.now(tz="UTC").floor("min")
    hb = heartbeat(now)
    last = None
    if LOG.exists() and LOG.read_text().strip():
        r = json.loads(LOG.read_text().splitlines()[-1])
        last = f"FPL {r['fpl_fixture']} {r['home_id']} v {r['away_id']}, logged {r['logged_utc']} ({r['hours_before']:.1f} h ahead)"
    md = heartbeat_markdown(hb, last)
    extra = []
    try:  # amendment A6 monitoring; never breaks the heartbeat
        hb.update(results_behind(now))
        d = hb["results_detail"]
        if d:
            extra.append(f"- Results file: {d['local_rows']} matches locally, {d['remote_rows']} at football-data "
                         f"(last check {d['last_attempt_utc']}, ok={d['ok']})")
        if hb["results_behind"]:
            extra.append("- **Results file behind football-data for 2 days in a row**")
    except Exception as e:
        extra.append(f"- Results file check failed: {e}")
    try:
        from . import odds_archive
        from .forecast import LIVE_SEASON, upcoming_fixtures
        from .fpl import fetch_live_season
        allfx = pd.DataFrame(json.loads((fetch_live_season(LIVE_SEASON, refresh=False) / "fixtures.json").read_text()))
        allfx = allfx.rename(columns={"id": "fpl_fixture", "event": "gameweek"})[["fpl_fixture", "gameweek"]]
        hb.update(odds_archive.status(now, upcoming_fixtures(refresh=False), gameweek_fixtures=allfx))
        cov = hb["odds_coverage"]
        if cov:
            extra.append(f"- Odds archive: gameweek {cov['gameweek']} has a snapshot for {cov['with_snapshot']} of "
                         f"{cov['fixtures']} fixtures; last fetch {hb['odds_last_fetch_utc'] or 'never'}")
        if hb["odds_missing"]:
            extra.append(f"- **No odds snapshot yet for fixtures kicking off within 24 h: FPL {hb['odds_missing']}**")
    except Exception as e:
        extra.append(f"- Odds archive check failed: {e}")
    print(md + "\n".join(extra) + ("\n" if extra else ""))
    if args.json:
        print(json.dumps(hb))
    return 0


def cmd_site_data(args: argparse.Namespace) -> int:
    from .site_data import build, stale
    if args.if_stale is not None and not stale(args.if_stale):
        print("site data is fresh; not rebuilt")
        return 0
    d = build(preview=not args.no_preview)
    print(f"site data: {len(d['matches'])} matches written to site/data/matches.json")
    return 0


def cmd_check_log(args: argparse.Namespace) -> int:
    from .prospective import duplicate_keys
    dup = duplicate_keys()
    if dup:
        print(f"duplicate log keys: {dup}")
        return 1
    print("log OK: no fixture/kickoff appears twice")
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

    lg = sub.add_parser("log-forecasts", help="log primary (24-26 h) and fallback (26-32 h) records for due fixtures "
                                              "(amendment A4); safe to run often")
    lg.add_argument("--horizon", type=float, default=None,
                    help="with --window: log a single window instead of the protocol's two (testing only)")
    lg.add_argument("--window", "--every", dest="window", type=float, default=None,
                    help="window width in hours for --horizon. Not the run interval; --every is a deprecated alias")
    lg.add_argument("--dry-run", action="store_true", help="list the fixtures a run would log; fetches only the FPL fixture list and logs nothing")
    lg.add_argument("--now", help="with --dry-run only: pretend the run happens at this UTC time, e.g. 2026-11-20T05:00")
    hb = sub.add_parser("heartbeat", help="print logger status as Markdown (for the Actions job summary)")
    hb.add_argument("--json", action="store_true", help="also print the status as JSON")
    sub.add_parser("check-log", help="fail if any fixture/kickoff appears twice in the prospective log")
    sd = sub.add_parser("site-data", help="rebuild site/data/matches.json for the match explorer")
    sd.add_argument("--if-stale", type=float, default=None, metavar="HOURS",
                    help="only rebuild if older than HOURS or the forecast log has new records")
    sd.add_argument("--no-preview", action="store_true", help="skip the live preview forecasts (no network)")
    lg.set_defaults(func=cmd_log_forecasts)
    hb.set_defaults(func=cmd_heartbeat)
    sub.choices["check-log"].set_defaults(func=cmd_check_log)
    sd.set_defaults(func=cmd_site_data)

    sub.add_parser("check-sources", help="check each live data source is reachable and parses").set_defaults(
        func=cmd_check_sources)

    ao = sub.add_parser("archive-odds", help="fetch football-data's fixtures.csv once now and archive it (benchmark "
                                             "only; log-forecasts already does this hourly)")
    ao.add_argument("--dir", type=Path, default=None, help="archive directory (default results/prospective/odds)")
    ao.set_defaults(func=cmd_archive_odds)

    sc = sub.add_parser("score-forecasts", help="score logged forecasts against confirmed XIs and results")
    sc.add_argument("--horizon", type=float, default=24.0)
    sc.set_defaults(func=cmd_score_forecasts)

    args = p.parse_args(argv)
    return args.func(args)
