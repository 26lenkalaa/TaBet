"""Data for the site (pitch, match explorer), written to site/data/ at the repository root:
  index.json                built time, log record count, seasons and the per-season file names
  matches-<season>.json     every match of that season with its forecasts and market odds
  lineups-<season>.json     starting XIs (actual and predicted), loaded only by the match page
These files are canonical; there is no combined matches.json.

Every match from 2022-23 onward gets the forecasts that exist for it, each labelled by kind:
  backtest  frozen walk-forward forecasts (results/xg_strength/ledger.csv): V0, V1x, V2x. Made with
            only information available before each match, but computed afterwards.
  logged    the prospective record made 24-32 h before kickoff (results/prospective/log.jsonl; primary
            record if it exists, else the fallback), with p0 (lineup-free) and pE (predicted lineups).
  preview   for upcoming fixtures without a logged record yet: the model run when this file was built,
            without lineups and, where Fantasy Football Scout and RotoWire have published predicted XIs
            (usually the coming gameweek), with them. Not a record; it changes as results and team
            news come in. The logged record is still made 24-32 h before kickoff (the protocol).
Market probabilities come from football-data average odds with the margin removed proportionally:
pre-closing (collected Friday/Tuesday afternoon) and closing. For upcoming fixtures, football-data's
fixtures.csv gives pre-closing odds once published. For a logged forecast, "Market at our forecast time" is
the bookmaker average archived at that record's cutoff (amendments A6, A7): The Odds API snapshot from the
same run, else a football-data snapshot taken by then. Odds are a benchmark only, never a model input:
this module reads the odds archive for display and nothing here feeds a forecast.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .fpl import _ssl_context, fetch_live_season
from .ingest import implied_probabilities, load_all
from .teams import TEAMS, try_resolve_team

ROOT = Path(__file__).resolve().parent.parent
SITE_DATA = ROOT.parent / "site" / "data"   # the shared site at the repository root
INDEX = SITE_DATA / "index.json"   # seasons, build time and per-season file names; the other files sit beside it
LIVE_SEASON = "2026-27"
LEDGER = ROOT / "results" / "xg_strength" / "ledger.csv"
LOG = ROOT / "results" / "prospective" / "log.jsonl"
BACKTEST = {"V0 Dixon-Coles": ("V0", "Results only (V0)"),
            "V1x adjusted xG strength (replaces raw)": ("V1x", "Chances created (V1x)"),
            "V2x XI strength with adjusted xG": ("V2x", "With confirmed lineups (V2x)")}


def _p(row) -> list[float]:
    return [round(float(row["p_home"]), 4), round(float(row["p_draw"]), 4), round(float(row["p_away"]), 4)]


def _odds_upcoming() -> dict[tuple[str, str], list[float]]:
    """Pre-closing odds for upcoming PL fixtures from football-data's fixtures.csv (if reachable)."""
    try:
        req = urllib.request.Request("https://www.football-data.co.uk/fixtures.csv", headers={"User-Agent": "tabet-research/0.1"})
        raw = urllib.request.urlopen(req, timeout=60, context=_ssl_context()).read()
    except Exception:
        return {}
    f = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    f = f[f["Div"] == "E0"]
    out = {}
    for r in f.itertuples():
        h, a = try_resolve_team(r.HomeTeam), try_resolve_team(r.AwayTeam)
        if h and a and pd.notna(r.AvgH):
            q = 1 / np.array([r.AvgH, r.AvgD, r.AvgA], float)
            out[(h, a)] = [round(float(x), 4) for x in q / q.sum()]
    return out


def _logged() -> dict[int, dict]:
    """Per FPL fixture, the record scoring would use: primary (24 h) if present, else fallback."""
    if not LOG.exists():
        return {}
    best: dict[int, dict] = {}
    for r in map(json.loads, LOG.read_text().splitlines()):
        cur = best.get(r["fpl_fixture"])
        if cur is None or r["horizon_h"] < cur["horizon_h"]:
            best[r["fpl_fixture"]] = r
    return best


# Which price the site shows (amendment A8): Pinnacle with its margin removed by Shin's method; if
# Pinnacle is missing from a snapshot, the median over bookmakers (exchanges excluded, A9); then the
# bookmaker average (older snapshots, football-data). Display only: nothing here feeds a forecast.
_PREFERENCE = (("pinnacle_shin", "Pinnacle, margin removed (Shin's method)"),
               ("median", "median of the bookmakers, margins removed"),
               ("Avg", "bookmaker average{n}, margin removed"))


def _snapshot_price(rows: pd.DataFrame) -> tuple[list[float], str] | None:
    """The preferred price among one snapshot's rows for one fixture, as probabilities and a note."""
    from .market_benchmark import _implied
    for book, text in _PREFERENCE:
        x = rows[rows["book"] == book]
        if len(x):
            x = x.iloc[0]
            n = f" of {int(x['n_books'])} bookmakers" if "n_books" in x and pd.notna(x.get("n_books")) else ""
            p = _implied(x[["odds_h", "odds_d", "odds_a"]].to_numpy(float)[None, :])[0]
            return [round(float(v), 4) for v in p], text.format(n=n)
    return None


def _matched_market(r: dict, odds_log: pd.DataFrame) -> dict | None:
    """The market at a logged record's cutoff, or None if no snapshot was taken by then."""
    from .market_benchmark import select_snapshots
    from .odds_archive import API_SOURCE
    if odds_log is None or odds_log.empty:
        return None
    rec = pd.DataFrame([{k: r[k] for k in ("fpl_fixture", "logged_utc", "kickoff_utc")}])
    for source, label in ((API_SOURCE, "The Odds API"), ("football-data", "football-data")):
        s = select_snapshots(rec, odds_log, source=source, with_last=False)
        if len(s) and not bool(s["later"].iloc[0]) and abs(float(s["hours_after_cutoff"].iloc[0])) <= 1.0:
            x = s.iloc[0]
            snap = odds_log[(odds_log["source"] == source) & (odds_log["retrieved_utc"] == x["retrieved_utc"])
                            & (pd.to_numeric(odds_log["fpl_fixture"], errors="coerce") == r["fpl_fixture"])]
            price = _snapshot_price(snap)
            if price is None:
                continue
            p, what = price
            hrs = (pd.Timestamp(r["kickoff_utc"]) - x["retrieved_utc"]).total_seconds() / 3600
            note = f"{what}; {label}, retrieved {x['retrieved_utc']:%d %b %H:%M} UTC, {hrs:.1f} h before kickoff"
            return {"key": "matched", "label": "Market at our forecast time", "note": note, "p": p}
    return None


def _latest_market(fixture: int, odds_log: pd.DataFrame) -> dict | None:
    """The newest The Odds API snapshot for an upcoming fixture (refreshed at least daily), priced by the
    same preference as the matched market: Pinnacle (Shin), else the median, else the average."""
    from .odds_archive import API_SOURCE
    if odds_log is None or odds_log.empty or "source" not in odds_log:
        return None
    o = odds_log[(odds_log["source"] == API_SOURCE) & (pd.to_numeric(odds_log["fpl_fixture"], errors="coerce") == fixture)]
    if o.empty:
        return None
    last = o["retrieved_utc"].max()
    price = _snapshot_price(o[o["retrieved_utc"] == last])
    if price is None:
        return None
    p, what = price
    return {"key": "latest", "label": "Latest market odds",
            "note": f"{what}; The Odds API, retrieved {last:%d %b %H:%M} UTC", "p": p}


def _odds_log() -> pd.DataFrame:
    try:
        from .odds_archive import load
        return load()
    except Exception as ex:  # the explorer must still build without the archive
        print(f"odds archive not read ({ex})")
        return pd.DataFrame()


POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
LINEUP_SOURCES = (("ffs", "Fantasy Football Scout"), ("rotowire", "RotoWire"))


def _xis(codes: dict, who: dict) -> dict:
    """Predicted XIs for the match page: per source and side, [name, position] for each player (by FPL
    player code, positions from FPL). Only the two scraped sources; display only."""
    out = {}
    for src, _ in LINEUP_SOURCES:
        c = codes.get(src) or {}
        if c.get("home") and c.get("away"):
            out[src] = {side: [[int(k), *who.get(int(k), (str(k), "MID"))] for k in c[side]] for side in ("home", "away")}
    return out


_MOJIBAKE = ("Ã", "Ä", "Å", "Â", "Ð", "â€")


def _fix_text(s: str) -> str:
    """Repair names that were UTF-8 but decoded as Latin-1/cp1252 somewhere upstream (the FPL archive
    has 'DoucourÃ©' for 'Doucouré', 'BegoviÄ\\x87' for 'Begović'). Display only: names are never
    used to join anything here. A string that does not round-trip cleanly is returned unchanged."""
    s = str(s)
    if not any(m in s for m in _MOJIBAKE):
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def _short(name: str) -> str:
    """'Bukayo Saka' -> 'B. Saka' for players no longer in the current FPL game."""
    name = _fix_text(name)
    t = name.split()
    return name if len(t) < 2 else f"{t[0][0]}. {t[-1]}"


def _actual_xis(app: pd.DataFrame, who: dict) -> dict:
    """Confirmed starting XIs from FPL appearances: {match_id: {"home": [[code, name, pos], ...], "away": ...,
    "inferred": bool}} (older seasons infer starters from minutes)."""
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    st = app[app["started"] == 1]
    out: dict = {}
    for (mid, venue), g in st.groupby(["match_id", "venue"], sort=False):
        side = "home" if venue == "H" else "away"
        g = g.assign(o=g["position"].map(order).fillna(2)).sort_values("o", kind="stable")
        out.setdefault(mid, {})[side] = [[int(c), who.get(int(c), (_short(n),))[0], p or "MID"]
                                         for c, n, p in zip(g["player_code"], g["player_name"], g["position"])]
        if bool(g["started_inferred"].any()):
            out[mid]["inferred"] = True
    return {k: v for k, v in out.items() if len(v.get("home", [])) == 11 and len(v.get("away", [])) == 11}


FORECASTS_DIR = ROOT / "results" / "forecasts"


def _saved_forecasts() -> dict[int, list[tuple[pd.Timestamp, dict]]]:
    """Full forecast runs saved by `python -m tabet forecast` (results/forecasts/*.json), by FPL fixture:
    [(cutoff, fixture forecast)], oldest first. Used only where no logged record exists."""
    out: dict = {}
    for f in sorted(FORECASTS_DIR.glob("forecast_*.json")):
        try:
            d = json.loads(f.read_text())
            cut = pd.Timestamp(d["cutoff_utc"])
        except Exception:
            continue
        for x in d.get("fixtures", []):
            out.setdefault(int(x["fpl_fixture"]), []).append((cut, x))
    return {k: sorted(v, key=lambda t: t[0]) for k, v in out.items()}


def _saved_entries(saved: dict, fixture: int | None, kickoff: pd.Timestamp) -> list[dict]:
    """The latest saved forecast made before kickoff, labelled for what it is: a real pre-match forecast
    that is not a protocol record."""
    before = [(c, x) for c, x in saved.get(fixture, []) if c < kickoff]
    if fixture is None or not before:
        return []
    cut, x = before[-1]
    note = f"made {cut:%d %b %H:%M} UTC, {(kickoff - cut).total_seconds() / 3600:.0f} h before kickoff (a saved forecast run, not a logged record)"
    out = []
    w = x.get("with_lineups") or {}
    if "source_average" in w:
        out.append({"key": "saved_lineups", "label": "TaBet, predicted lineups (pre-match)", "kind": "saved", "note": note,
                    "p": _p(w["source_average"])})
    out.append({"key": "saved", "label": "TaBet, no lineups (pre-match)", "kind": "saved", "note": note, "p": _p(x)})
    return out


def _logged_forecasts(r: dict) -> list[dict]:
    role = "primary" if r["horizon_h"] == 24.0 else "fallback"
    note = f"logged {r['hours_before']:.1f} h before kickoff ({role} record)"
    return [{"key": key, "label": label, "kind": "logged", "note": note, "p": _p(r["candidates"][key])}
            for key, label in (("p0", "TaBet, no lineups (p0)"), ("pE", "TaBet, predicted lineups (pE)"))
            if key in r["candidates"]]


def build(preview: bool = True, out: Path = INDEX) -> dict:
    """Build the site data. `out` is the index.json path; the per-season files are written beside it.
    Returns the full data dict (built_utc, n_log_records, seasons, matches)."""
    matches, odds = load_all(ROOT / "res")
    pre = implied_probabilities(odds, "Avg").set_index("match_id")
    close = implied_probabilities(odds, "AvgC").set_index("match_id")
    led = pd.read_csv(LEDGER)
    led = {v: led[led["variant"] == v].drop_duplicates("match_id").set_index("match_id") for v in BACKTEST}
    logged = _logged()
    odds_log = _odds_log()
    saved = _saved_forecasts()

    d = fetch_live_season(LIVE_SEASON, refresh=preview)
    boot = json.loads((d / "bootstrap.json").read_text())
    tid = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    fpl = pd.DataFrame(json.loads((d / "fixtures.json").read_text()))
    fpl["home_id"], fpl["away_id"] = fpl["team_h"].map(tid), fpl["team_a"].map(tid)
    gw_of = {(h, a): int(e) for h, a, e in zip(fpl["home_id"], fpl["away_id"], fpl["event"]) if pd.notna(e)}
    fpl_id = {(h, a): int(i) for h, a, i in zip(fpl["home_id"], fpl["away_id"], fpl["id"])}
    who = {int(e["code"]): (e["web_name"], POS.get(e["element_type"], "MID")) for e in boot["elements"]}
    pred: dict = {}

    entries = []
    done = matches[(matches["status"] == "completed") & (matches["season"] >= "2022-23")]
    seen_live = set()
    for m in done.itertuples():
        fc = [{"key": key, "label": label, "kind": "backtest", "p": _p(led[v].loc[m.match_id])}
              for v, (key, label) in BACKTEST.items() if m.match_id in led[v].index]
        mk = []
        if m.match_id in pre.index and pre.loc[m.match_id].notna().all():
            mk.append({"key": "pre", "label": "Earlier odds", "p": _p(pre.loc[m.match_id])})
        if m.match_id in close.index and close.loc[m.match_id].notna().all():
            mk.append({"key": "close", "label": "Closing odds", "p": _p(close.loc[m.match_id])})
        e = {"id": m.match_id, "season": m.season, "kickoff": m.kickoff_utc.isoformat(), "home": TEAMS.get(m.home_id, m.home_team),
             "away": TEAMS.get(m.away_id, m.away_team), "score": f"{m.home_goals}-{m.away_goals}", "result": m.result,
             "status": "played", "forecasts": fc, "market": mk}
        if m.season == LIVE_SEASON:
            e["gameweek"] = gw_of.get((m.home_id, m.away_id))
            seen_live.add((m.home_id, m.away_id))
            # a logged record stays on the site after the match reaches the results file
            r = logged.get(fpl_id.get((m.home_id, m.away_id), -1))
            if r:
                e["forecasts"] = _logged_forecasts(r) + e["forecasts"]
                xi = _xis(r.get("lineup_codes") or {}, who)
                if xi:
                    pred[e["id"]] = {"pred": xi, "note": f"predicted XIs saved with the logged forecast, {r['hours_before']:.0f} h before kickoff"}
                mm = _matched_market(r, odds_log)
                if mm:
                    e["market"].insert(0, mm)
            elif not e["forecasts"]:
                e["forecasts"] = _saved_entries(saved, fpl_id.get((m.home_id, m.away_id)), m.kickoff_utc)
                e["missing"] = "The logger missed this match's window, so there is no logged record." + (
                    " The forecast shown was saved before kickoff by a separate forecast run." if e["forecasts"] else "")
        entries.append(e)

    # current season: fixtures not (yet) in the football-data results file
    up_odds = _odds_upcoming()
    new = np.array([(h, a) not in seen_live for h, a in zip(fpl["home_id"], fpl["away_id"])], dtype=bool)
    pending = fpl[new & fpl["kickoff_time"].notna().to_numpy()]
    previews = {}
    if preview and len(pending[~pending["finished_provisional"].astype(bool)]):
        from .forecast import forecast
        try:
            fcst = forecast(cutoff=pd.Timestamp.now(tz="UTC").floor("min"), gameweeks=2, refresh=True, lineups=True)
            previews = {f["fpl_fixture"]: f for f in fcst["fixtures"]}
            preview_time = fcst["cutoff_utc"]
        except Exception as ex:  # the explorer must still build if a live fetch fails
            print(f"preview forecasts skipped ({ex})")
    for f in pending.sort_values("kickoff_time").itertuples():
        played = bool(f.finished_provisional)
        started = pd.Timestamp(f.kickoff_time) <= pd.Timestamp.now(tz="UTC")
        e = {"id": f"fpl-{f.id}", "season": LIVE_SEASON, "gameweek": int(f.event) if pd.notna(f.event) else None,
             "kickoff": pd.Timestamp(f.kickoff_time).isoformat(), "home": TEAMS.get(f.home_id, f.home_id),
             "away": TEAMS.get(f.away_id, f.away_id),
             "status": "played" if played else ("awaiting" if started else "upcoming"),
             "forecasts": [], "market": []}
        if played:
            h, a = int(f.team_h_score), int(f.team_a_score)
            e.update(score=f"{h}-{a}", result="H" if h > a else ("A" if a > h else "D"))
        r = logged.get(int(f.id))
        if r:
            e["forecasts"] += _logged_forecasts(r)
            xi = _xis(r.get("lineup_codes") or {}, who)
            if xi:
                pred[e["id"]] = {"pred": xi, "note": f"predicted XIs saved with the logged forecast, {r['hours_before']:.0f} h before kickoff"}
            mm = _matched_market(r, odds_log)
            if mm:
                e["market"].append(mm)
        elif int(f.id) in previews and not played:
            pv = previews[int(f.id)]
            built = f"built {preview_time[:16].replace('T', ' ')} UTC"
            w = pv.get("with_lineups") or {}
            if "source_average" in w:
                src = " and ".join({"ffs": "Fantasy Football Scout", "rotowire": "RotoWire"}[s] for s in w.get("source_average_of", []))
                e["forecasts"].append({"key": "preview_lineups", "label": "TaBet, predicted lineups (preview)", "kind": "preview",
                                       "note": f"{built} with {src}'s predicted XIs as published then; the logged forecast comes 24-32 h before kickoff",
                                       "p": _p(w["source_average"])})
            xi = _xis(w.get("lineup_codes") or {}, who)
            if xi:
                pred[e["id"]] = {"pred": xi, "note": f"predicted XIs as published when this page was {built}"}
            e["forecasts"].append({"key": "preview", "label": "TaBet, no lineups (preview)", "kind": "preview",
                                   "note": f"{built}; the logged forecast comes 24-32 h before kickoff",
                                   "p": _p(pv)})
        elif played or started:
            e["forecasts"] = _saved_entries(saved, int(f.id), pd.Timestamp(f.kickoff_time))
            e["missing"] = "The logger missed this match's window, so there is no logged record." + (
                " The forecast shown was saved before kickoff by a separate forecast run." if e["forecasts"] else "")
        if not played and not started:
            lm = _latest_market(int(f.id), odds_log)
            if lm:
                e["market"].append(lm)
        if (f.home_id, f.away_id) in up_odds:
            e["market"].append({"key": "pre", "label": "Earlier odds", "p": up_odds[(f.home_id, f.away_id)]})
        entries.append(e)

    entries.sort(key=lambda x: x["kickoff"])
    try:   # paper betting from GW12 (descriptive; tabet/paper_bets.py)
        from .paper_bets import build as build_bets
        fin = fpl[fpl["finished"].astype(bool) | fpl["finished_provisional"].astype(bool)]
        results = {int(f.id): ("H" if f.team_h_score > f.team_a_score else "A" if f.team_a_score > f.team_h_score else "D")
                   for f in fin.itertuples() if pd.notna(f.team_h_score) and pd.notna(f.team_a_score)}
        build_bets(logged, odds_log, results, TEAMS, out=out.parent / "paper-bets.json")
    except Exception as ex:  # the site data must still build
        print(f"paper bets skipped ({ex})")
    n_log = len(LOG.read_text().splitlines()) if LOG.exists() else 0
    data = {"built_utc": pd.Timestamp.now(tz="UTC").floor("min").isoformat(), "n_log_records": n_log,
            "seasons": sorted({e["season"] for e in entries}), "matches": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in data.items() if k != "matches"}
    # one small file per season plus an index, so a page loads only the season it shows; one match per
    # line, so the daily rebuild commits a small diff instead of rewriting the file
    files = {}
    for season in data["seasons"]:
        name = f"matches-{season}.json"
        rows = ",\n".join(json.dumps(e, separators=(",", ":")) for e in entries if e["season"] == season)
        (out.parent / name).write_text('{"season":' + json.dumps(season) + ',"matches":[\n' + rows + "\n]}\n")
        files[season] = name
    # starting XIs (actual for played matches, predicted where we have them), one file per season,
    # downloaded only by the match page
    try:
        from .fpl import load_appearances
        from .forecast import ARCHIVE_SEASONS
        app = load_appearances(matches, [s for s in ARCHIVE_SEASONS if s >= data["seasons"][0]], LIVE_SEASON, refresh_live=False)
        actual = _actual_xis(app, who)
    except Exception as ex:  # the site must build even if the appearance data is unavailable
        print(f"actual lineups skipped ({ex})")
        actual = {}
    lfiles = {}
    for season in data["seasons"]:
        rows = []
        for e in entries:
            if e["season"] != season:
                continue
            rec = dict(pred.get(e["id"], {}))
            if e["id"] in actual:
                rec["actual"] = {k: v for k, v in actual[e["id"]].items() if k in ("home", "away")}
                if actual[e["id"]].get("inferred"):
                    rec["inferred"] = True
            if rec:
                rows.append(json.dumps(e["id"]) + ":" + json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
        name = f"lineups-{season}.json"
        (out.parent / name).write_text('{"season":' + json.dumps(season) + ',"lineups":{\n' + ",\n".join(rows) + "\n}}\n")
        lfiles[season] = name
    out.write_text(json.dumps(dict(head, files=files, lineup_files=lfiles), separators=(",", ":")) + "\n")
    return data


def stale(max_age_h: float = 20.0, out: Path = INDEX) -> bool:
    """Rebuild if index.json (`out`) is missing or unreadable, older than max_age_h, the log has records
    it has not seen, or The Odds API has a snapshot newer than the build (so new odds reach the site the
    same day)."""
    if not out.exists():
        return True
    try:
        head = json.loads(out.read_text())
        pd.Timestamp(head["built_utc"])
    except (ValueError, KeyError, TypeError):
        return True
    built, now = pd.Timestamp(head["built_utc"]), pd.Timestamp.now(tz="UTC")
    age = (now - built).total_seconds() / 3600
    n_log = len(LOG.read_text().splitlines()) if LOG.exists() else 0
    return (age >= max_age_h or n_log != head.get("n_log_records") or _newer_odds(built)
            or _match_moment_passed(built, now))


# After these offsets from a logged match's kickoff the site should show something new: "awaiting
# result" at kickoff, then the score (FPL marks a match finished about two hours after kickoff).
RESULT_MOMENTS = (pd.Timedelta(0), pd.Timedelta(hours=2, minutes=15), pd.Timedelta(hours=4))


def _match_moment_passed(built: pd.Timestamp, now: pd.Timestamp) -> bool:
    """True if a logged match kicked off, or reached the time its result should be in, since the build."""
    if not LOG.exists():
        return False
    for line in LOG.read_text().splitlines():
        k = json.loads(line).get("kickoff_utc")
        if not k:
            continue
        ko = pd.Timestamp(k)
        if any(built < ko + d <= now for d in RESULT_MOMENTS):
            return True
    return False


def _newer_odds(built: pd.Timestamp) -> bool:
    from .odds_archive import API_FETCH_LOG, ODDS_DIR
    p = ODDS_DIR / API_FETCH_LOG
    if not p.exists():
        return False
    f = pd.read_csv(p, dtype=str, keep_default_na=False)
    ok = f[f["status"] == "ok"]
    return bool(len(ok)) and pd.Timestamp(ok["retrieved_utc"].iloc[-1]) > built
