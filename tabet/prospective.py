"""Prospective forecast log and scoring (docs/PROSPECTIVE_PROTOCOL.md).

log_due():  run every SCHEDULE_EVERY_H (an external trigger every 30 min, with
            GitHub's own schedule as a backup). Amendment A4: each fixture can get
            two records per kickoff time, a *primary* one from the first run that
            finds its kickoff 24-26 h ahead and a *fallback* one from the first run
            that finds it 26-32 h ahead. Scoring uses the primary record when it
            exists and the fallback otherwise (select_records). The windows are
            protocol values; the run interval is how often something checks. The results file is
            refreshed, forecasts are made at that moment, and one JSON record per
            fixture is appended to results/prospective/log.jsonl. A fixture whose
            kickoff moves after it was logged is logged again against the new
            kickoff; scoring uses only the record whose kickoff matches the one
            played (amendment A2). Raw lineup pages are kept gzipped next to it;
            nothing is ever overwritten. Each run also keeps one daily snapshot
            of live FPL xG for recent gameweeks, to measure later revisions.
            Amendment A6: the first run of each UTC day also refreshes the results file
            (daily_refresh), whether or not anything is due.
score():    after matches are played, joins the confirmed XI and result (FPL
            live data) to every logged forecast and writes per-candidate scores.
fit_alpha(): the one-parameter shrinkage p = (1-a) p0 + a pE, fitted on earlier
            logged matches only (used once the fitting period has been logged).

Candidates logged per fixture (all at the same cutoff):
  p0           V1x, lineup-free (the default early forecast)
  ffs, rotowire  V2x with that site's predicted XI
  pE           plain average of the ffs and rotowire outcome probabilities
  fpl_proj     V2x with our projection minus FPL-flagged absentees
  fpl_proj_hist  the same projection without the availability filter
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import _probs
from .features import FeatureParams, known, match_features
from .forecast import ARCHIVE_SEASONS, LIVE_SEASON, forecast, upcoming_fixtures
from .fpl import API, RAW_DIR, _get, _ssl_context, fetch_live_season, load_appearances
from .ingest import load_all
from .store import RESULT_LAG, MatchStore
from .teams import try_resolve_team
from .validate import has_errors, validate

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "prospective"
LOG = LOG_DIR / "log.jsonl"
HORIZON_H = 24.0          # protocol: forecasts are made at least this long before kickoff
WINDOW_H = 2.0            # protocol: primary record while kickoff is 24-26 h ahead
FALLBACK_H = 26.0         # protocol (amendment A4): fallback record while kickoff is 26-32 h ahead
FALLBACK_WINDOW_H = 6.0
LOG_HORIZONS = ((HORIZON_H, WINDOW_H), (FALLBACK_H, FALLBACK_WINDOW_H))  # (horizon, window): primary first
SCHEDULE_EVERY_H = 0.5    # operations only: how often a run is triggered (external cron, 30 min)
RESULTS_URL = "https://football-data.co.uk/mmz4281/2627/E0.csv"
RESULTS_FILE = ROOT / "res" / "Premier League 26-27.csv"
CANDIDATES = ("p0", "ffs", "rotowire", "pE", "fpl_proj", "fpl_proj_hist", "pBP")
XG_SNAP_DIR = LOG_DIR / "xg_snapshots"
ALPHA_MIN_FIXTURES = 20  # fewer fitting fixtures than this: alpha defaults to 0 (amendment A2)


def _ko(x) -> str:
    return pd.Timestamp(x).tz_convert("UTC").isoformat()


def _n_rows(path: Path) -> int:
    """Data rows in a football-data CSV (blank trailing rows of commas are not matches)."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[1:]
    return sum(1 for ln in lines if ln.strip(", \t"))


def refresh_results(info: dict | None = None) -> bool:
    """Download the current-season results file; replace ours only if the new
    one parses and validates and has at least as many matches. `info`, if given,
    receives the downloaded and resulting local row counts (for monitoring)."""
    tmp = RESULTS_FILE.with_suffix(".download")
    req = urllib.request.Request(RESULTS_URL, headers={"User-Agent": "tabet-research/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        tmp.write_bytes(r.read())
    if info is not None:
        info["remote_rows"] = _n_rows(tmp)
    backup = RESULTS_FILE.with_suffix(".bak")
    shutil.copy(RESULTS_FILE, backup)
    shutil.move(tmp, RESULTS_FILE)
    try:
        m, odds = load_all(ROOT / "res")
        old_n = sum(1 for _ in backup.open(encoding="utf-8-sig")) - 1
        ok = not has_errors(validate(m, odds)) and (m["season"] == LIVE_SEASON).sum() >= old_n
    except Exception:
        ok = False
    if not ok:
        shutil.move(backup, RESULTS_FILE)
    else:
        backup.unlink()
    if info is not None:
        info["local_rows"] = _n_rows(RESULTS_FILE)
    return bool(ok)


RESULTS_REFRESH_LOG = LOG_DIR / "results_refresh.csv"
REFRESH_RETRY = pd.Timedelta(minutes=55)


def _refresh_log() -> pd.DataFrame:
    if not RESULTS_REFRESH_LOG.exists():
        d = pd.DataFrame(columns=["attempted_utc", "ok", "remote_rows", "local_rows", "error"])
    else:
        d = pd.read_csv(RESULTS_REFRESH_LOG)
    d["attempted_utc"] = pd.to_datetime(d["attempted_utc"], utc=True)
    return d


def daily_refresh(now: pd.Timestamp) -> dict | None:
    """Keep the results file current when no forecast is due (amendment A6, item 4). On the first run
    of each UTC day, call refresh_results() (validate-or-restore unchanged). After a failed attempt it
    retries at most hourly until one succeeds that day. Every attempt is appended to
    results/prospective/results_refresh.csv. Returns the attempt, or None if not needed."""
    log = _refresh_log()
    today = log[log["attempted_utc"].dt.strftime("%Y-%m-%d") == f"{now:%Y-%m-%d}"]
    if (today["ok"].astype(str) == "True").any():
        return None
    if len(log) and now - log["attempted_utc"].iloc[-1] < REFRESH_RETRY:
        return None
    info: dict = {"attempted_utc": now.isoformat(), "ok": False, "remote_rows": None, "local_rows": None, "error": ""}
    try:
        info["ok"] = refresh_results(info)
    except Exception as e:
        info["error"] = str(e)[:200]
        if RESULTS_FILE.exists():
            info["local_rows"] = _n_rows(RESULTS_FILE)
    new = not RESULTS_REFRESH_LOG.exists()
    RESULTS_REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([info]).to_csv(RESULTS_REFRESH_LOG, mode="a", header=new, index=False)
    return info


def results_behind(now: pd.Timestamp, log: pd.DataFrame | None = None) -> dict:
    """Monitoring: a UTC day is "behind" if it had refresh attempts but none left the local file with
    as many matches as football-data's (a failed download counts as behind). Alert when today and
    yesterday are both behind."""
    log = _refresh_log() if log is None else log
    if log.empty:
        return {"results_behind": False, "results_detail": None}
    log = log.assign(day=log["attempted_utc"].dt.strftime("%Y-%m-%d"),
                     current=pd.to_numeric(log["local_rows"], errors="coerce")
                     >= pd.to_numeric(log["remote_rows"], errors="coerce"))
    day_ok = log.groupby("day")["current"].any()
    d0, d1 = f"{now:%Y-%m-%d}", f"{now - pd.Timedelta(days=1):%Y-%m-%d}"
    behind = d0 in day_ok and d1 in day_ok and not day_ok[d0] and not day_ok[d1]
    last = log.iloc[-1]
    return {"results_behind": bool(behind),
            "results_detail": {"last_attempt_utc": last["attempted_utc"].isoformat(), "ok": str(last["ok"]) == "True",
                               "local_rows": None if pd.isna(last["local_rows"]) else int(last["local_rows"]),
                               "remote_rows": None if pd.isna(last["remote_rows"]) else int(last["remote_rows"])}}


def _logged() -> set[tuple[int, float, str]]:
    """(fixture, horizon, kickoff) already logged: a moved kickoff is a new key."""
    if not LOG.exists():
        return set()
    return {(r["fpl_fixture"], r["horizon_h"], _ko(r["kickoff_utc"])) for r in map(json.loads, LOG.read_text().splitlines())}


def snapshot_live_xg(now: pd.Timestamp, n_recent: int = 4) -> Path | None:
    """Once per UTC day, re-download the live data of the most recent finished
    gameweeks and keep player xG/xA/xGC as published at that moment."""
    dest = XG_SNAP_DIR / f"{now:%Y-%m-%d}.csv.gz"
    if dest.exists():
        return None
    tmp = RAW_DIR / "xg_snapshots"
    boot = json.loads(_get(f"{API}/bootstrap-static/", tmp / "bootstrap.json", refresh=True).read_text())
    code = {e["id"]: e["code"] for e in boot["elements"]}
    rows = []
    for ev in [e for e in boot["events"] if e["finished"]][-n_recent:]:
        live = json.loads(_get(f"{API}/event/{ev['id']}/live/", tmp / f"gw{ev['id']:02d}.json", refresh=True).read_text())
        for e in live["elements"]:
            s = e["stats"]
            if not s["minutes"]:
                continue
            rows.append({"gameweek": ev["id"], "player_code": code.get(e["id"]), "element": e["id"],
                         "fixtures": ";".join(str(x["fixture"]) for x in e["explain"]), "minutes": s["minutes"],
                         "expected_goals": s["expected_goals"], "expected_assists": s["expected_assists"],
                         "expected_goals_conceded": s["expected_goals_conceded"], "goals_scored": s["goals_scored"],
                         "assists": s["assists"]})
    XG_SNAP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(retrieved_utc=now.isoformat()).to_csv(dest, index=False, compression="gzip")
    return dest


def due_fixtures(now: pd.Timestamp, horizon_h: float = HORIZON_H, window_h: float = WINDOW_H,
                 fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fixtures whose kickoff lies in [now + horizon_h, now + horizon_h + window_h) and that have no
    record yet for that kickoff. The first run inside the window logs a fixture; later runs in the
    same window find it already logged. `fixtures` is for tests; by default the live FPL list."""
    fx = upcoming_fixtures(refresh=True) if fixtures is None else fixtures
    lo, hi = now + pd.Timedelta(hours=horizon_h), now + pd.Timedelta(hours=horizon_h + window_h)
    due = fx[(fx["kickoff_utc"] >= lo) & (fx["kickoff_utc"] < hi)]
    done = _logged()
    keep = np.array([(int(i), horizon_h, _ko(k)) not in done for i, k in zip(due["fpl_fixture"], due["kickoff_utc"])],
                    dtype=bool)
    return due[keep]  # a boolean array keeps the columns even when nothing is due (a plain [] would drop them)


def log_due(now: pd.Timestamp | None = None, horizons: tuple[tuple[float, float], ...] = LOG_HORIZONS,
            dry_run: bool = False) -> list[dict]:
    now = (now or pd.Timestamp.now(tz="UTC")).floor("min")
    if not dry_run:
        try:  # a failed snapshot must never stop forecast logging
            snapshot_live_xg(now)
        except Exception as e:
            print(f"xG snapshot failed ({e})")
        try:  # nor may the daily results refresh (amendment A6)
            r = daily_refresh(now)
            if r:
                print(f"daily results refresh: ok={r['ok']} local={r['local_rows']} football-data={r['remote_rows']}"
                      + (f" ({r['error']})" if r["error"] else ""))
        except Exception as e:
            print(f"daily results refresh failed ({e})")
    fx = upcoming_fixtures(refresh=True)
    due_by_h = {h: due_fixtures(now, h, w, fixtures=fx) for h, w in horizons}
    if dry_run:
        return [dict(r, horizon_h=h) for h, d in due_by_h.items() for r in d.to_dict("records")]
    wanted = {(int(i), h) for h, d in due_by_h.items() for i in d["fpl_fixture"]}
    if not wanted:
        return []
    due = pd.concat(due_by_h.values()).drop_duplicates("fpl_fixture")
    results_refreshed = refresh_results()
    fc = forecast(cutoff=now, fixture_ids=[int(i) for i in due["fpl_fixture"]], refresh=True, lineups=True)
    stamp = fc["lineup_sources"]["retrieved_utc"]
    raw_dir = LOG_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {}
    for src in ("ffs", "rotowire"):
        page = ROOT / "data" / "raw" / "lineups" / src / f"{stamp}.html"
        if page.exists():
            dest = raw_dir / f"{stamp}_{src}.html.gz"
            with page.open("rb") as a, gzip.open(dest, "wb") as b:
                shutil.copyfileobj(a, b)
            raw_paths[src] = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "tabet"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    records = []
    for f in fc["fixtures"]:
        w = f.get("with_lineups") or {}
        teams = {f["home_id"], f["away_id"]}
        cands = {"p0": {k: f[k] for k in ("p_home", "p_draw", "p_away")}}
        for src, p in w.get("sources", {}).items():
            cands[src] = {k: p[k] for k in ("p_home", "p_draw", "p_away")}
        if f.get("bivariate_poisson"):  # secondary contrast pBP - p0 (amendment A5)
            cands["pBP"] = dict(f["bivariate_poisson"])
        if "source_average" in w:
            cands["pE"] = {k: w["source_average"][k] for k in ("p_home", "p_draw", "p_away")}
        rec = {
            "fpl_fixture": f["fpl_fixture"], "horizon_h": None, "logged_utc": now.isoformat(),
            "kickoff_utc": f["kickoff_utc"], "hours_before": (pd.Timestamp(f["kickoff_utc"]) - now).total_seconds() / 3600,
            "gameweek": f["gameweek"], "home_id": f["home_id"], "away_id": f["away_id"],
            "git_commit": fc["git_commit"], "code_dirty": dirty, "results_through": fc["results_through"],
            "results_refreshed": results_refreshed, "n_train_matches": fc["n_train_matches"],
            "candidates": cands, "missing_candidates": [c for c in CANDIDATES if c not in cands],
            "pE_sources": w.get("source_average_of", []),
            "lineup_codes": w.get("lineup_codes", {}), "xi_features": w.get("xi_features", {}),
            "components": f["components"], "xi_stage2": fc["xi_stage2"],
            "lineup_retrieved_utc": stamp, "raw_snapshots": raw_paths,
            "source_updated": {k: v for k, v in fc["source_updated"].items() if k.split(":")[1] in teams},
            "lineup_unresolved": {s: {t: n for t, n in d.items() if t in teams} for s, d in fc["lineup_unresolved"].items()},
            "lineup_invalid": {s: {t: n for t, n in d.items() if t in teams} for s, d in fc.get("lineup_invalid", {}).items()},
            "availability": [a for a in fc["availability"] if a["team_id"] in teams and a["status"] != "a"],
        }
        for h, _ in horizons:  # one record per window this fixture is due in (same forecast, same cutoff)
            if (int(f["fpl_fixture"]), h) in wanted:
                records.append(dict(rec, horizon_h=h, record_role="primary" if h == HORIZON_H else "fallback"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = _logged()  # re-read just before appending: never write a key that is already there
    records = [r for r in records if (r["fpl_fixture"], r["horizon_h"], _ko(r["kickoff_utc"])) not in done]
    with LOG.open("a") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return records


def duplicate_keys(path: Path = LOG) -> list[tuple]:
    """(fixture, horizon, kickoff) keys that appear more than once in the log (should be none)."""
    if not path.exists():
        return []
    keys = [(r["fpl_fixture"], r["horizon_h"], _ko(r["kickoff_utc"])) for r in map(json.loads, path.read_text().splitlines())]
    seen, dup = set(), []
    for k in keys:
        if k in seen:
            dup.append(k)
        seen.add(k)
    return dup


A4_EFFECTIVE = pd.Timestamp("2026-11-19", tz="UTC")  # misses before this are in the operations log


def heartbeat(now: pd.Timestamp, fixtures: pd.DataFrame | None = None, logged: set | None = None) -> dict:
    """Status for the job summary: the next fixture with no record yet, when its windows open, whether
    one is open now, fixtures whose primary window closed with no record at all (missed), and fixtures
    that only got a fallback record."""
    fx = upcoming_fixtures(refresh=False) if fixtures is None else fixtures
    done = _logged() if logged is None else logged
    first = max(h + w for h, w in LOG_HORIZONS)
    fx = fx.assign(opens=fx["kickoff_utc"] - pd.Timedelta(hours=first),
                   primary_opens=fx["kickoff_utc"] - pd.Timedelta(hours=HORIZON_H + WINDOW_H),
                   closes=fx["kickoff_utc"] - pd.Timedelta(hours=HORIZON_H))
    key = [(int(i), _ko(k)) for i, k in zip(fx["fpl_fixture"], fx["kickoff_utc"])]
    have = {(f, k) for f, h, k in done}
    have_primary = {(f, k) for f, h, k in done if h == HORIZON_H}
    fx["any"] = [k in have for k in key]
    fx["primary"] = [k in have_primary for k in key]
    recent = (fx["kickoff_utc"] > now - pd.Timedelta(days=3)) & (fx["kickoff_utc"] >= A4_EFFECTIVE)
    pending = fx[~fx["any"] & (fx["closes"] > now)].sort_values("opens")
    out = {"now": now.isoformat(), "window_open_now": bool((pending["opens"] <= now).any()),
           "missed": [int(i) for i in fx.loc[~fx["any"] & (fx["closes"] <= now) & recent, "fpl_fixture"]],
           "fallback_only": [int(i) for i in fx.loc[fx["any"] & ~fx["primary"] & (fx["closes"] <= now) & recent, "fpl_fixture"]],
           "next": None}
    if len(pending):
        n = pending.iloc[0]
        out["next"] = {"fpl_fixture": int(n["fpl_fixture"]), "match": f"{n['home_id']} v {n['away_id']}",
                       "kickoff_utc": n["kickoff_utc"].isoformat(), "window_opens_utc": n["opens"].isoformat(),
                       "primary_window_opens_utc": n["primary_opens"].isoformat(),
                       "window_closes_utc": n["closes"].isoformat(),
                       "hours_to_window": round((n["opens"] - now).total_seconds() / 3600, 1)}
    return out


def heartbeat_markdown(hb: dict, last_record: str | None) -> str:
    lines = ["### Prospective logger heartbeat", "", f"- Run time: {hb['now']}",
             f"- Last logged record: {last_record or 'none yet'}"]
    n = hb["next"]
    if n:
        when = "open now" if hb["window_open_now"] else f"opens in {n['hours_to_window']} h"
        lines.append(f"- Next fixture to log: {n['match']} (FPL {n['fpl_fixture']}), kickoff {n['kickoff_utc']}; "
                     f"fallback window opens {n['window_opens_utc']}, primary {n['primary_window_opens_utc']} to "
                     f"{n['window_closes_utc']} ({when})")
    else:
        lines.append("- Next fixture to log: none scheduled")
    if hb.get("fallback_only"):
        lines.append(f"- Fallback record only (primary window passed): FPL fixtures {hb['fallback_only']}")
    if hb["missed"]:
        lines.append(f"- **Missed (window closed, no record): FPL fixtures {hb['missed']}**")
    return "\n".join(lines) + "\n"


# ---- scoring -----------------------------------------------------------------

def _actual(season: str = LIVE_SEASON) -> tuple[dict[int, dict], dict[int, str]]:
    """Confirmed starters by fixture and side, and results, from the FPL live data."""
    d = fetch_live_season(season, refresh=True)
    fx = json.loads((d / "fixtures.json").read_text())
    boot = json.loads((d / "bootstrap.json").read_text())
    tid = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    code = {e["id"]: e["code"] for e in boot["elements"]}
    side = {}
    res = {}
    for x in fx:
        if not x["finished"]:
            continue
        h, a = x["team_h_score"], x["team_a_score"]
        res[x["id"]] = "H" if h > a else ("A" if a > h else "D")
        for st in x["stats"] or []:
            if st["identifier"] == "bps":
                side.update({(x["id"], e["element"]): "home" for e in st["h"]})
                side.update({(x["id"], e["element"]): "away" for e in st["a"]})
    xi: dict[int, dict] = {}
    for f in sorted(d.glob("gw*.json")):
        for e in json.loads(f.read_text())["elements"]:
            if e["stats"]["starts"] and len(e["explain"]) == 1:
                fid = e["explain"][0]["fixture"]
                s = side.get((fid, e["id"]))
                if s and fid in res:
                    xi.setdefault(fid, {"home": [], "away": []})[s].append(code[e["id"]])
    return xi, res


def _lam(rec: dict, side: str, xi_feat: dict) -> float:
    st = rec["xi_stage2"]
    c = rec["components"]
    x = np.array(c[f"xgadj_{side}"] + [xi_feat["f_xi_position"],
                 xi_feat["f_backline_opp"] if xi_feat["f_backline_opp"] is not None else st["mean"][3]])
    z = (x - np.array(st["mean"])) / np.array(st["sd"])
    return c[f"dc_rate_{side}"] * float(np.exp(st["coef"][0] + z @ np.array(st["coef"][1:])))


def _updated_days(updated: dict, source: str, logged_utc: str) -> float | None:
    """Amendment A5: whole days between a source's "last updated" date (e.g. "Tue 3rd Nov") and the
    record's UK date, the larger over the fixture's two teams. None if no text parses."""
    import re
    logged = pd.Timestamp(logged_utc).tz_convert("Europe/London")
    out = []
    for key, text in updated.items():
        if key.split(":")[0] != source or not isinstance(text, str):
            continue
        mt = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3})", text)
        if not mt:
            continue
        try:
            d = pd.Timestamp(f"{mt.group(1)} {mt.group(2)} {logged.year}")
        except ValueError:
            continue
        if d.date() > logged.date():  # e.g. "31st Dec" read in January
            d = d - pd.DateOffset(years=1)
        out.append(float((pd.Timestamp(logged.date()) - d).days))
    return max(out) if out else None


def score(out: Path = LOG_DIR / "scores.csv") -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    recs = [json.loads(l) for l in LOG.read_text().splitlines()]
    refresh_results()  # played matches must be in the match table before appearances can join
    xi, res = _actual()
    pending = [r for r in recs if r["fpl_fixture"] in res and r["fpl_fixture"] in xi]
    played_ko = _played_kickoffs()
    if not pending:
        return pd.DataFrame()
    matches, _ = load_all(ROOT / "res")
    app = load_appearances(matches, ARCHIVE_SEASONS, LIVE_SEASON, refresh_live=False)
    store = MatchStore(matches)
    rows = []
    for r in pending:
        y = res[r["fpl_fixture"]]
        actual = xi[r["fpl_fixture"]]
        cutoff = pd.Timestamp(r["logged_utc"])
        # controlled comparison: the same frozen early model with the confirmed XI substituted
        fx = pd.DataFrame([{"match_id": str(r["fpl_fixture"]), "home_id": r["home_id"], "away_id": r["away_id"],
                            "season": LIVE_SEASON}])
        pos = app.drop_duplicates("player_code", keep="last").set_index("player_code")["position"]
        lu = pd.DataFrame([{"match_id": str(r["fpl_fixture"]), "team_id": r[f"{s}_id"], "player_code": c,
                            "position": pos.get(c, "MID"), "started": 1.0} for s in ("home", "away") for c in actual[s]])
        feats = match_features(fx, lu, known(app, cutoff), store.completed_before(cutoff), cutoff, FeatureParams(),
                               priors=("position",))
        act_feat = {s: {"f_xi_position": float(feats.loc[feats.side == sd, "f_xi_position"].iloc[0]),
                        "f_backline_opp": (None if pd.isna(feats.loc[feats.side == sd, "f_backline_opp"].iloc[0])
                                           else float(feats.loc[feats.side == sd, "f_backline_opp"].iloc[0]))}
                    for s, sd in (("home", "H"), ("away", "A"))}
        lh, la = _lam(r, "home", act_feat["home"]), _lam(r, "away", act_feat["away"])
        pc = _probs(np.array([lh]), np.array([la]), np.array([r["components"]["rho"]]))[0]
        cands = dict(r["candidates"])
        cands["controlled_confirmed_xi"] = {"p_home": pc[0], "p_draw": pc[1], "p_away": pc[2]}
        pe_basis = "+".join(r.get("pE_sources") or [s for s in ("ffs", "rotowire") if s in cands])
        if "pE" not in cands:  # neither source had both XIs: pE falls back to p0 (amendment A2)
            cands["pE"], pe_basis = dict(cands["p0"]), "p0"
        ages = {src: _updated_days(r.get("source_updated", {}), src, r["logged_utc"]) for src in ("ffs", "rotowire")}
        superseded = _ko(r["kickoff_utc"]) != played_ko.get(r["fpl_fixture"], _ko(r["kickoff_utc"]))
        for name, p in cands.items():
            prob = {"H": p["p_home"], "D": p["p_draw"], "A": p["p_away"]}[y]
            row = {"fpl_fixture": r["fpl_fixture"], "gameweek": r["gameweek"], "horizon_h": r["horizon_h"],
                   "home_id": r["home_id"], "away_id": r["away_id"],
                   "kickoff_utc": r["kickoff_utc"], "hours_before": r.get("hours_before"),
                   "logged_utc": r["logged_utc"], "candidate": name, "result": y, "log_loss": float(-np.log(prob)),
                   "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"], "superseded": superseded,
                   "pE_basis": pe_basis if name == "pE" else None, "record_role": r.get("record_role"),
                   "ffs_updated_days": ages["ffs"], "rotowire_updated_days": ages["rotowire"],
                   "conf_feat_home": act_feat["home"]["f_xi_position"], "conf_feat_away": act_feat["away"]["f_xi_position"]}
            codes = r["lineup_codes"].get(name)
            if codes:
                row["starters_correct"] = sum(len(set(codes[s]) & set(actual[s])) for s in ("home", "away")) / 2
                pf = r["xi_features"][name]
                row["xi_feature_abs_error"] = float(np.mean([abs(pf[sd]["f_xi_position"] - act_feat[s]["f_xi_position"])
                                                            for s, sd in (("home", "H"), ("away", "A"))]))
                row["pred_feat_home"], row["pred_feat_away"] = pf["H"]["f_xi_position"], pf["A"]["f_xi_position"]
                row["unresolved_names"] = sum(r["lineup_unresolved"].get(name, {}).values())
            rows.append(row)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _played_kickoffs(season: str = LIVE_SEASON) -> dict[int, str]:
    d = fetch_live_season(season, refresh=False)
    return {x["id"]: _ko(x["kickoff_time"]) for x in json.loads((d / "fixtures.json").read_text())
            if x["finished"] and x["kickoff_time"]}


def select_records(scores: pd.DataFrame) -> pd.DataFrame:
    """Amendment A4: one record per fixture. Drop superseded records (moved kickoffs), then use the
    primary (24-26 h) record where one exists, else the fallback (26-32 h) one."""
    s = scores
    if "superseded" in s:
        s = s[~s["superseded"].astype(bool)]
    if s.empty:
        return s
    best = s.groupby("fpl_fixture")["horizon_h"].min()
    return s[s["horizon_h"] == s["fpl_fixture"].map(best)]


def summarise(scores: pd.DataFrame, primary_horizon: float = HORIZON_H) -> pd.DataFrame:
    """Per candidate: coverage, lineup accuracy, feature error, and log-loss vs p0
    on the fixtures both have (each fixture counted once; superseded records
    from moved kickoffs are excluded and counted separately)."""
    s = select_records(scores)
    base = s[s["candidate"] == "p0"].set_index("fpl_fixture")["log_loss"]
    out = []
    for c, g in s.groupby("candidate"):
        g = g.set_index("fpl_fixture")
        d = g["log_loss"] - base.reindex(g.index)
        out.append({"candidate": c, "n": len(g), "log_loss": g["log_loss"].mean(), "minus_p0": d.mean(),
                    "se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                    "starters_correct_of_11": g["starters_correct"].mean() if "starters_correct" in g else np.nan,
                    "xi_feature_abs_error": g["xi_feature_abs_error"].mean() if "xi_feature_abs_error" in g else np.nan,
                    "pE_fallback_rate": (g["pE_basis"] == "p0").mean() if c == "pE" and "pE_basis" in g else np.nan})
    return pd.DataFrame(out).sort_values("log_loss")


def robustness(scores: pd.DataFrame, candidate: str = "pE", primary_horizon: float = HORIZON_H) -> dict:
    """Amendment A3 reporting: the candidate - p0 mean with fallbacks included (primary), on complete
    cases only (both sources valid), and with each gameweek left out in turn."""
    s = select_records(scores)
    ll = s.pivot_table(index="fpl_fixture", columns="candidate", values="log_loss")
    if candidate not in ll or "p0" not in ll:
        return {}
    d = (ll[candidate] - ll["p0"]).dropna()
    gw = s.drop_duplicates("fpl_fixture").set_index("fpl_fixture")["gameweek"].reindex(d.index)
    out = {"n": int(len(d)), "mean_incl_fallbacks": float(d.mean())}
    if "pE_basis" in s and candidate == "pE":
        basis = s[s["candidate"] == "pE"].set_index("fpl_fixture")["pE_basis"].reindex(d.index)
        cc = d[basis == "ffs+rotowire"]
        out.update({"n_complete": int(len(cc)), "mean_complete_case": float(cc.mean()) if len(cc) else None})
    out["leave_one_gameweek_out"] = {int(g): float(d[gw != g].mean()) for g in sorted(gw.unique())}
    # paired SE and a gameweek-clustered SE with a t reference on G - 1 df; the clustered SE is the
    # primary measure for the gameweek 20-38 confirmatory read (amendment A5), plus a sign-flip test
    from scipy.stats import t as tdist
    out["se_paired"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    g = d.groupby(gw.to_numpy())
    G = g.ngroups
    if G > 1:
        resid = (g.sum() - g.size() * d.mean())
        se_c = float(np.sqrt(G / (G - 1) * (resid ** 2).sum()) / len(d))
        q = float(tdist.ppf(0.975, G - 1))
        sums = g.sum().to_numpy()
        flips = np.random.default_rng(0).choice([-1.0, 1.0], size=(10_000, G))
        null = np.abs(flips @ sums)
        out["sign_flip_p_gameweek"] = float((1 + (null >= abs(sums.sum()) - 1e-12).sum()) / (1 + len(null)))
        out.update({"se_gameweek_clustered": se_c, "ci_gameweek_clustered": [float(d.mean() - q * se_c), float(d.mean() + q * se_c)],
                    "n_gameweeks": int(G)})
    return out


def by_hours_before(scores: pd.DataFrame, bands=((24, 26), (26, 32)), primary_horizon: float = HORIZON_H) -> pd.DataFrame:
    """Amendment A4: each candidate - p0 by how long before kickoff the record was made."""
    s = select_records(scores)
    if "hours_before" not in s or s["hours_before"].isna().all():
        return pd.DataFrame()
    base = s[s["candidate"] == "p0"].set_index("fpl_fixture")["log_loss"]
    out = []
    for lo, hi in bands:
        b = s[(s["hours_before"] >= lo) & (s["hours_before"] < hi)]
        for c, g in b.groupby("candidate"):
            d = g.set_index("fpl_fixture")["log_loss"] - base.reindex(g["fpl_fixture"]).to_numpy()
            out.append({"hours_before": f"{lo}-{hi}", "candidate": c, "n": len(g), "minus_p0": float(d.mean())})
    return pd.DataFrame(out)


def decomposition(scores: pd.DataFrame, primary_horizon: float = HORIZON_H) -> pd.DataFrame:
    """Pre-registered secondary analysis (amendment A2). For each lineup source:
    capture = 1 - mean((predicted XI feature - confirmed XI feature)^2) / var(confirmed feature),
    over team-matches; confirmed_gain = controlled_confirmed_xi - p0 log-loss on the same fixtures;
    implied_gain = capture * confirmed_gain, compared with the observed source - p0 gain.
    The feature error is measured per team-match, so it is far less noisy than log-loss."""
    s = select_records(scores)
    ll = s.pivot_table(index="fpl_fixture", columns="candidate", values="log_loss")
    out = []
    for src in ("ffs", "rotowire", "fpl_proj", "fpl_proj_hist"):
        g = s[(s["candidate"] == src) & s["pred_feat_home"].notna()] if "pred_feat_home" in s else s.iloc[0:0]
        if g.empty or "controlled_confirmed_xi" not in ll:
            continue
        pred = np.r_[g["pred_feat_home"], g["pred_feat_away"]].astype(float)
        conf = np.r_[g["conf_feat_home"], g["conf_feat_away"]].astype(float)
        cap = 1 - np.mean((pred - conf) ** 2) / np.var(conf, ddof=1) if len(conf) > 2 else np.nan
        f = g["fpl_fixture"]
        conf_gain = (ll.loc[f, "controlled_confirmed_xi"] - ll.loc[f, "p0"]).mean()
        obs = (ll.loc[f, src] - ll.loc[f, "p0"]).mean()
        out.append({"source": src, "n_fixtures": len(g), "feature_capture": cap, "feature_mae": np.mean(np.abs(pred - conf)),
                    "confirmed_gain": conf_gain, "implied_gain": cap * conf_gain, "observed_gain": obs})
    return pd.DataFrame(out)


def fit_alpha(scores: pd.DataFrame, through_gameweek: int, known_before: pd.Timestamp | None = None) -> float:
    """alpha in [0, 1] minimising log-loss of (1-a) p0 + a pE on gameweeks <= through_gameweek.
    Fitted once (not walk-forward) at the end of the fitting period and frozen. Defaults to 0 when
    the fit is degenerate: fewer than ALPHA_MIN_FIXTURES fixtures, or a flat log-loss curve."""
    scores = select_records(scores)
    if known_before is not None:  # amendment A3: only results known before the first evaluation cutoff
        scores = scores[pd.to_datetime(scores["kickoff_utc"], utc=True) + RESULT_LAG <= known_before]
    s = scores[(scores["gameweek"] <= through_gameweek) & scores["candidate"].isin(["p0", "pE"])]
    w = s.pivot_table(index=["fpl_fixture", "result"], columns="candidate", values=["p_home", "p_draw", "p_away"]).dropna()
    if len(w) < ALPHA_MIN_FIXTURES:
        return 0.0
    y = w.index.get_level_values("result")
    col = {"H": "p_home", "D": "p_draw", "A": "p_away"}
    p0 = np.array([w.loc[i, (col[r], "p0")] for i, r in zip(w.index, y)])
    pe = np.array([w.loc[i, (col[r], "pE")] for i, r in zip(w.index, y)])
    grid = np.linspace(0, 1, 101)
    ll = np.array([-np.log((1 - a) * p0 + a * pe).mean() for a in grid])
    if np.ptp(ll) < 1e-12:
        return 0.0
    return float(grid[int(np.argmin(ll))])


def by_record_role(scores: pd.DataFrame, candidates=("pE", "pBP")) -> pd.DataFrame:
    """Amendment A5: candidate - p0 separately for primary and fallback records (the scored record
    per fixture), and by days since each lineup source last updated."""
    s = select_records(scores)
    if s.empty:
        return pd.DataFrame()
    role = np.where(s["horizon_h"] == HORIZON_H, "primary", "fallback")
    s = s.assign(role=role)
    base = s[s["candidate"] == "p0"].set_index("fpl_fixture")["log_loss"]
    out = []
    for c in candidates:
        g = s[s["candidate"] == c]
        d = g.set_index("fpl_fixture")["log_loss"] - base.reindex(g["fpl_fixture"]).to_numpy()
        for r in ("primary", "fallback"):
            dd = d[(g["role"] == r).to_numpy()]
            out.append({"candidate": c, "split": f"record {r}", "n": len(dd), "minus_p0": float(dd.mean()) if len(dd) else np.nan})
        for src in ("ffs", "rotowire"):
            col = f"{src}_updated_days"
            if col not in g:
                continue
            age = g[col].to_numpy(dtype=float)
            for lab, m in ((f"{src} updated same day", age == 0), (f"{src} updated 1+ days before", age >= 1)):
                dd = d[m]
                out.append({"candidate": c, "split": lab, "n": len(dd), "minus_p0": float(dd.mean()) if len(dd) else np.nan})
    return pd.DataFrame(out)


def alpha_report(scores: pd.DataFrame, through_gameweek: int = 19, known_before: pd.Timestamp | None = None,
                 eval_gameweeks: tuple[int, int] = (20, 38)) -> dict:
    """Amendment A5: the fitted alpha, its likelihood interval (grid values whose summed log-loss is
    within 1.92 of the minimum), and the evaluation contrast pa - p0 at alpha = 0, 0.5 and 1."""
    a = fit_alpha(scores, through_gameweek, known_before)
    s = select_records(scores)
    fit = s
    if known_before is not None:
        fit = fit[pd.to_datetime(fit["kickoff_utc"], utc=True) + RESULT_LAG <= known_before]
    col = {"H": "p_home", "D": "p_draw", "A": "p_away"}

    def probs(df):
        w = df[df["candidate"].isin(["p0", "pE"])].pivot_table(index=["fpl_fixture", "result"], columns="candidate",
                                                                values=["p_home", "p_draw", "p_away"]).dropna()
        y = w.index.get_level_values("result")
        return (np.array([w.loc[i, (col[r], "p0")] for i, r in zip(w.index, y)]),
                np.array([w.loc[i, (col[r], "pE")] for i, r in zip(w.index, y)]))

    out = {"alpha": a}
    p0, pe = probs(fit[fit["gameweek"] <= through_gameweek])
    if len(p0) >= ALPHA_MIN_FIXTURES:
        grid = np.linspace(0, 1, 101)
        tot = np.array([-np.log((1 - g) * p0 + g * pe).sum() for g in grid])
        ok = grid[tot <= tot.min() + 1.92]
        out.update({"n_fit": int(len(p0)), "interval": [float(ok.min()), float(ok.max())]})
    lo, hi = eval_gameweeks
    e0, ee = probs(s[(s["gameweek"] >= lo) & (s["gameweek"] <= hi)])
    if len(e0):
        out["eval_minus_p0"] = {str(g): float((-np.log((1 - g) * e0 + g * ee) + np.log(e0)).mean()) for g in (0.0, 0.5, 1.0)}
        out["n_eval"] = int(len(e0))
    return out

