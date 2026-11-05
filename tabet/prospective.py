"""Prospective forecast log and scoring (docs/PROSPECTIVE_PROTOCOL.md).

log_due():  run every 2 hours. Each fixture is logged once, by the last run at
            or before kickoff - HORIZON (24 h): the results file is refreshed,
            forecasts are made at that moment, and one JSON record per fixture
            is appended to results/prospective/log.jsonl. Raw lineup pages are
            kept gzipped next to it; nothing is ever overwritten.
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
from .fpl import _ssl_context, fetch_live_season, load_appearances
from .ingest import load_all
from .store import RESULT_LAG, MatchStore
from .teams import try_resolve_team
from .validate import has_errors, validate

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "prospective"
LOG = LOG_DIR / "log.jsonl"
HORIZON_H = 24.0
RUN_EVERY_H = 2.0
RESULTS_URL = "https://football-data.co.uk/mmz4281/2627/E0.csv"
RESULTS_FILE = ROOT / "res" / "Premier League 26-27.csv"
CANDIDATES = ("p0", "ffs", "rotowire", "pE", "fpl_proj", "fpl_proj_hist")


def refresh_results() -> bool:
    """Download the current-season results file; replace ours only if the new
    one parses and validates and has at least as many matches."""
    tmp = RESULTS_FILE.with_suffix(".download")
    req = urllib.request.Request(RESULTS_URL, headers={"User-Agent": "tabet-research/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        tmp.write_bytes(r.read())
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
        return False
    backup.unlink()
    return True


def _logged() -> set[tuple[int, float]]:
    if not LOG.exists():
        return set()
    return {(r["fpl_fixture"], r["horizon_h"]) for r in map(json.loads, LOG.read_text().splitlines())}


def due_fixtures(now: pd.Timestamp, horizon_h: float = HORIZON_H, every_h: float = RUN_EVERY_H) -> pd.DataFrame:
    """Fixtures whose kickoff - horizon falls in [now, now + every_h): this run is
    the last scheduled run at or before their forecast time."""
    fx = upcoming_fixtures(refresh=True)
    lo, hi = now + pd.Timedelta(hours=horizon_h), now + pd.Timedelta(hours=horizon_h + every_h)
    due = fx[(fx["kickoff_utc"] >= lo) & (fx["kickoff_utc"] < hi)]
    done = _logged()
    return due[[(int(i), horizon_h) not in done for i in due["fpl_fixture"]]]


def log_due(now: pd.Timestamp | None = None, horizon_h: float = HORIZON_H, every_h: float = RUN_EVERY_H,
            dry_run: bool = False) -> list[dict]:
    now = (now or pd.Timestamp.now(tz="UTC")).floor("min")
    due = due_fixtures(now, horizon_h, every_h)
    if due.empty or dry_run:
        return due.to_dict("records") if dry_run else []
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
        if "source_average" in w:
            cands["pE"] = {k: w["source_average"][k] for k in ("p_home", "p_draw", "p_away")}
        rec = {
            "fpl_fixture": f["fpl_fixture"], "horizon_h": horizon_h, "logged_utc": now.isoformat(),
            "kickoff_utc": f["kickoff_utc"], "hours_before": (pd.Timestamp(f["kickoff_utc"]) - now).total_seconds() / 3600,
            "gameweek": f["gameweek"], "home_id": f["home_id"], "away_id": f["away_id"],
            "git_commit": fc["git_commit"], "code_dirty": dirty, "results_through": fc["results_through"],
            "results_refreshed": results_refreshed, "n_train_matches": fc["n_train_matches"],
            "candidates": cands, "missing_candidates": [c for c in CANDIDATES if c not in cands],
            "lineup_codes": w.get("lineup_codes", {}), "xi_features": w.get("xi_features", {}),
            "components": f["components"], "xi_stage2": fc["xi_stage2"],
            "lineup_retrieved_utc": stamp, "raw_snapshots": raw_paths,
            "source_updated": {k: v for k, v in fc["source_updated"].items() if k.split(":")[1] in teams},
            "lineup_unresolved": {s: {t: n for t, n in d.items() if t in teams} for s, d in fc["lineup_unresolved"].items()},
            "availability": [a for a in fc["availability"] if a["team_id"] in teams and a["status"] != "a"],
        }
        records.append(rec)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return records


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


def score(out: Path = LOG_DIR / "scores.csv") -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    recs = [json.loads(l) for l in LOG.read_text().splitlines()]
    refresh_results()  # played matches must be in the match table before appearances can join
    xi, res = _actual()
    pending = [r for r in recs if r["fpl_fixture"] in res and r["fpl_fixture"] in xi]
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
        for name, p in cands.items():
            prob = {"H": p["p_home"], "D": p["p_draw"], "A": p["p_away"]}[y]
            row = {"fpl_fixture": r["fpl_fixture"], "gameweek": r["gameweek"], "horizon_h": r["horizon_h"],
                   "logged_utc": r["logged_utc"], "candidate": name, "result": y, "log_loss": float(-np.log(prob)),
                   "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"]}
            codes = r["lineup_codes"].get(name)
            if codes:
                row["starters_correct"] = sum(len(set(codes[s]) & set(actual[s])) for s in ("home", "away")) / 2
                pf = r["xi_features"][name]
                row["xi_feature_abs_error"] = float(np.mean([abs(pf[sd]["f_xi_position"] - act_feat[s]["f_xi_position"])
                                                            for s, sd in (("home", "H"), ("away", "A"))]))
                row["unresolved_names"] = sum(r["lineup_unresolved"].get(name, {}).values())
            rows.append(row)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarise(scores: pd.DataFrame, primary_horizon: float = HORIZON_H) -> pd.DataFrame:
    """Per candidate: coverage, lineup accuracy, feature error, and log-loss vs p0
    on the fixtures both have (each fixture counted once)."""
    s = scores[scores["horizon_h"] == primary_horizon]
    base = s[s["candidate"] == "p0"].set_index("fpl_fixture")["log_loss"]
    out = []
    for c, g in s.groupby("candidate"):
        g = g.set_index("fpl_fixture")
        d = g["log_loss"] - base.reindex(g.index)
        out.append({"candidate": c, "n": len(g), "log_loss": g["log_loss"].mean(), "minus_p0": d.mean(),
                    "se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                    "starters_correct_of_11": g["starters_correct"].mean() if "starters_correct" in g else np.nan,
                    "xi_feature_abs_error": g["xi_feature_abs_error"].mean() if "xi_feature_abs_error" in g else np.nan})
    return pd.DataFrame(out).sort_values("log_loss")


def fit_alpha(scores: pd.DataFrame, through_gameweek: int) -> float:
    """alpha in [0, 1] minimising log-loss of (1-a) p0 + a pE on gameweeks <= through_gameweek."""
    s = scores[(scores["gameweek"] <= through_gameweek) & scores["candidate"].isin(["p0", "pE"])]
    w = s.pivot_table(index=["fpl_fixture", "result"], columns="candidate", values=["p_home", "p_draw", "p_away"]).dropna()
    if w.empty:
        return 0.0
    y = w.index.get_level_values("result")
    col = {"H": "p_home", "D": "p_draw", "A": "p_away"}
    p0 = np.array([w.loc[i, (col[r], "p0")] for i, r in zip(w.index, y)])
    pe = np.array([w.loc[i, (col[r], "pE")] for i, r in zip(w.index, y)])
    grid = np.linspace(0, 1, 101)
    ll = [-np.log((1 - a) * p0 + a * pe).mean() for a in grid]
    return float(grid[int(np.argmin(ll))])
