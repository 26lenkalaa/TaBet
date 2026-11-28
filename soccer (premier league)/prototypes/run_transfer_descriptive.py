"""Descriptive look (development data, no decision): does summer transfer spending explain where the
results-only Dixon-Coles (D0) is wrong early in a season, especially for promoted teams?

    python prototypes/run_transfer_descriptive.py

PL 2016-17 -> 2025-26 (2015-16 burn-in; 2026-27 left out: it is the live test). For every team-match,
D0's pre-match expected goals (refitted every matchday on earlier results) and the club's transfer
features at that kickoff (tabet.transfers.club_features: moves from 1 May up to the day before).
Residual = goal difference minus expected goal difference, from the team's side. A positive residual
means the team did better than D0 expected. Not a test: it can only suggest what to pre-register.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import transfers as T  # noqa: E402
from tabet.history import load_history_matches  # noqa: E402
from tabet.ingest import LOCAL_TZ, load_all  # noqa: E402
from tabet.model import DixonColes, Hyperparams, promoted_team_seasons  # noqa: E402
from tabet.store import MatchStore  # noqa: E402

SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2016, 2026)]


def _matches():
    h = load_history_matches()
    m, _ = load_all(ROOT / "res")
    m = pd.concat([h, m[m["season"] > h["season"].max()]], ignore_index=True)
    return m[m["season"] <= "2025-26"]


def d0_rates(season: str) -> pd.DataFrame:
    m = _matches()
    store, promoted = MatchStore(m), promoted_team_seasons(m)
    tgt = m[(m["season"] == season) & (m["status"] == "completed")]
    out = []
    for _, g in tgt.groupby(tgt["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date, sort=True):
        cut = g["kickoff_utc"].min()
        dc = DixonColes(Hyperparams()).fit(store.completed_before(cut), cut, promoted)
        for r in g.itertuples():
            lh, la = dc.expected_goals(r.home_id, r.away_id, r.season)
            out.append({"match_id": r.match_id, "lam_home": lh, "lam_away": la})
    return pd.DataFrame(out)


def main():
    m = _matches()
    with ProcessPoolExecutor(5) as ex:
        rates = pd.concat(ex.map(d0_rates, SEASONS), ignore_index=True)
    mm = m[m["season"].isin(SEASONS) & (m["status"] == "completed")].merge(rates, on="match_id")
    promoted = promoted_team_seasons(m)
    moves = T.load()
    index = T.fee_index(moves)
    cache = {}

    def cost(team, kickoff):
        k = (team, pd.Timestamp(kickoff).normalize())
        if k not in cache:
            cache[k] = T.squad_cost(moves, team, kickoff, index)["squad_cost"]
        return cache[k]
    rows = []
    for r in mm.itertuples():
        for side, team, gf, ga, lf, la in (("H", r.home_id, r.home_goals, r.away_goals, r.lam_home, r.lam_away),
                                           ("A", r.away_id, r.away_goals, r.home_goals, r.lam_away, r.lam_home)):
            rows.append({"season": r.season, "team": team, "kickoff_utc": r.kickoff_utc, "side": side,
                         "resid_gd": (gf - ga) - (lf - la), "promoted": (team, r.season) in promoted,
                         "squad_cost": cost(team, r.kickoff_utc),
                         **T.club_features(moves, team, int(r.season[:4]), r.kickoff_utc)})
    d = pd.DataFrame(rows).sort_values("kickoff_utc")
    d["n"] = d.groupby(["season", "team"]).cumcount()
    d["phase"] = np.where(d["n"] < 10, "first 10", "rest")
    # each club's spend relative to its season's other clubs (z of log(1 + fees in), and of net spend)
    ts = d.groupby(["season", "team"]).agg(fees_in=("fees_in", "max"), net=("net_spend", "max"), promoted=("promoted", "first"),
                                           signings=("signings", "max"), squad_cost=("squad_cost", "first"))
    ts["z_log_in"] = ts.groupby("season")["fees_in"].transform(lambda x: (np.log1p(x) - np.log1p(x).mean()) / np.log1p(x).std())
    ts["z_net"] = ts.groupby("season")["net"].transform(lambda x: (x - x.mean()) / x.std())
    ts["z_log_cost"] = ts.groupby("season")["squad_cost"].transform(lambda x: (np.log1p(x) - np.log1p(x).mean()) / np.log1p(x).std())
    res = {"note": "descriptive; development data PL 2016-26; no decision", "team_seasons": int(len(ts)),
           "promoted_team_seasons": int(ts["promoted"].sum())}
    for phase in ("first 10", "rest"):
        g = d[d["phase"] == phase].groupby(["season", "team"])["resid_gd"].mean().rename("resid")
        t = ts.join(g)
        for who, sub in (("all", t), ("promoted", t[t["promoted"]]), ("not promoted", t[~t["promoted"]])):
            res[f"{phase} | {who}"] = {
                "n": int(len(sub)), "mean residual GD per match": float(sub["resid"].mean()),
                "corr(resid, z log fees in)": float(sub["resid"].corr(sub["z_log_in"])),
                "corr(resid, z net spend)": float(sub["resid"].corr(sub["z_net"])),
                "corr(resid, signings)": float(sub["resid"].corr(sub["signings"])),
                "corr(resid, z log squad cost)": float(sub["resid"].corr(sub["z_log_cost"])),
                "both in one fit: GD per +1 sd (fees in, squad cost)": [float(v) for v in np.linalg.lstsq(
                    np.c_[np.ones(len(sub)), sub["z_log_in"], sub["z_log_cost"]], sub["resid"].to_numpy(), rcond=None)[0][1:]]
                    if len(sub) > 3 else None,
                "slope GD per +1 sd log fees in": float(np.polyfit(sub["z_log_in"], sub["resid"], 1)[0]) if len(sub) > 2 else None}
    pt = ts[ts["promoted"]].join(d[d["phase"] == "first 10"].groupby(["season", "team"])["resid_gd"].mean().rename("resid_first10"))
    res["promoted clubs, by summer fees in"] = pt.sort_values("fees_in", ascending=False)[["fees_in", "net", "signings", "squad_cost", "resid_first10"]] \
        .round(2).reset_index().to_dict("records")
    out = ROOT / "results" / "transfers_descriptive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    for k, v in res.items():
        if k != "promoted clubs, by summer fees in":
            print(k, json.dumps(v, default=lambda x: round(float(x), 3)))
    print(pd.DataFrame(res["promoted clubs, by summer fees in"]).to_string())


if __name__ == "__main__":
    main()
