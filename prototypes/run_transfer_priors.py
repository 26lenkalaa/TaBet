"""Transfer priors (SPEC 6.3): does a player's previous-league output predict his first
Premier League season better than the position average?

Usage: python prototypes/run_transfer_priors.py HF_DIR FPL_HF_MAP APP_PKL [OUT_DIR]

Population: players whose first PL season (in FPL data, 2022-23 onward) follows a season with
>= 600 minutes in La Liga, Serie A, Bundesliga or Ligue 1 (Hugging Face soccer-dataset), and who
play >= 450 PL minutes that first season.
Target: first-PL-season (xG + xA) per 90 from FPL, weighted by PL minutes.
Predictors, all from data available before the PL season:
  P0   position average of established PL players (>= 900 min) in the previous PL season (current prior)
  P0b  position average of earlier arrival cohorts (no player-specific information)
  P1   ridge regression on earlier cohorts: position + previous-league per-90 goals+assists,
       shots and key passes, each shrunk toward its league-position mean by minutes (k = 900 min)
  P2   P1 + league dummies
Pre-specified: primary P1 - P0; secondary P0b - P0, P1 - P0b, P2 - P1 (minutes-weighted MSE,
bootstrap over players). Evaluated cohorts: 2023-24, 2024-25, 2025-26 (2022-23 has no P0, since
2021-22 has no FPL xG, so it only serves as training data). Gate: the match-level test runs only if
P1 beats P0 with a CI excluding 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HF, MAP, APP = (Path(a) for a in sys.argv[1:4])
out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parent.parent / "results" / "transfer_priors"
out.mkdir(parents=True, exist_ok=True)
K90 = 10.0  # shrinkage weight: 900 minutes
OTHER = {140: "SP1", 135: "I1", 78: "D1", 61: "F1"}

# ---- previous-league player seasons -------------------------------------------
lg = pd.read_parquet(HF / "leagues.parquet")
ids = lg[lg["api_football_id"].isin(OTHER)]
fx = pd.read_parquet(HF / "fixtures.parquet", columns=["id", "date_utc", "league_id", "is_played"])
fx = fx[fx["league_id"].isin(ids["id"]) & fx["is_played"]]
d = pd.to_datetime(fx["date_utc"], utc=True)
start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
fx["season"] = pd.Series(start, index=fx.index).astype(str) + "-" + pd.Series((start + 1) % 100, index=fx.index).astype(str).str.zfill(2)
fx["league"] = fx["league_id"].map(dict(zip(ids["id"], ids["api_football_id"].map(OTHER))))
st = pd.read_parquet(HF / "fixture_players_stats_flat.parquet",
                     columns=["fixture_id", "player_id", "games_minutes", "games_position", "goals_total",
                              "goals_assists", "shots_total", "passes_key"],
                     filters=[("fixture_id", "in", fx["id"].tolist())])
st = st.merge(fx[["id", "season", "league"]].rename(columns={"id": "fixture_id"}), on="fixture_id")
st = st[st["games_minutes"] > 0].fillna({"goals_total": 0, "goals_assists": 0, "shots_total": 0, "passes_key": 0})
st["ga"] = st["goals_total"] + st["goals_assists"]
prev = st.groupby(["player_id", "season", "league"]).agg(
    minutes=("games_minutes", "sum"), ga=("ga", "sum"), shots=("shots_total", "sum"), kp=("passes_key", "sum"),
    pos=("games_position", lambda s: s.mode().iat[0] if len(s.mode()) else None)).reset_index()
prev = prev.sort_values("minutes").drop_duplicates(["player_id", "season"], keep="last")  # main league that season
for c in ("ga", "shots", "kp"):
    prev[f"{c}90"] = prev[c] / (prev["minutes"] / 90)

# ---- FPL first PL seasons ------------------------------------------------------
app = pd.read_pickle(APP)
app = app[app["season"] <= "2025-26"]
first = app.groupby("player_code")["season"].min()
pls = app[app["xg"].notna()].groupby(["player_code", "season"]).agg(
    minutes=("minutes", "sum"), xgxa=("xg", "sum"), xa=("xa", "sum"), position=("position", "last")).reset_index()
pls["xgxa"] = pls["xgxa"] + pls["xa"]
pls["rate"] = pls["xgxa"] / (pls["minutes"] / 90)
mp = pd.read_csv(MAP)
code2hf = dict(zip(mp["code"], mp["hf_id"]))


def prev_season(s: str) -> str:
    y = int(s[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


rows = []
for code, s in first.items():
    if s < "2022-23" or code not in code2hf:
        continue
    t = pls[(pls["player_code"] == code) & (pls["season"] == s)]
    if t.empty or t["minutes"].iat[0] < 450 or t["position"].iat[0] == "GK":
        continue
    p = prev[(prev["player_id"] == code2hf[code]) & (prev["season"] == prev_season(s))]
    if p.empty or p["minutes"].iat[0] < 600:
        continue
    rows.append({"player_code": code, "cohort": s, "position": t["position"].iat[0], "pl_minutes": t["minutes"].iat[0],
                 "target": t["rate"].iat[0], **{k: p[k].iat[0] for k in ("league", "minutes", "ga90", "shots90", "kp90", "ga", "shots", "kp")}})
co = pd.DataFrame(rows)
print("cohort sizes:", co.groupby("cohort").size().to_dict(), "| by league:", co["league"].value_counts().to_dict())

# shrink previous-league rates toward that league-season-position mean (from all players with >= 600 min)
prev["posg"] = prev["pos"].map({"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"})
ref = prev[prev["minutes"] >= 600].groupby(["league", "season", "posg"])[["ga90", "shots90", "kp90"]].mean()
pos_map = {"DEF": "DEF", "MID": "MID", "FWD": "FWD"}
for c in ("ga", "shots", "kp"):
    m = [ref.loc[(lg_, prev_season(s), pos_map[p]), f"{c}90"] if (lg_, prev_season(s), pos_map[p]) in ref.index else np.nan
         for lg_, s, p in zip(co["league"], co["cohort"], co["position"])]
    co[f"{c}_shr"] = (co[c] + K90 * np.array(m)) / (co["minutes"] / 90 + K90)


def p0(cohort: str, pos: str) -> float:
    q = pls[(pls["season"] == prev_season(cohort)) & (pls["minutes"] >= 900) & (pls["position"] == pos)]
    return float(q["xgxa"].sum() / (q["minutes"].sum() / 90))


def design(df: pd.DataFrame, leagues: bool) -> np.ndarray:
    X = [np.ones(len(df))] + [(df["position"] == p).to_numpy(float) for p in ("MID", "FWD")]
    X += [df[c].to_numpy(float) for c in ("ga_shr", "shots_shr", "kp_shr")]
    if leagues:
        X += [(df["league"] == l).to_numpy(float) for l in ("I1", "D1", "F1")]
    return np.column_stack(X)


def ridge(X, y, w, lam=1.0):
    W = np.diag(w / w.mean())
    pen = lam * np.eye(X.shape[1]); pen[0, 0] = 0
    return np.linalg.solve(X.T @ W @ X + pen, X.T @ W @ y)


preds = []
for s in ("2023-24", "2024-25", "2025-26"):
    tr, te = co[co["cohort"] < s], co[co["cohort"] == s].copy()
    if tr.empty or te.empty:
        continue
    te["P0"] = [p0(s, p) for p in te["position"]]
    te["P0b"] = te["position"].map(tr.groupby("position").apply(lambda g: np.average(g["target"], weights=g["pl_minutes"]),
                                                                  include_groups=False)).fillna(te["P0"])
    for name, lgs in (("P1", False), ("P2", True)):
        b = ridge(design(tr, lgs), tr["target"].to_numpy(), tr["pl_minutes"].to_numpy(float))
        te[name] = design(te, lgs) @ b
    te["n_train"] = len(tr)
    preds.append(te)
P = pd.concat(preds, ignore_index=True)
w = P["pl_minutes"].to_numpy(float)
err = {k: (P[k] - P["target"]).to_numpy() ** 2 for k in ("P0", "P0b", "P1", "P2")}
res = {"n_players": len(P), "cohorts": P.groupby("cohort").size().to_dict(), "train_sizes": P.groupby("cohort")["n_train"].first().to_dict(),
       "wmse": {k: float(np.average(e, weights=w)) for k, e in err.items()}, "contrasts": {}}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(P), size=(4000, len(P)))
for name, (a, b) in {"primary: P1 - P0": ("P1", "P0"), "P0b - P0": ("P0b", "P0"), "P1 - P0b": ("P1", "P0b"),
                     "P2 - P1": ("P2", "P1")}.items():
    boots = np.array([np.average(err[a][i], weights=w[i]) - np.average(err[b][i], weights=w[i]) for i in idx])
    res["contrasts"][name] = {"mean": float(np.average(err[a], weights=w) - np.average(err[b], weights=w)),
                              "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
corr = {c: float(np.corrcoef(P[c], P["target"])[0, 1]) for c in ("ga_shr", "shots_shr", "kp_shr", "P0", "P1")}
res["correlation_with_target"] = corr
res["gate_passed"] = res["contrasts"]["primary: P1 - P0"]["ci"][1] < 0
print(json.dumps(res, indent=2))
P.to_csv(out / "players.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
