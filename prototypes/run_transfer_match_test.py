"""Match-level test of transfer priors (docs/PREREGISTRATION.md, addendum T).

Usage: python prototypes/run_transfer_match_test.py HF_DIR FPL_HF_MAP APP_PKL [OUT_DIR]

Priors (known before each PL season S):
  eligible arrival = first FPL season is S, matched to the Hugging Face archive, >= 600 minutes in
  La Liga / Serie A / Bundesliga / Ligue 1 in season S-1, outfield. No condition on PL minutes in S
  (the player-level gate used >= 450, which is not known pre-season).
  prior = the frozen P1 mapping (ridge on position + shrunk previous-league ga/shots/key passes per 90),
  fitted only on earlier cohorts' completed first PL seasons, exactly as in the gate. Clipped at 0.
Blend: the prior replaces the position-prior mean in the unchanged V2 rating formula and fades as
PL minutes accumulate (900-minute weight). Debutants get their prior as rating. No new tuning:
stage-2 L2 = 100 (frozen V2x), xG half-life 90 days, V1x/V2x features otherwise identical.
Contrasts (log-loss, 14-day block bootstrap, 2,000 resamples, seed 0):
  primary     V2xT - V2x, 2024-25 + 2025-26, all matches
  co-primary  the same, on matches where an arrival with a prior starts within his first 10 PL appearances
  reported    2023-24 (the only earlier season with priors); |change in XI ratio| for those matches;
              the arrivals with the largest prior - position-prior gap.
Disclosure: the player-level gate already saw the 2023-26 cohorts' first-season output, and
2024-26 is reused validation for other contrasts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import block_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.features import FeatureParams, known, match_features  # noqa: E402
from tabet.ingest import LOCAL_TZ, load_all  # noqa: E402
from tabet.store import MatchStore  # noqa: E402

HF, MAP, APP = (Path(a) for a in sys.argv[1:4])
out = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "results" / "transfer_match_test"
out.mkdir(parents=True, exist_ok=True)
K90, FIRST_N = 10.0, 10
OTHER = {140: "SP1", 135: "I1", 78: "D1", 61: "F1"}
COHORTS = ("2023-24", "2024-25", "2025-26")


def prev_season(s: str) -> str:
    y = int(s[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


# ---- previous-league player seasons (same construction as the gate) ----
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
prev = prev.sort_values("minutes").drop_duplicates(["player_id", "season"], keep="last")
for c in ("ga", "shots", "kp"):
    prev[f"{c}90"] = prev[c] / (prev["minutes"] / 90)
prev["posg"] = prev["pos"].map({"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"})
ref = prev[prev["minutes"] >= 600].groupby(["league", "season", "posg"])[["ga90", "shots90", "kp90"]].mean()

app = pd.read_pickle(APP)
first = app.groupby("player_code")["season"].min()
pos_first = app.sort_values("kickoff_utc").groupby(["player_code", "season"])["position"].first()
pls = app[app["xg"].notna()].groupby(["player_code", "season"]).agg(
    minutes=("minutes", "sum"), xg=("xg", "sum"), xa=("xa", "sum"), position=("position", "last")).reset_index()
pls["rate"] = (pls["xg"] + pls["xa"]) / (pls["minutes"] / 90)
code2hf = dict(zip(*pd.read_csv(MAP)[["code", "hf_id"]].to_numpy().T))

rows = []
for code, s in first.items():
    if s < "2022-23" or s > "2025-26" or code not in code2hf:
        continue
    pos = pos_first.get((code, s))
    p = prev[(prev["player_id"] == code2hf[code]) & (prev["season"] == prev_season(s))]
    if pos == "GK" or p.empty or p["minutes"].iat[0] < 600:
        continue
    t = pls[(pls["player_code"] == code) & (pls["season"] == s)]
    r = {"player_code": int(code), "cohort": s, "position": pos, "league": p["league"].iat[0], "prev_minutes": p["minutes"].iat[0],
         "pl_minutes": t["minutes"].iat[0] if len(t) else 0.0, "target": t["rate"].iat[0] if len(t) else np.nan}
    for c in ("ga", "shots", "kp"):
        key = (r["league"], prev_season(s), pos)
        mref = ref.loc[key, f"{c}90"] if key in ref.index else np.nan
        r[f"{c}_shr"] = (p[c].iat[0] + K90 * mref) / (p["minutes"].iat[0] / 90 + K90)
    rows.append(r)
co = pd.DataFrame(rows).dropna(subset=["ga_shr", "shots_shr", "kp_shr"])


def design(df):
    return np.column_stack([np.ones(len(df)), (df["position"] == "MID").to_numpy(float), (df["position"] == "FWD").to_numpy(float),
                            df["ga_shr"], df["shots_shr"], df["kp_shr"]])


def ridge(X, y, w, lam=1.0):
    W = np.diag(w / w.mean())
    pen = lam * np.eye(X.shape[1]); pen[0, 0] = 0
    return np.linalg.solve(X.T @ W @ X + pen, X.T @ W @ y)


priors, prior_rows = {}, []
for s in COHORTS:
    tr = co[(co["cohort"] < s) & (co["pl_minutes"] >= 450)]  # earlier cohorts, completed seasons (gate's rule)
    te = co[co["cohort"] == s].copy()
    b = ridge(design(tr), tr["target"].to_numpy(), tr["pl_minutes"].to_numpy(float))
    te["prior"] = np.clip(design(te) @ b, 0, None)
    te["n_train"] = len(tr)
    priors.update(dict(zip(te["player_code"], te["prior"])))
    prior_rows.append(te)
PR = pd.concat(prior_rows, ignore_index=True)
print("eligible arrivals with a prior:", PR.groupby("cohort").size().to_dict(),
      "| with >= 450 PL min:", PR[PR.pl_minutes >= 450].groupby("cohort").size().to_dict())
cohort_of = dict(zip(PR["player_code"], PR["cohort"]))

# ---- XI features with transfer priors, same matchdays as the frozen rows ----
base_rows = pd.read_pickle(ROOT / "results/review_round2/rows.pkl")
cache = out / "xi_transfer.pkl"
if cache.exists():
    xt = pd.read_pickle(cache)
else:
    m, _ = load_all(ROOT / "res")
    store = MatchStore(m)
    target = m[(m["season"] >= "2022-23") & (m["season"] <= "2025-26") & (m["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    parts = []
    for _, g in target.groupby(day, sort=True):
        cutoff = g["kickoff_utc"].min()
        season = g["season"].iloc[0]
        ov = {c: v for c, v in priors.items() if cohort_of[c] <= season}
        f = match_features(g, app[app["match_id"].isin(g["match_id"])], known(app, cutoff), store.completed_before(cutoff),
                           cutoff, FeatureParams(), priors=("position", "transfer"), prior_overrides=ov)
        parts.append(f[["match_id", "side", "f_xi_position", "f_xi_transfer"]])
    xt = pd.concat(parts, ignore_index=True)
    xt.to_pickle(cache)
R = base_rows.merge(xt.rename(columns={"f_xi_position": "chk"}), on=["match_id", "side"], how="left")
have = R["chk"].notna()
assert np.allclose(R.loc[have, "chk"], R.loc[have, "f_xi_position"], atol=1e-9), "recomputed XI feature != frozen rows"
R["f_xi_transfer"] = R["f_xi_transfer"].fillna(R["f_xi_position"])  # 2026-27 rows: no priors, unchanged

# ---- arrivals' early matches ----
a = app[(app["player_code"].isin(priors)) & (app["minutes"] > 0)].sort_values("kickoff_utc")
a = a[a["season"] == a["player_code"].map(cohort_of)]
a["n_app"] = a.groupby("player_code").cumcount() + 1
early = set(a[(a["n_app"] <= FIRST_N) & (a["started"] == 1)]["match_id"])

ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
res = {"n_priors": PR.groupby("cohort").size().to_dict(), "contrasts": {}}
for label, seasons in (("2024-26 (primary)", ("2024-25", "2025-26")), ("2023-24", ("2023-24",))):
    v2 = run_variant(R, ADJ + ["f_xi_position", "f_backline_opp"], 100.0, seasons)
    v2t = run_variant(R, ADJ + ["f_xi_transfer", "f_backline_opp"], 100.0, seasons).set_index("match_id").reindex(v2["match_id"]).reset_index()
    d = per_match_logloss(v2t) - per_match_logloss(v2)
    for sub, mk in (("all matches", np.ones(len(v2), bool)), (f"arrival starts in first {FIRST_N}", v2["match_id"].isin(early).to_numpy())):
        mean, lo, hi = block_bootstrap(per_match_logloss(v2t)[mk], per_match_logloss(v2)[mk],
                                       v2.loc[mk, "kickoff_utc"].reset_index(drop=True), 14)
        se = d[mk].std(ddof=1) / np.sqrt(mk.sum())
        shift = (R[R["match_id"].isin(v2.loc[mk, "match_id"])].eval("f_xi_transfer - f_xi_position").abs().mean())
        res["contrasts"][f"{label} | {sub}"] = {"n": int(mk.sum()), "mean": mean, "se": se, "ci": [lo, hi], "mean_abs_xi_shift": shift}
        print(f"{label:18s} {sub:28s} n={mk.sum():4d} V2xT - V2x {mean:+.5f} (SE {se:.5f}) [{lo:+.5f}, {hi:+.5f}]  |dXI| {shift:.4f}")

# signature arrivals: largest prior over the position prior
pos_prior = {}
for s in COHORTS:
    q = pls[(pls["season"] == prev_season(s)) & (pls["minutes"] >= 900)]
    pos_prior.update({(s, p): float((g["xg"].sum() + g["xa"].sum()) / (g["minutes"].sum() / 90)) for p, g in q.groupby("position")})
PR["position_prior"] = [pos_prior.get((s, p), np.nan) for s, p in zip(PR["cohort"], PR["position"])]
PR["gap"] = PR["prior"] - PR["position_prior"]
names = app.drop_duplicates("player_code", keep="last").set_index("player_code")["player_name"]
PR["name"] = PR["player_code"].map(names)
top = PR.sort_values("gap", ascending=False).head(12)[["name", "cohort", "position", "league", "prior", "position_prior", "target", "pl_minutes"]]
print("\nlargest priors above the position prior:\n" + top.round(3).to_string(index=False))
res["top_arrivals"] = top.to_dict("records")
PR.to_csv(out / "priors.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
