"""TaBet vs the betting market on existing out-of-sample predictions (benchmark only).

Usage: python prototypes/run_market_comparison.py [APP_PKL] [OUT_DIR]   (default results/market_comparison)

Forecasts: the frozen walk-forward ledger (results/xg_strength/ledger.csv), 2022-23 -> 2026-27 GW1-9,
  V2x = confirmed-XI model (cutoff = the matchday's first kickoff; XIs ~1 h before each kickoff)
  V1x = lineup-free model (same cutoff); V0 and V1x rows also give the per-layer decomposition.
Market: football-data average odds, Avg* (pre-closing: collected Friday afternoon for weekend games,
  Tuesday afternoon for midweek, so EARLIER information than V2x) and AvgC* (closing: LATER
  information than V2x). Implied probability = (1/o_k) / sum_j (1/o_j) (proportional margin removal).
Picks: highest-probability outcome, draws included. A forecast whose top two probabilities are within
  1e-9 counts as tied; tied matches are reported and left out of the pick analysis only.
Uncertainty: 14- and 28-day calendar-block bootstrap (2,000 resamples, seed 0) on paired log-loss.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import block_bootstrap  # noqa: E402
from tabet.features import known  # noqa: E402
from tabet.ingest import implied_probabilities, load_all  # noqa: E402

APP = Path(sys.argv[1]) if len(sys.argv) > 1 else None
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "market_comparison"
out.mkdir(parents=True, exist_ok=True)
OUT = ("H", "D", "A")
COLS = ["p_home", "p_draw", "p_away"]

m, odds = load_all(ROOT / "res")
led = pd.read_csv(ROOT / "results/xg_strength/ledger.csv", parse_dates=["kickoff_utc"])
names = {"V2x": "V2x XI strength with adjusted xG", "V1x": "V1x adjusted xG strength (replaces raw)",
         "V0": "V0 Dixon-Coles"}
F = {k: led[led["variant"] == v].drop_duplicates("match_id").set_index("match_id") for k, v in names.items()}
mk = {"pre-closing": implied_probabilities(odds, "Avg").set_index("match_id"),
      "closing": implied_probabilities(odds, "AvgC").set_index("match_id")}
base = F["V2x"][["season", "split", "result", "kickoff_utc"]].copy()
ids = base.index[base.index.isin(mk["pre-closing"].dropna().index) & base.index.isin(mk["closing"].dropna().index)]
base = base.loc[ids]
y = base["result"].map({o: i for i, o in enumerate(OUT)}).to_numpy()


def P(df):
    return df.loc[ids, COLS].to_numpy(float)


def picks(p):
    s = np.sort(p, axis=1)
    return p.argmax(axis=1), (s[:, -1] - s[:, -2]) < 1e-9


def ll(p):
    return -np.log(p[np.arange(len(p)), y])


def block(a, b, mask, days):
    return block_bootstrap(a[mask], b[mask], base["kickoff_utc"][mask].reset_index(drop=True), days)


periods = {"2022-26 (1,520)": base["season"] < "2026-27", "2022-24": base["split"].eq("dev"),
           "2024-26": base["split"].eq("test"), "2026-27 GW1-9": base["season"].eq("2026-27")}
res = {"n_matched": len(ids), "comparisons": {}}
tables = []
for model in ("V2x", "V1x"):
    pt = P(F[model])
    for mname, mdf in mk.items():
        pm = P(mdf)
        kt, tt = picks(pt)
        km, tm = picks(pm)
        lt, lm = ll(pt), ll(pm)
        key = f"{model} vs {mname}"
        res["comparisons"][key] = {}
        for pname, per in periods.items():
            per = per.to_numpy()
            ok = per & ~tt & ~tm
            agree = ok & (kt == km)
            dis = ok & (kt != km)
            t_only = dis & (kt == y)
            m_only = dis & (km == y)
            both = dis & (kt != y) & (km != y)
            r = {"n": int(per.sum()), "ties_tabet": int((per & tt).sum()), "ties_market": int((per & tm).sum()),
                 "agree": int(agree.sum()), "agree_rate": float(agree.sum() / ok.sum()),
                 "agree_both_right": float((agree & (kt == y)).sum() / max(agree.sum(), 1)),
                 "disagree": int(dis.sum()), "tabet_only_right": int(t_only.sum()), "market_only_right": int(m_only.sum()),
                 "both_wrong": int(both.sum()),
                 "tabet_pick_draw_share_on_disagree": float((dis & (kt == 1)).sum() / max(dis.sum(), 1)),
                 "market_pick_draw_share_on_disagree": float((dis & (km == 1)).sum() / max(dis.sum(), 1)),
                 "tabet_hit_rate": float((kt == y)[ok].mean()), "market_hit_rate": float((km == y)[ok].mean())}
            for sub, sm in (("all", per), ("agree", agree), ("disagree", dis)):
                if sm.sum() < 2:
                    continue
                d14 = block(lt, lm, sm, 14)
                d28 = block(lt, lm, sm, 28)
                r[f"logloss_{sub}"] = {"n": int(sm.sum()), "tabet": float(lt[sm].mean()), "market": float(lm[sm].mean()),
                                       "diff": d14[0], "block14_ci": [d14[1], d14[2]], "block28_ci": [d28[1], d28[2]]}
            r["tabet_more_prob_on_actual"] = float((pt[np.arange(len(y)), y] > pm[np.arange(len(y)), y])[per].mean())
            res["comparisons"][key][pname] = r
            tables.append({"comparison": key, "period": pname, **{k: v for k, v in r.items() if not isinstance(v, dict)},
                           "ll_tabet": r["logloss_all"]["tabet"], "ll_market": r["logloss_all"]["market"],
                           "ll_diff": r["logloss_all"]["diff"], "ll_ci14": r["logloss_all"]["block14_ci"]})

# ---- sign test on decisive disagreements, and promoted sides (known before kickoff) ----
from scipy.stats import binomtest  # noqa: E402
from tabet.model import promoted_team_seasons  # noqa: E402
prom = promoted_team_seasons(m)
mi = m.set_index("match_id").loc[ids]
first_n = {}
order = pd.concat([mi[["season", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team"}).reset_index(),
                   mi[["season", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team"}).reset_index()]).sort_values("kickoff_utc")
order["n"] = order.groupby(["team", "season"]).cumcount() + 1
order["early_prom"] = [(t, s_) in prom and n <= 10 for t, s_, n in zip(order["team"], order["season"], order["n"])]
early_prom = order.groupby("match_id")["early_prom"].any().reindex(ids).to_numpy()
any_prom = np.array([(h, s_) in prom or (a, s_) in prom for h, a, s_ in zip(mi["home_id"], mi["away_id"], mi["season"])])
res["promoted"] = {}
for key, d in res["comparisons"].items():
    for pname, r in d.items():
        dec = r["tabet_only_right"] + r["market_only_right"]
        r["sign_test_p"] = float(binomtest(r["tabet_only_right"], dec, 0.5).pvalue) if dec else None
for mname, mdf in mk.items():
    pt, pm = P(F["V2x"]), P(mdf)
    lt, lm = ll(pt), ll(pm)
    kt, km = pt.argmax(1), pm.argmax(1)
    per = (base["season"] < "2026-27").to_numpy()
    for sub, sm in (("promoted side in its first 10 matches", early_prom), ("any promoted side", any_prom), ("no promoted side", ~any_prom)):
        mask = per & sm
        d14 = block(lt, lm, mask, 14)
        res["promoted"][f"V2x vs {mname} | {sub}"] = {"n": int(mask.sum()), "share_disagree": float((kt != km)[mask].mean()),
                                                     "diff": d14[0], "block14_ci": [d14[1], d14[2]]}
        print(f"V2x vs {mname:11s} 2022-26 {sub:38s} n={mask.sum():4d} disagree {(kt != km)[mask].mean():.1%} "
              f"log-loss diff {d14[0]:+.4f} [{d14[1]:+.4f}, {d14[2]:+.4f}]")
for key, d in res["comparisons"].items():
    r = d["2022-26 (1,520)"]
    print(f"{key}: decisive disagreements {r['tabet_only_right']} TaBet vs {r['market_only_right']} market, sign test p = {r['sign_test_p']:.4f}")

T = pd.DataFrame(tables)
pd.set_option("display.width", 250)
show = T[["comparison", "period", "n", "agree_rate", "disagree", "tabet_only_right", "market_only_right", "both_wrong",
          "ll_tabet", "ll_market", "ll_diff", "ll_ci14", "tabet_more_prob_on_actual"]]
print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for key, d in res["comparisons"].items():
    r = d["2022-26 (1,520)"]
    print(f"\n{key}, 2022-26: ties TaBet {r['ties_tabet']}, market {r['ties_market']}; agree {r['agree']} "
          f"(both right {r['agree_both_right']:.1%}); draw picks on disagreements: TaBet {r['tabet_pick_draw_share_on_disagree']:.1%}, "
          f"market {r['market_pick_draw_share_on_disagree']:.1%}")
    for sub in ("agree", "disagree"):
        x = r[f"logloss_{sub}"]
        print(f"  log-loss on {sub} (n={x['n']}): TaBet {x['tabet']:.4f} market {x['market']:.4f} diff {x['diff']:+.4f} "
              f"14d {x['block14_ci'][0]:+.4f},{x['block14_ci'][1]:+.4f} 28d {x['block28_ci'][0]:+.4f},{x['block28_ci'][1]:+.4f}")

# ---- largest disagreements in each direction (V2x vs closing), with drivers ----
pt, pm = P(F["V2x"]), P(mk["closing"])
p1, p0 = P(F["V1x"]), P(F["V0"])
kt, _ = picks(pt)
km, _ = picks(pm)
gap = np.abs(pt - pm).sum(axis=1) / 2  # total-variation distance between the two forecasts
rows = pd.read_pickle(ROOT / "results/review_round2/rows.pkl").set_index(["match_id", "side"])
mt = m.set_index("match_id")
app = pd.read_pickle(APP) if APP else None


def absentees(mid, side):
    """Players who started >= 60% of the team's previous league matches *this season* (up to 10) but did
    not start this one (documented from lineups; the reason is not recorded in our data). Before the
    team's 4th match of the season there is no reliable "usual XI", so None is returned."""
    if app is None:
        return None
    r = mt.loc[mid]
    team = r["home_id"] if side == "H" else r["away_id"]
    cutoff = pd.Timestamp(base.loc[mid, "kickoff_utc"])
    hist = known(app, cutoff)
    hist = hist[(hist["team_id"] == team) & (hist["season"] == r["season"])]
    last = hist.drop_duplicates("match_id").sort_values("kickoff_utc").tail(10)["match_id"]
    if len(last) < 3:
        return None
    share = hist[hist["match_id"].isin(last)].groupby("player_code")["started"].sum() / max(len(last), 1)
    regular = share[share >= 0.6].index
    now = app[(app["match_id"] == mid) & (app["team_id"] == team) & (app["started"] == 1)]["player_code"]
    gone = sorted(set(regular) - set(now))
    nm = app.drop_duplicates("player_code", keep="last").set_index("player_code")["player_name"]
    return [_clean(nm.get(c, str(c))) for c in gone]


def _clean(name: str) -> str:
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def fmt(p):
    return " / ".join(f"{100 * v:.0f}%" for v in p)


cases = []
for label, sel in (("market right, TaBet wrong", (km == y) & (kt != y)), ("TaBet right, market wrong", (kt == y) & (km != y))):
    idx = np.where(sel)[0]
    for i in idx[np.argsort(-gap[idx])][:6]:
        mid = ids[i]
        r = mt.loc[mid]
        side_feats = {}
        for sd in ("H", "A"):
            f = rows.loc[(mid, sd)] if (mid, sd) in rows.index else None
            if f is not None:
                side_feats[sd] = {"xg_att": float(f["f_xgadj_att_90"]), "xg_opp_def": float(f["f_xgadj_def_opp_90"]),
                                  "xi_vs_usual": float(np.expm1(f["f_xi_position"])), "opp_backline": float(f["f_backline_opp"])}
        cases.append({
            "group": label, "match_id": mid, "match": f"{r['home_team']} v {r['away_team']}", "season": r["season"],
            "date": str(pd.Timestamp(r["kickoff_utc"]).date()),
            "score": f"{int(r['home_goals'])}-{int(r['away_goals'])}", "result": OUT[y[i]],
            "tabet_V2x": fmt(pt[i]), "market_closing": fmt(pm[i]), "market_pre": fmt(P(mk["pre-closing"])[i]),
            "gap_tv": float(gap[i]), "dixon_coles_V0": fmt(p0[i]), "plus_xg_V1x": fmt(p1[i]),
            "drivers": side_feats, "home_absent_regulars": absentees(mid, "H"), "away_absent_regulars": absentees(mid, "A"),
            "post_kickoff": {"home_reds": int(r["home_reds"]), "away_reds": int(r["away_reds"]),
                             "shots": f"{int(r['home_shots'])}-{int(r['away_shots'])}",
                             "shots_on_target": f"{int(r['home_shots_on_target'])}-{int(r['away_shots_on_target'])}"},
        })
C = pd.DataFrame(cases)
C.to_json(out / "largest_disagreements.json", orient="records", indent=2)
for c in cases:
    print(f"\n[{c['group']}] {c['match']} {c['date']} {c['score']} ({c['result']}) | TaBet {c['tabet_V2x']} | close {c['market_closing']} "
          f"| pre {c['market_pre']} | V0 {c['dixon_coles_V0']} -> V1x {c['plus_xg_V1x']} -> V2x {c['tabet_V2x']}")
    for sd, f in c["drivers"].items():
        print(f"   {sd}: xG attack {f['xg_att']:+.2f}, opp xG defence {f['xg_opp_def']:+.2f}, XI vs usual {f['xi_vs_usual']:+.1%}, opp back line {f['opp_backline']:.2f}")
    print(f"   absent regulars: H {c['home_absent_regulars']} | A {c['away_absent_regulars']}")
    print(f"   post-kickoff: reds {c['post_kickoff']['home_reds']}-{c['post_kickoff']['away_reds']}, shots {c['post_kickoff']['shots']}, SoT {c['post_kickoff']['shots_on_target']}")
T.to_csv(out / "summary.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"\nwrote {out}")
