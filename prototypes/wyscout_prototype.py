"""Wyscout 2017-18 prototype: do team-style matchups or player similarity help?

Data: Pappalardo et al. (2019), "A public data set of spatio-temporal match events
in soccer competitions", Scientific Data. Figshare, CC BY 4.0. One season
(2017-18) of the top five leagues. It's used here only as an open prototype
bed. It does not overlap our 2021-27 Premier League period and doesn't replace
recent player histories.

  A. Team style. A per-league Dixon-Coles walk-forward model is compared with
     style main effects and attack-style x defence-style interactions, all
     added as a penalised Poisson log-rate adjustment, plus England-only vs
     pooled coefficients. Every variant is scored on the same fixtures
     (gameweek 20 onward).
  B. Player similarity. Second-half per-90 output is predicted from first-half
     data: role mean, own rate shrunk to the role mean, or own rate shrunk to
     the mean of similar players.

Usage: python3 -I prototypes/wyscout_prototype.py [DATA_DIR]   (default data/raw/wyscout)
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import poisson  # noqa: E402

from tabet.backtest import scores  # noqa: E402
from tabet.model import MAX_GOALS, DixonColes, Hyperparams  # noqa: E402
from tabet.store import RESULT_LAG  # noqa: E402

LEAGUES = ["England", "Spain", "Germany", "Italy", "France"]
FIGSHARE = {"competitions.json": 15073685, "matches.zip": 14464622, "events.zip": 14464685,
            "players.json": 15073721, "teams.json": 15073697}
EVAL_FROM_GW = 20      # scored fixtures: gameweek >= 20 (identical for every variant)
STAGE2_FROM_GW = 6     # stage-2 training rows: gameweek >= 6 (DC has some data by then)
MIN_STAGE2 = 100       # same rule for every variant: < 100 training matches -> DC alone
ALPHA = 10.0           # fixed L2 on stage-2 coefficients (not tuned on the scored fixtures)
STYLE_SHRINK = 3       # prior matches' worth of shrinkage toward league mean style
N_BOOT = 2000
RNG = np.random.default_rng(0)
ATT = ["pass_share", "pass_len", "cross_rate", "shot_dist", "tempo"]
DEF = ["def_height", "ppda"]
STYLE = ATT + DEF


# ---------------------------------------------------------------- data ----

def ensure_data(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, fid in FIGSHARE.items():
        if not (d / name).exists():
            print(f"downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{fid}", d / name)
    for z in ("matches", "events"):
        if not (d / z).is_dir():
            with zipfile.ZipFile(d / f"{z}.zip") as f:
                f.extractall(d / z)


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def load_matches(d: Path, league: str, caveats: dict):
    rows, mins = [], []
    raw = json.load(open(d / "matches" / f"matches_{league}.json"))
    no_form = 0
    for m in raw:
        sides = {v["side"]: v for v in m["teamsData"].values()}
        h, a = sides["home"], sides["away"]
        rows.append(dict(match_id=m["wyId"], league=league, season="2017-18", gameweek=m["gameweek"],
                         kickoff_utc=pd.Timestamp(m["dateutc"], tz="UTC"), status="completed",
                         home_id=str(h["teamId"]), away_id=str(a["teamId"]),
                         home_goals=int(h["score"]), away_goals=int(a["score"])))
        for t in (h, a):
            if not t["hasFormation"]:
                no_form += 1
                continue
            f = t["formation"]
            start = {p["playerId"]: 0 for p in f["lineup"]}
            end = {pid: 90 for pid in start}
            subs = f["substitutions"] if isinstance(f["substitutions"], list) else []
            for s in sorted(subs, key=lambda s: s["minute"]):
                mi = min(_int(s["minute"]), 90)
                if s["playerOut"] in end:
                    end[s["playerOut"]] = min(end[s["playerOut"]], mi)
                start[s["playerIn"]] = mi
                end.setdefault(s["playerIn"], 90)
            for p in f["lineup"] + (f["bench"] if isinstance(f["bench"], list) else []):
                r = _int(p.get("redCards"))
                if r > 0 and p["playerId"] in end:
                    end[p["playerId"]] = min(end[p["playerId"]], min(r, 90))
            for pid in start:
                mins.append((m["wyId"], str(t["teamId"]), pid, max(0, end[pid] - start[pid]), m["gameweek"]))
    caveats[f"{league}: matches"] = len(raw)
    caveats[f"{league}: team-sides without formation"] = no_form
    return pd.DataFrame(rows), pd.DataFrame(mins, columns=["match_id", "team_id", "player_id", "minutes", "gameweek"])


def load_events(d: Path, league: str, caveats: dict) -> pd.DataFrame:
    raw = json.load(open(d / "events" / f"events_{league}.json"))
    rows = []
    for e in raw:
        pos = e["positions"] or []
        x0, y0 = (pos[0]["x"], pos[0]["y"]) if len(pos) > 0 else (np.nan, np.nan)
        x1, y1 = (pos[1]["x"], pos[1]["y"]) if len(pos) > 1 else (np.nan, np.nan)
        tags = {t["id"] for t in e["tags"]}
        rows.append((e["matchId"], str(e["teamId"]), e["playerId"], e["eventName"], e["subEventName"],
                     e["matchPeriod"], e["eventSec"], x0, y0, x1, y1,
                     101 in tags, 102 in tags, 301 in tags, 302 in tags, 1401 in tags))
    E = pd.DataFrame(rows, columns=["match_id", "team_id", "player_id", "ev", "sub", "period", "sec",
                                    "x0", "y0", "x1", "y1", "goal", "own_goal", "assist", "key_pass", "intercept"])
    del raw
    caveats[f"{league}: events"] = len(E)
    caveats[f"{league}: events with playerId=0"] = f"{(E.player_id == 0).mean():.2%}"
    caveats[f"{league}: events with no position"] = f"{E.x0.isna().mean():.2%}"
    caveats[f"{league}: passes with no end position"] = f"{E.loc[E.ev == 'Pass', 'x1'].isna().mean():.2%}"
    caveats[f"{league}: events with empty subEventName"] = f"{(E['sub'] == '').mean():.2%}"
    return E


def flags(E: pd.DataFrame) -> pd.DataFrame:
    E = E.copy()
    E["is_pass"] = E.ev == "Pass"
    E["is_cross"] = E["sub"] == "Cross"
    E["is_shot"] = (E.ev == "Shot") | E["sub"].isin(["Free kick shot", "Penalty"])
    E["is_def"] = ((E.ev == "Duel") & (E["sub"] == "Ground defending duel")) | E.intercept | (E.ev == "Foul")
    E["pass_len"] = np.where(E.is_pass, np.hypot((E.x1 - E.x0) * 1.05, (E.y1 - E.y0) * 0.68), np.nan)
    E["shot_dist"] = np.where(E.is_shot, np.hypot((100 - E.x0) * 1.05, (50 - E.y0) * 0.68), np.nan)
    return E


def match_styles(E: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): that match's style numbers (known only after it)."""
    E = E.sort_values(["match_id", "period", "sec"], kind="stable")
    same = (E.match_id == E.match_id.shift()) & (E.period == E.period.shift()) & (E.team_id == E.team_id.shift())
    gap = (E.sec - E.sec.shift()).where(same & E.is_pass & E.is_pass.shift(fill_value=False))
    E = E.assign(pass_gap=gap.where(gap < 15),
                 def_x=E.x0.where(E.is_def),
                 high_def=E.is_def & (E.x0 >= 40),
                 low_pass=E.is_pass & (E.x0 < 60))
    g = E.groupby(["match_id", "team_id"])
    s = pd.DataFrame({"passes": g.is_pass.sum(), "crosses": g.is_cross.sum(), "pass_len": g.pass_len.mean(),
                      "shot_dist": g.shot_dist.mean(), "tempo": g.pass_gap.mean(), "def_height": g.def_x.mean(),
                      "high_def": g.high_def.sum(), "low_pass": g.low_pass.sum()}).reset_index()
    opp = pd.concat([M[["match_id", "home_id", "away_id"]].rename(columns={"home_id": "team_id", "away_id": "opp_id"}),
                     M[["match_id", "away_id", "home_id"]].rename(columns={"away_id": "team_id", "home_id": "opp_id"})])
    s = s.merge(opp, on=["match_id", "team_id"])
    o = s[["match_id", "team_id", "passes", "low_pass"]].rename(
        columns={"team_id": "opp_id", "passes": "opp_passes", "low_pass": "opp_low_pass"})
    s = s.merge(o, on=["match_id", "opp_id"])
    s["pass_share"] = s.passes / (s.passes + s.opp_passes)
    s["cross_rate"] = s.crosses / s.passes
    s["ppda"] = s.opp_low_pass / s.high_def.replace(0, np.nan)
    return s[["match_id", "team_id", *STYLE]]


# ---------------------------------------------------- experiment A ----

def dc_probs(lh: np.ndarray, la: np.ndarray, rho: np.ndarray) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    out = np.empty((len(lh), 3))
    for i, (h, a, r) in enumerate(zip(lh, la, rho)):
        m = np.outer(poisson.pmf(g, h), poisson.pmf(g, a))
        m[0, 0] *= 1 - h * a * r
        m[0, 1] *= 1 + h * r
        m[1, 0] *= 1 + a * r
        m[1, 1] *= 1 - r
        m /= m.sum()
        out[i] = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    return out


def walk_forward_league(M: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Per matchday: DC fit on known results, plus point-in-time team style z-scores."""
    M = M.sort_values("kickoff_utc").reset_index(drop=True)
    teams = sorted(set(M.home_id) | set(M.away_id))
    Sx = S.set_index(["match_id", "team_id"])
    sums = pd.DataFrame(0.0, index=teams, columns=STYLE)
    cnts = pd.DataFrame(0.0, index=teams, columns=STYLE)
    added: set[int] = set()
    out = []
    for _, grp in M.groupby(M.kickoff_utc.dt.date, sort=True):
        cutoff = grp.kickoff_utc.min()
        known = M[M.kickoff_utc + RESULT_LAG <= cutoff]
        if len(known) == 0:
            continue
        for mid, h, a in known[["match_id", "home_id", "away_id"]].itertuples(index=False):
            if mid in added:
                continue
            added.add(mid)
            for t in (h, a):
                v = Sx.loc[(mid, t), STYLE].astype(float)
                ok = v.notna()
                sums.loc[t, ok] += v[ok]
                cnts.loc[t, ok] += 1
        prior = sums.sum() / cnts.sum()
        val = (sums + STYLE_SHRINK * prior) / (cnts + STYLE_SHRINK)
        z = ((val - val.mean()) / val.std(ddof=0).replace(0, np.nan)).fillna(0.0)
        dc = DixonColes(Hyperparams()).fit(known, cutoff, set())
        rows = grp.copy()
        lam = [dc.expected_goals(h, a) for h, a in zip(grp.home_id, grp.away_id)]
        rows["lh"], rows["la"] = [l[0] for l in lam], [l[1] for l in lam]
        rows["rho"] = dc.rho
        rows["cutoff"] = cutoff
        for side in ("home", "away"):
            zz = z.loc[grp[f"{side}_id"]].to_numpy()
            for j, c in enumerate(STYLE):
                rows[f"{side}_{c}"] = zz[:, j]
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def design(R: pd.DataFrame, interactions: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack home-scoring and away-scoring rows. Returns X_home, X_away, offsets, targets."""
    def side(att: str, dfn: str) -> np.ndarray:
        A = R[[f"{att}_{c}" for c in STYLE]].to_numpy()
        D = R[[f"{dfn}_{c}" for c in STYLE]].to_numpy()
        cols = [A, D]
        if interactions:
            cols.append(np.stack([R[f"{att}_{a}"] * R[f"{dfn}_{d}"] for a in ATT for d in DEF], axis=1))
        return np.hstack(cols)
    return side("home", "away"), side("away", "home"), np.log(R[["lh", "la"]].to_numpy()), R[["home_goals", "away_goals"]].to_numpy(float)


def fit_stage2(R: pd.DataFrame, interactions: bool) -> np.ndarray:
    Xh, Xa, off, y = design(R, interactions)
    X = np.vstack([Xh, Xa])
    o = np.concatenate([off[:, 0], off[:, 1]])
    yy = np.concatenate([y[:, 0], y[:, 1]])

    def f(b):
        eta = o + X @ b
        mu = np.exp(eta)
        return -(yy * eta - mu).sum() + ALPHA * b @ b, -X.T @ (yy - mu) + 2 * ALPHA * b
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def experiment_a(W: pd.DataFrame) -> pd.DataFrame:
    W = W.sort_values("kickoff_utc").reset_index(drop=True)
    train_pool = W[W.gameweek >= STAGE2_FROM_GW]
    ev = W[W.gameweek >= EVAL_FROM_GW].copy()
    base = dc_probs(ev.lh.to_numpy(), ev.la.to_numpy(), ev.rho.to_numpy())
    preds = {"V0 DC only": base}
    variants = {"V1 + style main (pooled)": (False, None), "V2 + style x style (pooled)": (True, None),
                "V2 + style x style (England-only coefs)": (True, "England")}
    for name, (inter, only) in variants.items():
        P = np.full((len(ev), 3), np.nan)
        for (lg, cutoff), grp in ev.groupby(["league", "cutoff"]):
            if only and lg != only:
                continue
            tr = train_pool[train_pool.kickoff_utc + RESULT_LAG <= cutoff]
            if only:
                tr = tr[tr.league == only]
            pos = ev.index.get_indexer(grp.index)
            if len(tr) < MIN_STAGE2:
                P[pos] = base[pos]
                continue
            b = fit_stage2(tr, inter)
            Xh, Xa, off, _ = design(grp, inter)
            P[pos] = dc_probs(np.exp(off[:, 0] + Xh @ b), np.exp(off[:, 1] + Xa @ b), grp.rho.to_numpy())
        preds[name] = P

    res = ev.home_goals.sub(ev.away_goals).map(lambda d: "H" if d > 0 else ("A" if d < 0 else "D")).reset_index(drop=True)
    y = res.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    rows = []
    for scope, mask in (("all 5 leagues", np.ones(len(ev), bool)), ("England", (ev.league == "England").to_numpy())):
        ll0 = -np.log(base[mask][np.arange(mask.sum()), y[mask]])
        for name, P in preds.items():
            if np.isnan(P[mask]).any():
                continue
            s = scores(P[mask], res[mask].reset_index(drop=True))
            ll = -np.log(np.clip(P[mask][np.arange(mask.sum()), y[mask]], 1e-15, 1))
            d = ll - ll0
            boot = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            rows.append({"fixtures": scope, "variant": name, "n": s["n"], "log_loss": s["log_loss"],
                         "brier": s["brier"], "rps": s["rps"], "d_logloss_vs_V0": d.mean(),
                         "ci95_lo": np.percentile(boot, 2.5), "ci95_hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------- experiment B ----

def player_halves(E: pd.DataFrame, MIN: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    E = E[E.player_id != 0].merge(M[["match_id", "gameweek", "league"]], on="match_id")
    E = E.assign(half=np.where(E.gameweek <= 19, 1, 2), duel=E.ev == "Duel", accel=E["sub"] == "Acceleration",
                 goal_s=E.is_shot & E.goal & ~E.own_goal)
    g = E.groupby(["player_id", "half"])
    c = pd.DataFrame({"shots": g.is_shot.sum(), "key_passes": g.key_pass.sum(), "goals": g.goal_s.sum(),
                      "assists": g.assist.sum(), "passes": g.is_pass.sum(), "crosses": g.is_cross.sum(),
                      "duels": g.duel.sum(), "def_actions": g.is_def.sum(), "accels": g.accel.sum(),
                      "mean_x": g.x0.mean()})
    MIN = MIN.assign(half=np.where(MIN.gameweek <= 19, 1, 2)).merge(M[["match_id", "league"]], on="match_id")
    mm = MIN.groupby(["player_id", "half"]).minutes.sum()
    lg = MIN[MIN.half == 1].groupby(["player_id", "league"]).minutes.sum().reset_index() \
        .sort_values("minutes").drop_duplicates("player_id", keep="last").set_index("player_id").league
    out = c.join(mm, how="outer").fillna({k: 0 for k in c.columns if k != "mean_x"}).fillna({"minutes": 0})
    out = out.unstack("half")
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["league"] = lg
    out["out1_1"] = out.shots_1 + out.key_passes_1
    out["out1_2"] = out.shots_2 + out.key_passes_2
    out["ga_1"] = out.goals_1 + out.assists_1
    out["ga_2"] = out.goals_2 + out.assists_2
    return out


def eb_k(rate_counts: pd.Series, n90: pd.Series) -> tuple[float, float]:
    """Method-of-moments empirical-Bayes prior (mean, k in 90-minute units)."""
    mu = rate_counts.sum() / n90.sum()
    r = rate_counts / n90
    var_w = (n90 * (r - mu) ** 2).sum() / n90.sum()
    tau2 = max(var_w - len(r) * mu / n90.sum(), 1e-6)
    return mu, mu / tau2


def experiment_b(P: pd.DataFrame, players: pd.DataFrame, caveats: dict) -> pd.DataFrame:
    P = P.join(players, how="left")
    P = P[P.role.notna() & (P.role != "Goalkeeper")]
    m1, m2 = P.minutes_1.fillna(0), P.minutes_2.fillna(0)
    pop = P[(m1 >= 90) & (m1 <= 600) & (m2 >= 450)].copy()
    pool = P[m1 >= 900].copy()
    caveats["B: target players (90-600 first-half min, >=450 second-half)"] = len(pop)
    caveats["B: neighbour pool (>=900 first-half min)"] = len(pool)

    prof = ["passes", "crosses", "duels", "def_actions", "accels", "shots", "key_passes"]

    def feats(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame({c: df[f"{c}_1"] / (df.minutes_1 / 90) for c in prof})
        f["mean_x"] = df.mean_x_1
        f["height"] = df.height
        return f
    fp, ft = feats(pool), feats(pop)
    mu_, sd_ = fp.mean(), fp.std().replace(0, 1)
    fp, ft = ((fp - mu_) / sd_).fillna(0), ((ft - mu_) / sd_).fillna(0)
    for r in ("Defender", "Midfielder", "Forward"):
        fp[f"role_{r}"], ft[f"role_{r}"] = 3.0 * (pool.role == r), 3.0 * (pop.role == r)
    for fo in ("right", "left", "both"):
        fp[f"foot_{fo}"], ft[f"foot_{fo}"] = 1.0 * (pool.foot == fo), 1.0 * (pop.foot == fo)

    rows = []
    for metric, label in (("out1", "shots + key passes per 90"), ("ga", "goals + assists per 90")):
        base = P[m1 >= 90]
        prior = {r: eb_k(g[f"{metric}_1"], g.minutes_1 / 90) for r, g in base.groupby("role")}
        mu = pop.role.map(lambda r: prior[r][0]).to_numpy()
        k = pop.role.map(lambda r: prior[r][1]).to_numpy()
        c1, n1 = pop[f"{metric}_1"].to_numpy(), (pop.minutes_1 / 90).to_numpy()
        target, w = (pop[f"{metric}_2"] / (pop.minutes_2 / 90)).to_numpy(), pop.minutes_2.to_numpy()
        pool_rate = (pool[f"{metric}_1"] / (pool.minutes_1 / 90)).to_numpy()
        dist = ((ft.to_numpy()[:, None, :] - fp.to_numpy()[None, :, :]) ** 2).sum(axis=2)
        same = pop.league.to_numpy()[:, None] == pool.league.to_numpy()[None, :]
        knn_all = np.array([pool_rate[np.argsort(d)[:20]].mean() for d in dist])
        knn_lg = np.array([pool_rate[np.argsort(np.where(s, d, np.inf))[:20]].mean() for d, s in zip(dist, same)])
        preds = {"P0 role mean": mu,
                 "P1 own rate shrunk to role mean": (c1 + k * mu) / (n1 + k),
                 "P2 own rate shrunk to 20 similar players (all leagues)": (c1 + k * knn_all) / (n1 + k),
                 "P3 own rate shrunk to 20 similar players (same league)": (c1 + k * knn_lg) / (n1 + k)}
        e_ref = (preds["P1 own rate shrunk to role mean"] - target) ** 2
        for name, p in preds.items():
            e = (p - target) ** 2
            idx = [RNG.integers(0, len(e), len(e)) for _ in range(N_BOOT)]
            boot = [np.average(e[i], weights=w[i]) - np.average(e_ref[i], weights=w[i]) for i in idx]
            rows.append({"metric": label, "method": name, "n": len(e), "wMSE": np.average(e, weights=w),
                         "wMAE": np.average(np.abs(p - target), weights=w),
                         "d_wMSE_vs_P1": np.average(e, weights=w) - np.average(e_ref, weights=w),
                         "ci95_lo": np.percentile(boot, 2.5), "ci95_hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- main ----

def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "wyscout"
    ensure_data(d)
    caveats: dict = {}
    pl = pd.DataFrame(json.load(open(d / "players.json")))
    caveats["players: height == 0"] = f"{(pl.height == 0).mean():.2%}"
    caveats["players: foot missing ('' or null)"] = f"{pl.foot.isin(['', 'null']).mean() + pl.foot.isna().mean():.2%}"
    players = pd.DataFrame({"role": pl.role.map(lambda r: r.get("name") if isinstance(r, dict) else None).to_numpy(),
                            "height": pl.height.replace(0, np.nan).to_numpy(), "foot": pl.foot.to_numpy()},
                           index=pd.Index(pl.wyId.to_numpy(), name="player_id"))

    walks, halves = [], []
    for lg in LEAGUES:
        M, MIN = load_matches(d, lg, caveats)
        E = flags(load_events(d, lg, caveats))
        walks.append(walk_forward_league(M, match_styles(E, M)))
        halves.append(player_halves(E, MIN, M))
        del E
    W = pd.concat(walks, ignore_index=True)
    H = pd.concat(halves)
    # A player who moved between two of these leagues mid-season has a row in each league.
    # Sum their counts, average their mean_x, and keep the league with more first-half minutes.
    caveats["B: players appearing in >1 league"] = int(H.index.duplicated().sum())
    H = H.sort_values("minutes_1", ascending=False)
    agg = {c: ("mean" if c.startswith("mean_x") else "first" if c == "league" else "sum") for c in H.columns}
    H = H.groupby(level=0).agg(agg)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("== Data caveats")
    for k, v in caveats.items():
        print(f"  {k}: {v}")
    print("\n== Experiment A: team style (Poisson log-rate adjustment on per-league Dixon-Coles), "
          f"fixtures gameweek >= {EVAL_FROM_GW}")
    print(experiment_a(W).round(4).to_string(index=False))
    print("\n== Experiment B: predict second-half per-90 from first-half data (minutes-weighted errors)")
    print(experiment_b(H, players, caveats).round(4).to_string(index=False))
    print("\n  " + "; ".join(f"{k}: {v}" for k, v in caveats.items() if k.startswith("B:")))


if __name__ == "__main__":
    main()
