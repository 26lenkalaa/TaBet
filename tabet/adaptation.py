"""Faster adaptation of the scoring environment (review item #4).

Dixon-Coles uses one decay for everything. These features measure how far the
*recent* league scoring level and home advantage depart from their long-run
levels (both from results known before the cutoff), so stage 2 can let them
move faster than team strengths:

    f_lg_<hl>    log(recent goals per team-match) - log(long-run goals per team-match)
                 (the same value for both sides of a match)
    f_home_<hl>  +/- 0.5 * [log(recent home/away goal ratio) - log(long-run ratio)]
                 (+ for the home side, - for the away side)

"recent" uses half-life hl days; "long-run" uses the Dixon-Coles half-life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Hyperparams
from .store import RESULT_LAG

LN2 = np.log(2)


def _levels(hist: pd.DataFrame, cutoff: pd.Timestamp, hl: float) -> tuple[float, float]:
    age = (cutoff - hist["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
    w = np.exp(-LN2 * age / hl)
    hg, ag = hist["home_goals"].to_numpy(float), hist["away_goals"].to_numpy(float)
    goals = np.sum(w * (hg + ag)) / (2 * w.sum())
    ratio = np.sum(w * hg) / max(np.sum(w * ag), 1e-9)
    return float(np.log(goals)), float(np.log(ratio))


def add_scoring_environment(rows: pd.DataFrame, matches: pd.DataFrame, short_half_lives: tuple[float, ...],
                            long_half_life: float | None = None) -> pd.DataFrame:
    long_hl = long_half_life or Hyperparams().half_life_days
    done = matches[matches["status"] == "completed"]
    out = rows.copy()
    sign = np.where(out["side"] == "H", 1.0, -1.0)
    for hl in short_half_lives:
        lg, hm = f"f_lg_{int(hl)}", f"f_home_{int(hl)}"
        out[lg], out[hm] = 0.0, 0.0
        for cutoff, grp in out.groupby("cutoff_utc"):
            hist = done[done["kickoff_utc"] + RESULT_LAG <= cutoff]
            if len(hist) < 50:
                continue
            g_s, r_s = _levels(hist, cutoff, hl)
            g_l, r_l = _levels(hist, cutoff, long_hl)
            out.loc[grp.index, lg] = g_s - g_l
            out.loc[grp.index, hm] = 0.5 * (r_s - r_l) * sign[out.index.get_indexer(grp.index)]
    return out
