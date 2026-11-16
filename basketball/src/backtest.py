"""Replay past seasons and grade the Elo model against the betting market.

Every game is predicted using only the games before it (see models/elo.py),
then the model and the market are scored on the same games:

  * Brier score and log loss for the home win probability (lower is better);
  * margin error: the average miss, in points, between predicted and real margin.

The market's win probability comes from the moneylines with the bookmaker's
margin removed; its margin comes from the closing spread.

It reads games and odds from the database and stores the model's predictions there.

Run from the repo root, after clean.py, fetch_odds.py and db.py:

    python3 basketball/src/backtest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASKETBALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASKETBALL_DIR))

from src import db, market  # noqa: E402
from src.models.elo import Elo, rate_games  # noqa: E402

SUMMARY_FILE = BASKETBALL_DIR / "data" / "processed" / "backtest_summary.csv"

WARMUP_SEASONS = 1   # every team starts at 1500, so the first season only teaches the ratings
# How far real margins land from the closing spread, in points (measured by fetch_odds.py on
# 2018-19 to 2025-26). Only used to turn a spread into a win probability when a game has no moneyline.
MARGIN_SD = 13.4
ODDS_COLUMNS = ["spread_close", "total_close", "home_ml", "away_ml"]
METRICS = ["brier_model", "brier_market", "logloss_model", "logloss_market",
           "margin_err_model", "margin_err_market"]


def brier_score(prob: pd.Series, home_won: pd.Series) -> float:
    """Average squared gap between the probability and what happened (1 = home win, 0 = home loss)."""
    return float(((prob - home_won) ** 2).mean())


def log_loss(prob: pd.Series, home_won: pd.Series) -> float:
    """Average surprise at the result; punishes confident wrong predictions hard."""
    prob = np.clip(prob.astype(float), 1e-12, 1 - 1e-12)
    return float(-(home_won * np.log(prob) + (1 - home_won) * np.log(1 - prob)).mean())


def margin_error(predicted: pd.Series, actual: pd.Series) -> float:
    """Average miss in points, ignoring direction."""
    return float((predicted - actual).abs().mean())


def build_table(games: pd.DataFrame, odds: pd.DataFrame, model: Elo | None = None) -> pd.DataFrame:
    """One row per game with the model's and the market's predictions side by side.

    Model columns: home_win_prob, predicted_margin (made before each game).
    Market columns: market_home_win_prob, market_margin (empty where a game has no odds).
    """
    rated, _ = rate_games(games, model)
    keep = ["game_id"] + [column for column in ODDS_COLUMNS if column in odds.columns]
    table = rated.merge(odds[keep].drop_duplicates("game_id"), on="game_id", how="left")

    table = market.add_market_predictions(table, MARGIN_SD)

    table["home_won"] = (table["home_pts"] > table["away_pts"]).astype(int)
    table["margin"] = table["home_pts"] - table["away_pts"]
    table["part"] = np.where(table["game_type"] == "regular", "regular", "playoffs")
    return table


def grade(table: pd.DataFrame) -> dict[str, float]:
    """Score model and market on the same games: those where the market has a prediction."""
    priced = table.dropna(subset=["market_home_win_prob"])
    lined = table.dropna(subset=["market_margin"])
    return {
        "games": len(priced),
        "brier_model": brier_score(priced["home_win_prob"], priced["home_won"]),
        "brier_market": brier_score(priced["market_home_win_prob"], priced["home_won"]),
        "logloss_model": log_loss(priced["home_win_prob"], priced["home_won"]),
        "logloss_market": log_loss(priced["market_home_win_prob"], priced["home_won"]),
        "margin_err_model": margin_error(lined["predicted_margin"], lined["margin"]),
        "margin_err_market": margin_error(lined["market_margin"], lined["margin"]),
    }


def summarize(table: pd.DataFrame, warmup_seasons: int = WARMUP_SEASONS) -> pd.DataFrame:
    """Scores per season, regular season and playoffs apart, with an 'all' row for each.

    The first `warmup_seasons` seasons are replayed (so ratings learn from them) but not graded.
    Seasons where no game has odds are left out.
    """
    seasons = sorted(table["season"].unique())
    graded = table[table["season"].isin(seasons[warmup_seasons:])]
    graded = graded[graded["market_home_win_prob"].notna() | graded["market_margin"].notna()]

    rows = []
    for part in ("regular", "playoffs"):
        games = graded[graded["part"] == part]
        if games.empty:
            continue
        for season, season_games in games.groupby("season"):
            rows.append({"part": part, "season": season, **grade(season_games)})
        rows.append({"part": part, "season": "all", **grade(games)})
    return pd.DataFrame(rows, columns=["part", "season", "games"] + METRICS)


def describe(summary: pd.DataFrame) -> str:
    titles = {"regular": "Regular season", "playoffs": "Playoffs (including play-in)"}
    blocks = []
    for part, rows in summary.groupby("part", sort=False):
        shown = rows.drop(columns="part").set_index("season")
        shown[METRICS[:4]] = shown[METRICS[:4]].round(4)
        shown[METRICS[4:]] = shown[METRICS[4:]].round(2)
        blocks.append(f"{titles[part]}\n{shown.to_string()}")
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the Elo model against the betting market.")
    parser.add_argument("--db", type=Path, default=db.DB_FILE, help="database file built by db.py")
    parser.add_argument("--output", type=Path, default=SUMMARY_FILE, help="where to write the summary")
    parser.add_argument("--warmup-seasons", type=int, default=WARMUP_SEASONS,
                        help=f"seasons replayed but not graded (default {WARMUP_SEASONS})")
    args = parser.parse_args(argv)

    try:
        engine = db.connect(args.db, must_exist=True)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    games = db.read_games(engine)
    odds = db.read_odds(engine)

    table = build_table(games, odds)
    db.save_predictions(engine, table, db.ELO_VERSION)
    summary = summarize(table, args.warmup_seasons)
    if summary.empty:
        print("Error: no games left to grade. Check that the odds table covers the seasons in the games table.",
              file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    warmup = sorted(games["season"].unique())[:args.warmup_seasons]
    if warmup:
        print(f"Warm-up (replayed, not graded): {', '.join(warmup)}")
    print("Lower is better in every column. Brier and log loss grade the win probability;")
    print("margin_err is the average miss in points.\n")
    print(describe(summary))
    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
