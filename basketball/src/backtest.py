"""Replay past seasons and grade a model against the betting market.

Every game is predicted using only the games before it (see models/),
then the model and the market are scored on the same games:

  * Brier score and log loss for the home win probability (lower is better);
  * margin error: the average miss, in points, between predicted and real margin.

The market's win probability comes from the moneylines with the bookmaker's
margin removed; its margin comes from the closing spread.

It reads games and odds from the database and stores the model's predictions there.

Run from the repo root, after clean.py, fetch_odds.py and db.py:

    python3 basketball/src/backtest.py                      # Elo (v1) against the market
    python3 basketball/src/backtest.py --model efficiency   # efficiency ratings (v2) against the market
    python3 basketball/src/backtest.py --compare            # both models and the market side by side
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
from src.models import elo, ratings  # noqa: E402

SUMMARY_FILE = BASKETBALL_DIR / "data" / "processed" / "backtest_summary.csv"

# name on the command line -> (version stored in the database, function that replays the games)
MODELS = {
    "elo": (db.ELO_VERSION, elo.rate_games),
    "efficiency": (db.EFFICIENCY_VERSION, ratings.rate_games),
}

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


def build_table(games: pd.DataFrame, odds: pd.DataFrame, model: str = "elo") -> pd.DataFrame:
    """One row per game with the model's and the market's predictions side by side.

    Model columns: home_win_prob, predicted_margin (made before each game).
    Market columns: market_home_win_prob, market_margin (empty where a game has no odds).
    """
    rated, _ = MODELS[model][1](games)
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


def compare(tables: dict[str, pd.DataFrame], warmup_seasons: int = WARMUP_SEASONS) -> pd.DataFrame:
    """Every model and the market, scored over all graded games; regular season and playoffs apart.

    `tables` maps a model's name to its build_table() result. All models are scored on the same games.
    """
    rows = []
    for name, table in tables.items():
        summary = summarize(table, warmup_seasons)
        for scores in summary[summary["season"] == "all"].to_dict("records"):
            rows.append({"part": scores["part"], "model": MODELS[name][0], "games": scores["games"],
                         "brier": scores["brier_model"], "logloss": scores["logloss_model"],
                         "margin_err": scores["margin_err_model"]})
            rows.append({"part": scores["part"], "model": "market", "games": scores["games"],
                         "brier": scores["brier_market"], "logloss": scores["logloss_market"],
                         "margin_err": scores["margin_err_market"]})
    comparison = pd.DataFrame(rows, columns=["part", "model", "games", "brier", "logloss", "margin_err"])
    comparison = comparison.drop_duplicates(["part", "model"])
    order = pd.DataFrame({"part_order": (comparison["part"] != "regular"),
                          "market_last": (comparison["model"] == "market")})
    return comparison.loc[order.sort_values(["part_order", "market_last"], kind="stable").index].reset_index(drop=True)


def verdict(comparison: pd.DataFrame, new: str = db.EFFICIENCY_VERSION, old: str = db.ELO_VERSION) -> str:
    """Does the new model replace the old one? Only if it is better on every regular-season measure."""
    regular = comparison[comparison["part"] == "regular"].set_index("model")
    measures = ["brier", "logloss", "margin_err"]
    better = [measure for measure in measures if regular.loc[new, measure] < regular.loc[old, measure]]
    if len(better) == len(measures):
        return f"{new} beats {old} on all three regular-season measures: keep {new}."
    if not better:
        return f"{new} does not beat {old} on any regular-season measure: {old} stays."
    return (f"{new} beats {old} only on {', '.join(better)} in the regular season: "
            f"not a clear win, so {old} stays.")


def describe_comparison(comparison: pd.DataFrame) -> str:
    titles = {"regular": "Regular season", "playoffs": "Playoffs (including play-in)"}
    blocks = []
    for part, rows in comparison.groupby("part", sort=False):
        shown = rows.drop(columns=["part", "games"]).set_index("model")
        shown[["brier", "logloss"]] = shown[["brier", "logloss"]].round(4)
        shown["margin_err"] = shown["margin_err"].round(2)
        blocks.append(f"{titles[part]}, {rows['games'].iloc[0]} games\n{shown.to_string()}")
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a model against the betting market.")
    parser.add_argument("--model", choices=MODELS, default="elo", help="model to grade (default elo)")
    parser.add_argument("--compare", action="store_true", help="grade every model and print them side by side")
    parser.add_argument("--db", type=Path, default=db.DB_FILE, help="database file built by db.py")
    parser.add_argument("--output", type=Path, help="where to write the summary")
    parser.add_argument("--warmup-seasons", type=int, default=WARMUP_SEASONS,
                        help=f"seasons replayed but not graded (default {WARMUP_SEASONS})")
    args = parser.parse_args(argv)

    try:
        engine = db.connect(args.db, must_exist=True)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    games = db.read_games(engine, possessions=True)
    odds = db.read_odds(engine)

    names = list(MODELS) if args.compare else [args.model]
    tables = {}
    for name in names:
        tables[name] = build_table(games, odds, name)
        db.save_predictions(engine, tables[name], MODELS[name][0])

    if args.compare:
        result = compare(tables, args.warmup_seasons)
        default_name = "backtest_comparison.csv"
    else:
        result = summarize(tables[args.model], args.warmup_seasons)
        default_name = "backtest_summary.csv" if args.model == "elo" else f"backtest_summary_{args.model}.csv"
    if result.empty:
        print("Error: no games left to grade. Check that the odds table covers the seasons in the games table.",
              file=sys.stderr)
        return 1

    output = args.output or SUMMARY_FILE.with_name(default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    warmup = sorted(games["season"].unique())[:args.warmup_seasons]
    if warmup:
        print(f"Warm-up (replayed, not graded): {', '.join(warmup)}")
    print("Lower is better in every column. Brier and log loss grade the win probability;")
    print("margin_err is the average miss in points.\n")
    if args.compare:
        print(describe_comparison(result))
        print(f"\nVerdict: {verdict(result)}")
    else:
        print(f"Model: {MODELS[args.model][0]}\n")
        print(describe(result))
    print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
