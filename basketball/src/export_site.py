"""Export NBA forecasts for TaBet's static site.

Writes one JSON snapshot, site/data/basketball.json,
which the NBA page (basketball.html) reads. It holds:

  * reports: for each exported season, every game with the model's pregame
    forecast, the final score and the market's closing line, plus a summary;
  * upcoming: forecasts for the scheduled games of the next few days.

Run from the repo root, after db.py (and fetch_schedule.py for upcoming games):

    python3 basketball/src/export_site.py
    python3 basketball/src/export_site.py --seasons 2024-25 2025-26 --days 7

Commit the regenerated JSON to update the site on its next deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import norm

BASKETBALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASKETBALL_DIR))

from src import backtest, db, predict  # noqa: E402
from src.ingest.fetch_schedule import SCHEDULE_FILE  # noqa: E402
from src.models import situational  # noqa: E402

SITE_DATA = BASKETBALL_DIR.parent / "site" / "data" / "basketball.json"
MODEL = "situational"
SEASONS_EXPORTED = 2    # the newest seasons in the database
UPCOMING_DAYS = 14


def number(value, digits: int):
    """A rounded number, or None where the value is missing."""
    if pd.isna(value):
        return None
    return round(float(value), digits) if digits else int(round(float(value)))


def season_report(table: pd.DataFrame, season: str) -> dict:
    """One season's games and scores. `table` is backtest.build_table()'s result."""
    games = table[table["season"] == season].sort_values(["date", "game_id"])
    priced = games.dropna(subset=["market_home_win_prob"])
    summary = {
        "season": season,
        "games": len(games),
        "scored_games": len(priced),            # games with a market price, scored for model and market alike
        "missing_odds": len(games) - len(priced),
        "model_brier": number(backtest.brier_score(priced["home_win_prob"], priced["home_won"]), 4)
        if len(priced) else None,
        "market_brier": number(backtest.brier_score(priced["market_home_win_prob"], priced["home_won"]), 4)
        if len(priced) else None,
        "model_brier_all": number(backtest.brier_score(games["home_win_prob"], games["home_won"]), 4),
    }
    rows = [{
        "game_id": game.game_id, "game_date": game.date, "game_type": game.game_type,
        "home_team": game.home, "away_team": game.away,
        "home_score": int(game.home_pts), "away_score": int(game.away_pts),
        "p_home": number(game.home_win_prob, 4), "margin": number(game.predicted_margin, 1),
        "market_p_home": number(game.market_home_win_prob, 4), "spread": number(game.spread_close, 1),
    } for game in games.itertuples(index=False)]
    return {"summary": summary, "games": rows}


def upcoming_games(games: pd.DataFrame, schedule: pd.DataFrame, as_of: date, days: int) -> list[dict]:
    """Forecasts for scheduled games from `as_of` through the next `days` days.

    Ratings are the ones after the last game played. Rest and travel follow the schedule, so a team's
    second upcoming game counts its first upcoming game as its previous one.
    """
    played_ids = set(games["game_id"])
    last = (as_of + timedelta(days=days)).isoformat()
    todo = schedule[(~schedule["game_id"].isin(played_ids)) & (schedule["date"] >= as_of.isoformat())
                    & (schedule["date"] <= last)]
    if todo.empty:
        return []

    fitted = predict.fit(games, MODEL)
    columns = ["game_id", "date", "home_team_id", "home", "away_team_id", "away"]
    timeline = situational.add_situation(pd.concat([games[columns], todo[columns]], ignore_index=True))
    timeline = timeline[timeline["game_id"].isin(set(todo["game_id"]))]

    result = []
    for game in timeline.itertuples(index=False):
        home_id, away_id = int(game.home_team_id), int(game.away_team_id)
        adjustment = fitted.adjustment(game.back_to_back, game.long_rest, game.travel_1000km)
        margin = fitted.base.predict(home_id, away_id)[1] + adjustment
        home_pts, away_pts = fitted.base.predict_score(home_id, away_id)
        result.append({
            "game_id": game.game_id, "game_date": game.date, "home_team": game.home, "away_team": game.away,
            "p_home": number(norm.cdf(margin / fitted.base.margin_sd), 4), "margin": number(margin, 1),
            "home_pts": number(home_pts + adjustment / 2, 0), "away_pts": number(away_pts - adjustment / 2, 0),
            "home_rest_days": number(game.home_rest_days, 0), "away_rest_days": number(game.away_rest_days, 0),
        })
    return result


def build_payload(games: pd.DataFrame, odds: pd.DataFrame, schedule: pd.DataFrame | None, as_of: date,
                  seasons: list[str] | None = None, days: int = UPCOMING_DAYS) -> dict:
    """Everything the NBA page shows. Only games before `as_of` count as played."""
    games = games[games["date"] < as_of.isoformat()]
    if games.empty:
        raise ValueError(f"No games before {as_of.isoformat()} in the database.")
    table = backtest.build_table(games, odds, MODEL)

    known = sorted(table["season"].unique())
    seasons = seasons or known[-SEASONS_EXPORTED:]
    unknown = [season for season in seasons if season not in known]
    if unknown:
        raise ValueError(f"No games for season {', '.join(unknown)}. Seasons available: {', '.join(known)}")

    return {
        "as_of": as_of.isoformat(),
        "model": backtest.MODELS[MODEL][0],
        "results_through": str(games["date"].max()),
        "reports": [season_report(table, season) for season in sorted(set(seasons), reverse=True)],
        "upcoming": [] if schedule is None else upcoming_games(games, schedule, as_of, days),
        "has_schedule": schedule is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export NBA forecasts for TaBet's static site.")
    parser.add_argument("--seasons", nargs="+", help=f"seasons to export, e.g. 2025-26 "
                        f"(default: the newest {SEASONS_EXPORTED} in the database)")
    parser.add_argument("--days", type=int, default=UPCOMING_DAYS,
                        help=f"days of upcoming games to forecast (default {UPCOMING_DAYS})")
    parser.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(timezone.utc).date(),
                        help="snapshot date, YYYY-MM-DD (default: today, UTC); games on or after it are not used")
    parser.add_argument("--db", type=Path, default=db.DB_FILE, help="database file built by db.py")
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_FILE, help="schedule from fetch_schedule.py")
    parser.add_argument("--output", type=Path, default=SITE_DATA, help="where to write the JSON")
    args = parser.parse_args(argv)

    try:
        engine = db.connect(args.db, must_exist=True)
        schedule = pd.read_csv(args.schedule, dtype={"game_id": str}) if args.schedule.exists() else None
        payload = build_payload(db.read_games(engine, possessions=True), db.read_odds(engine), schedule,
                                args.as_of, args.seasons, args.days)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    seasons = ", ".join(f"{report['summary']['season']} ({report['summary']['games']} games)"
                        for report in payload["reports"])
    print(f"Wrote {args.output}")
    print(f"  seasons: {seasons}")
    print(f"  upcoming games: {len(payload['upcoming'])}")
    if schedule is None:
        print(f"  {args.schedule} not found, so there are no upcoming games. "
              "Run basketball/src/ingest/fetch_schedule.py first to include them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
