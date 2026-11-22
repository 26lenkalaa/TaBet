"""Export NFL game forecasts for TaBet's static site."""

import argparse
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

from backtest import evaluate
from data import load_games, team_code
from model import forecast

ROOT = Path(__file__).resolve().parent.parent
SITE_DATA = ROOT / "site" / "data" / "football.json"


def build_payload(source: Path, seasons: list[int], as_of: date) -> dict[str, object]:
    games = [game for game in load_games(source) if game.game_date < as_of]
    reports: list[dict[str, object]] = []
    for season in sorted(set(seasons), reverse=True):
        rows, summary = evaluate(games, season)
        reports.append({"summary": summary, "games": rows})

    upcoming: list[dict[str, object]] = []
    with source.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["game_type"] != "REG" or row["home_score"] or row["away_score"]:
                continue
            day = date.fromisoformat(row["gameday"])
            season = int(row["season"])
            if day < as_of or season != max(seasons):
                continue
            prediction = forecast(
                games, team_code(row["home_team"]), team_code(row["away_team"]),
                day, season, row["location"] == "Neutral",
            )
            upcoming.append({"game_id": row["game_id"], **prediction})
    upcoming.sort(key=lambda game: str(game["game_date"]))
    return {"as_of": as_of.isoformat(), "reports": reports, "upcoming": upcoming}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", nargs="+", type=int)
    parser.add_argument("--data", type=Path, default=ROOT / "Football" / "res" / "games.csv")
    parser.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--output", type=Path, default=SITE_DATA)
    args = parser.parse_args()
    try:
        payload = build_payload(args.data, args.seasons, args.as_of)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
