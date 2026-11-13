"""Read completed regular-season games from the nflverse schedule CSV."""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# Keep franchise ratings across relocations.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def team_code(value: str) -> str:
    code = value.strip().upper()
    return TEAM_ALIASES.get(code, code)


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    game_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    neutral: bool
    home_moneyline: str = ""
    away_moneyline: str = ""

    @property
    def home_result(self) -> float:
        if self.home_score == self.away_score:
            return 0.5
        return float(self.home_score > self.away_score)


def load_games(path: Path) -> list[Game]:
    games: list[Game] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "game_id", "season", "game_type", "gameday", "home_team",
            "away_team", "home_score", "away_score", "location",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            if row["game_type"] != "REG":
                continue
            if row["home_score"] == "" and row["away_score"] == "":
                continue  # The schedule also includes future games.
            try:
                game = Game(
                    row["game_id"], int(row["season"]), date.fromisoformat(row["gameday"]),
                    team_code(row["home_team"]), team_code(row["away_team"]),
                    int(row["home_score"]), int(row["away_score"]),
                    row["location"] == "Neutral",
                    row.get("home_moneyline") or "", row.get("away_moneyline") or "",
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid game on CSV line {reader.line_num}: {error}") from error
            if (
                not all((game.game_id, game.home_team, game.away_team))
                or game.home_team == game.away_team
                or min(game.home_score, game.away_score) < 0
                or row["location"] not in ("Home", "Neutral")
            ):
                raise ValueError(f"Invalid game on CSV line {reader.line_num}.")
            if game.game_id in seen:
                raise ValueError(f"Duplicate game: {game.game_id}")
            seen.add(game.game_id)
            games.append(game)

    if not games:
        raise ValueError("No completed regular-season games found.")
    return sorted(games, key=lambda game: (game.game_date, game.game_id))
