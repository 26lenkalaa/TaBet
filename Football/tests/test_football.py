import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from backtest import backtest, market_home_probability
from data import Game, load_games
from model import Elo, forecast, replay


def game(day: str, home_score: int = 24, away_score: int = 17,
         odds: bool = True, game_id: str = "game") -> Game:
    played_on = date.fromisoformat(day)
    return Game(
        game_id, played_on.year, played_on, "KC", "BAL",
        home_score, away_score, False,
        "-150" if odds else "", "+130" if odds else "",
    )


class FootballTests(unittest.TestCase):
    def test_loader_skips_future_and_playoffs(self) -> None:
        header = "game_id,season,game_type,gameday,home_team,away_team,home_score,away_score,location\n"
        rows = (
            "future,2025,REG,2025-09-21,KC,BAL,,,Home\n"
            "playoff,2024,WC,2025-01-12,KC,BAL,24,17,Home\n"
            "played,2025,REG,2025-09-14,KC,OAK,24,17,Neutral\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.csv"
            path.write_text(header + rows, encoding="utf-8")
            games = load_games(path)
        self.assertEqual([item.game_id for item in games], ["played"])
        self.assertTrue(games[0].neutral)
        self.assertEqual(games[0].away_team, "LV")

    def test_home_advantage_and_day_cutoff(self) -> None:
        model = Elo()
        self.assertEqual(model.home_probability("KC", "BAL", neutral=True), 0.5)
        day = "2025-09-07"
        first, second = list(replay([
            game(day, game_id="a"), game(day, game_id="b"),
        ], model))
        self.assertEqual(first[1], second[1])
        self.assertGreater(first[1], 0.5)
        self.assertGreater(model.ratings["KC"], model.ratings["BAL"])

    def test_forecast_ignores_target_day_and_later(self) -> None:
        prior = game("2024-09-01")
        cutoff = date(2025, 9, 7)
        expected = forecast([prior], "KC", "BAL", cutoff, 2025)
        actual = forecast([
            prior, game("2025-09-07", 0, 99), game("2025-09-14", 0, 99),
        ], "KC", "BAL", cutoff, 2025)
        self.assertEqual(expected, actual)

    def test_market_and_backtest_use_same_games(self) -> None:
        scored = game("2025-09-07", game_id="scored")
        self.assertAlmostEqual(market_home_probability(scored), 69 / 119)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.csv"
            report = backtest([
                game("2024-09-01"), scored,
                game("2025-09-14", 20, 20, game_id="tie"),
                game("2025-09-21", odds=False, game_id="missing"),
            ], 2025, output)
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["status"] for row in rows], ["scored", "tie", "missing_odds"])
        self.assertEqual(report["scored_games"], 1)
        self.assertAlmostEqual(report["market_brier"], (1 - 69 / 119) ** 2)


if __name__ == "__main__":
    unittest.main()
