"""The basketball API: four endpoints over the database that db.py builds.

    GET  /games?date=YYYY-MM-DD        games on a date, model against market
    GET  /predict?home=BOS&away=NYK    predict one matchup
    GET  /backtest/summary             model against market scores, per season
    POST /admin/refresh                re-fetch data (needs the admin key)

Run it from the repo root (after db.py), then open http://127.0.0.1:8000/docs :

    python3 -m uvicorn src.api.main:app --app-dir basketball --reload

The admin key comes from ADMIN_KEY in basketball/.env (see basketball/.env.example).
"""

from __future__ import annotations

import datetime
import hmac
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

BASKETBALL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASKETBALL_DIR))

from src import backtest, db, market, predict  # noqa: E402

ModelName = Literal["efficiency", "elo"]
VERSIONS = {name: version for name, (version, _) in backtest.MODELS.items()}

# The scripts a refresh runs, in order. Odds are optional: the Kaggle files may not be on this machine.
REFRESH_STEPS = [
    ("fetch_games", "src/ingest/fetch_games.py", True),
    ("clean", "src/clean.py", True),
    ("fetch_odds", "src/ingest/fetch_odds.py", False),
    ("db", "src/db.py", True),
]


class ModelPrediction(BaseModel):
    version: str
    home_win_prob: float
    predicted_margin: float


class MarketLine(BaseModel):
    home_win_prob: float | None
    margin: float | None
    spread: float | None
    total: float | None
    home_ml: float | None
    away_ml: float | None


class Game(BaseModel):
    game_id: str
    date: str
    season: str
    game_type: str
    home: str
    away: str
    home_pts: int
    away_pts: int
    model: ModelPrediction | None
    market: MarketLine | None


class MatchupPrediction(BaseModel):
    home: str
    away: str
    model: str
    neutral: bool
    home_win_prob: float
    away_win_prob: float
    predicted_margin: float
    predicted_home_pts: float | None
    predicted_away_pts: float | None
    home_rating: float
    away_rating: float
    ratings_through: str


class SeasonScores(BaseModel):
    part: str
    season: str
    games: int
    brier_model: float
    brier_market: float
    logloss_model: float
    logloss_market: float
    margin_err_model: float | None
    margin_err_market: float | None


class BacktestSummary(BaseModel):
    model: str
    warmup_seasons: int
    scores: list[SeasonScores]


class RefreshStep(BaseModel):
    step: str
    ok: bool
    output: str


class RefreshResult(BaseModel):
    ok: bool
    steps: list[RefreshStep]


def run_refresh() -> list[dict]:
    """Run the fetch, clean, odds and database scripts one after another; stop at the first required failure."""
    results = []
    for step, script, required in REFRESH_STEPS:
        finished = subprocess.run([sys.executable, str(BASKETBALL_DIR / script)],
                                  capture_output=True, text=True, cwd=BASKETBALL_DIR.parent)
        ok = finished.returncode == 0
        tail = (finished.stdout + finished.stderr).strip().splitlines()[-5:]
        results.append({"step": step, "ok": ok, "output": "\n".join(tail), "required": required})
        if not ok and required:
            break
    return results


def none_if_missing(value):
    return None if pd.isna(value) else float(value)


def create_app(db_file: Path = db.DB_FILE, refresh: Callable[[], list[dict]] = run_refresh) -> FastAPI:
    load_dotenv(BASKETBALL_DIR / ".env")
    app = FastAPI(title="TaBet Basketball", description="NBA predictions against the betting market.")
    cache: dict[tuple, object] = {}

    def engine():
        try:
            return db.connect(db_file, must_exist=True)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None

    def cached(name: str, model: str, build: Callable[[], object]):
        """Replaying every game takes a moment, so keep the result until the database file changes."""
        key = (name, model, Path(db_file).stat().st_mtime)
        if key not in cache:
            for old in [other for other in cache if other[:2] == key[:2]]:
                del cache[old]
            cache[key] = build()
        return cache[key]

    @app.get("/games", response_model=list[Game], summary="Games on a date, model against market")
    def games(date: datetime.date = Query(description="YYYY-MM-DD"), model: ModelName = "efficiency"):
        connection = engine()
        played = db.read_games(connection, date=date.isoformat())
        if played.empty:
            return []
        played = played.merge(db.read_odds(connection), on="game_id", how="left")
        played = market.add_market_predictions(played, backtest.MARGIN_SD)
        stored = db.read_predictions(connection, VERSIONS[model]).set_index("game_id")

        result = []
        for game in played.to_dict("records"):
            prediction = None
            if game["game_id"] in stored.index:
                row = stored.loc[game["game_id"]]
                prediction = ModelPrediction(version=VERSIONS[model], home_win_prob=row["home_win_prob"],
                                             predicted_margin=row["predicted_margin"])
            line = MarketLine(
                home_win_prob=none_if_missing(game["market_home_win_prob"]),
                margin=none_if_missing(game["market_margin"]),
                spread=none_if_missing(game["spread_close"]), total=none_if_missing(game["total_close"]),
                home_ml=none_if_missing(game["home_ml"]), away_ml=none_if_missing(game["away_ml"]))
            has_line = any(value is not None for value in line.model_dump().values())
            result.append(Game(
                game_id=game["game_id"], date=game["date"], season=game["season"], game_type=game["game_type"],
                home=game["home"], away=game["away"], home_pts=game["home_pts"], away_pts=game["away_pts"],
                model=prediction, market=line if has_line else None))
        return result

    @app.get("/predict", response_model=MatchupPrediction, summary="Predict one matchup")
    def predict_matchup(home: str = Query(description="home team abbreviation, e.g. BOS"),
                        away: str = Query(description="away team abbreviation, e.g. NYK"),
                        neutral: bool = False, model: ModelName = "efficiency"):
        connection = engine()
        played, fitted = cached("fit", model, lambda: (
            (frame := db.read_games(connection, possessions=True)), predict.fit(frame, model)))
        if played.empty:
            raise HTTPException(status_code=503, detail="The database has no games. Run db.py.")
        try:
            result = predict.predict_game(played, home, away, neutral, model, fitted)
        except ValueError as error:
            unknown = str(error).startswith("Unknown team")
            raise HTTPException(status_code=404 if unknown else 400, detail=str(error)) from None
        return MatchupPrediction(
            home=result.home, away=result.away, model=VERSIONS[model], neutral=neutral,
            home_win_prob=result.home_win_prob, away_win_prob=1 - result.home_win_prob,
            predicted_margin=result.predicted_margin,
            predicted_home_pts=result.predicted_home_pts, predicted_away_pts=result.predicted_away_pts,
            home_rating=result.home_rating, away_rating=result.away_rating,
            ratings_through=result.last_game_date)

    @app.get("/backtest/summary", response_model=BacktestSummary, summary="Model against market scores")
    def backtest_summary(model: ModelName = "efficiency"):
        connection = engine()
        summary = cached("summary", model, lambda: backtest.summarize(backtest.build_table(
            db.read_games(connection, possessions=True), db.read_odds(connection), model)))
        rows = summary.astype(object).where(summary.notna(), None).to_dict("records")
        return BacktestSummary(model=VERSIONS[model], warmup_seasons=backtest.WARMUP_SEASONS, scores=rows)

    @app.post("/admin/refresh", response_model=RefreshResult, summary="Re-fetch data (admin key needed)")
    def admin_refresh(x_admin_key: str | None = Header(default=None)):
        admin_key = os.environ.get("ADMIN_KEY", "").strip()
        if not admin_key:
            raise HTTPException(status_code=503, detail="ADMIN_KEY is not set. Add it to basketball/.env.")
        if not x_admin_key or not hmac.compare_digest(x_admin_key.encode(), admin_key.encode()):
            raise HTTPException(status_code=401, detail="Wrong or missing X-Admin-Key header.")
        steps = refresh()
        ok = all(step["ok"] or not step.get("required", True) for step in steps) and len(steps) == len(REFRESH_STEPS)
        cache.clear()
        return RefreshResult(ok=ok, steps=[RefreshStep(step=step["step"], ok=step["ok"], output=step["output"])
                                           for step in steps])

    return app


app = create_app()
