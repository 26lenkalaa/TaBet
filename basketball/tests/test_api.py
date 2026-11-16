"""Offline tests for the API, using FastAPI's TestClient and a temporary database."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import backtest, db, market, predict  # noqa: E402
from src.api.main import create_app  # noqa: E402

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}


@pytest.fixture
def games():
    results = [
        ("2018-10-16", "2018-19", "regular", BOS, NYK, 110, 95),
        ("2018-10-17", "2018-19", "regular", LAL, GSW, 101, 104),
        ("2019-10-22", "2019-20", "regular", NYK, LAL, 99, 108),
        ("2019-10-22", "2019-20", "regular", GSW, BOS, 120, 111),
        ("2019-10-25", "2019-20", "regular", BOS, LAL, 105, 100),
        ("2020-08-17", "2019-20", "playoffs", LAL, BOS, 112, 107),
        ("2020-12-22", "2020-21", "regular", GSW, NYK, 125, 100),
    ]
    rows = []
    for number, (date, season, game_type, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00200{number:05d}", "season": season, "date": date,
            "game_type": game_type, "is_playoffs": game_type == "playoffs",
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
            "home_fga": 88, "home_fta": 22, "home_oreb": 10, "home_tov": 13,
            "away_fga": 86, "away_fta": 24, "away_oreb": 9, "away_tov": 14,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def odds(games):
    return pd.DataFrame({
        "game_id": games["game_id"].iloc[:6],                # the last game has no odds
        "spread_close": [-6.0, 2.5, 4.0, -3.0, -2.0, -1.5],
        "total_close": [215.5] * 6,
        "home_ml": [-240, 120, 160, -150, -130, -120],
        "away_ml": [200, -140, -190, 130, 110, 100],
    })


@pytest.fixture
def db_file(games, odds, tmp_path):
    file = tmp_path / "basketball.db"
    db.build(db.connect(file), games, odds)
    return file


@pytest.fixture
def refresh_calls():
    return []


@pytest.fixture
def client(db_file, refresh_calls, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "test-key")

    def fake_refresh():
        refresh_calls.append("ran")
        return [{"step": step, "ok": True, "output": "fine", "required": True}
                for step in ("fetch_games", "clean", "fetch_odds", "db")]

    return TestClient(create_app(db_file, refresh=fake_refresh))


# --- GET /games ----------------------------------------------------------------

def test_games_lists_the_games_on_that_date(client):
    response = client.get("/games", params={"date": "2019-10-22"})
    assert response.status_code == 200
    body = response.json()
    assert [(game["home"], game["away"]) for game in body] == [("NYK", "LAL"), ("GSW", "BOS")]
    assert body[0]["game_id"] == "0020000003"
    assert (body[0]["home_pts"], body[0]["away_pts"]) == (99, 108)


def test_games_show_model_against_market(client, db_file):
    game = client.get("/games", params={"date": "2018-10-16"}).json()[0]
    stored = db.read_predictions(db.connect(db_file), "eff-v2").set_index("game_id").loc["0020000001"]
    assert game["model"]["version"] == "eff-v2"
    assert game["model"]["home_win_prob"] == pytest.approx(stored["home_win_prob"])
    assert game["model"]["predicted_margin"] == pytest.approx(stored["predicted_margin"])
    assert game["market"]["spread"] == -6.0
    assert game["market"]["margin"] == 6.0
    assert game["market"]["home_win_prob"] == pytest.approx(market.moneylines_to_probs(-240, 200)[0])


def test_games_can_use_the_elo_model(client, db_file):
    game = client.get("/games", params={"date": "2018-10-16", "model": "elo"}).json()[0]
    stored = db.read_predictions(db.connect(db_file), "elo-v1").set_index("game_id").loc["0020000001"]
    assert game["model"]["version"] == "elo-v1"
    assert game["model"]["home_win_prob"] == pytest.approx(stored["home_win_prob"])


def test_a_game_without_odds_has_no_market(client):
    game = client.get("/games", params={"date": "2020-12-22"}).json()[0]
    assert game["market"] is None
    assert game["model"] is not None


def test_a_date_with_no_games_gives_an_empty_list(client):
    response = client.get("/games", params={"date": "2030-01-01"})
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("params", [{}, {"date": "yesterday"}, {"date": "2019-13-45"},
                                    {"date": "2019-10-22", "model": "magic"}])
def test_games_rejects_a_missing_or_bad_date_or_model(client, params):
    assert client.get("/games", params=params).status_code == 422


# --- GET /predict --------------------------------------------------------------

def test_predict_matches_predict_py(client, db_file):
    body = client.get("/predict", params={"home": "BOS", "away": "NYK"}).json()
    expected = predict.predict_game(predict.load_games(db_file), "BOS", "NYK", model="efficiency")
    assert body["home"] == "BOS" and body["away"] == "NYK"
    assert body["model"] == "eff-v2"
    assert body["home_win_prob"] == pytest.approx(expected.home_win_prob)
    assert body["predicted_margin"] == pytest.approx(expected.predicted_margin)
    assert body["predicted_home_pts"] == pytest.approx(expected.predicted_home_pts)
    assert body["home_win_prob"] + body["away_win_prob"] == pytest.approx(1.0)
    assert body["ratings_through"] == "2020-12-22"


def test_predict_accepts_lower_case_neutral_and_the_elo_model(client, db_file):
    games = predict.load_games(db_file)
    neutral = client.get("/predict", params={"home": "bos", "away": "nyk", "neutral": "true"}).json()
    assert neutral["home"] == "BOS" and neutral["neutral"] is True
    assert neutral["home_win_prob"] == pytest.approx(predict.predict_game(games, "BOS", "NYK", True, "efficiency").home_win_prob)

    with_elo = client.get("/predict", params={"home": "BOS", "away": "NYK", "model": "elo"}).json()
    assert with_elo["model"] == "elo-v1"
    assert with_elo["home_win_prob"] == pytest.approx(
        predict.predict_game(games, "BOS", "NYK", model="elo").home_win_prob)
    assert with_elo["predicted_home_pts"] is None


def test_predict_gives_the_same_answer_twice(client):
    first = client.get("/predict", params={"home": "LAL", "away": "GSW"}).json()
    second = client.get("/predict", params={"home": "LAL", "away": "GSW"}).json()
    assert first == second


def test_predict_unknown_team_is_404(client):
    response = client.get("/predict", params={"home": "BOS", "away": "ZZZ"})
    assert response.status_code == 404
    assert "Unknown team: ZZZ" in response.json()["detail"]


def test_predict_same_team_twice_is_400(client):
    assert client.get("/predict", params={"home": "BOS", "away": "BOS"}).status_code == 400


def test_predict_needs_both_teams(client):
    assert client.get("/predict", params={"home": "BOS"}).status_code == 422


# --- GET /backtest/summary -----------------------------------------------------

def test_backtest_summary_matches_backtest_py(client, db_file):
    body = client.get("/backtest/summary").json()
    engine = db.connect(db_file)
    expected = backtest.summarize(backtest.build_table(
        db.read_games(engine, possessions=True), db.read_odds(engine), "efficiency"))
    assert body["model"] == "eff-v2"
    assert body["warmup_seasons"] == 1
    assert [(row["part"], row["season"]) for row in body["scores"]] == list(zip(expected["part"], expected["season"]))
    assert [row["brier_model"] for row in body["scores"]] == pytest.approx(list(expected["brier_model"]))
    assert [row["brier_market"] for row in body["scores"]] == pytest.approx(list(expected["brier_market"]))
    assert [row["games"] for row in body["scores"]] == list(expected["games"])


def test_backtest_summary_for_elo(client):
    body = client.get("/backtest/summary", params={"model": "elo"}).json()
    assert body["model"] == "elo-v1"
    assert {"regular", "playoffs"} == {row["part"] for row in body["scores"]}


# --- POST /admin/refresh -------------------------------------------------------

def test_refresh_runs_with_the_right_key(client, refresh_calls):
    response = client.post("/admin/refresh", headers={"X-Admin-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert [step["step"] for step in body["steps"]] == ["fetch_games", "clean", "fetch_odds", "db"]
    assert refresh_calls == ["ran"]


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": ""}])
def test_refresh_is_refused_without_the_right_key(client, refresh_calls, headers):
    response = client.post("/admin/refresh", headers=headers)
    assert response.status_code == 401
    assert refresh_calls == []


def test_refresh_is_closed_when_no_admin_key_is_configured(client, refresh_calls, monkeypatch):
    monkeypatch.delenv("ADMIN_KEY")
    response = client.post("/admin/refresh", headers={"X-Admin-Key": ""})
    assert response.status_code == 503
    assert refresh_calls == []


def test_refresh_reports_a_failed_step(db_file, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "test-key")
    failing = lambda: [{"step": "fetch_games", "ok": False, "output": "NBA.com timed out", "required": True}]  # noqa: E731
    response = TestClient(create_app(db_file, refresh=failing)).post(
        "/admin/refresh", headers={"X-Admin-Key": "test-key"})
    assert response.status_code == 200
    assert response.json() == {"ok": False, "steps": [
        {"step": "fetch_games", "ok": False, "output": "NBA.com timed out"}]}


def test_refresh_still_succeeds_when_only_the_optional_odds_step_fails(db_file, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "test-key")
    steps = [{"step": "fetch_games", "ok": True, "output": "", "required": True},
             {"step": "clean", "ok": True, "output": "", "required": True},
             {"step": "fetch_odds", "ok": False, "output": "No CSV", "required": False},
             {"step": "db", "ok": True, "output": "", "required": True}]
    response = TestClient(create_app(db_file, refresh=lambda: steps)).post(
        "/admin/refresh", headers={"X-Admin-Key": "test-key"})
    assert response.json()["ok"] is True


def test_refresh_must_be_a_post(client):
    assert client.get("/admin/refresh").status_code == 405


def test_new_data_shows_up_after_the_database_is_rebuilt(client, db_file, games, odds):
    before = client.get("/predict", params={"home": "BOS", "away": "NYK"}).json()
    extra = games.iloc[[0]].assign(game_id="0022000099", season="2020-21", date="2020-12-25",
                                   home_pts=80, away_pts=130)
    db.build(db.connect(db_file), pd.concat([games, extra], ignore_index=True), odds)
    after = client.get("/predict", params={"home": "BOS", "away": "NYK"}).json()
    assert after["ratings_through"] == "2020-12-25"
    assert after["home_win_prob"] < before["home_win_prob"]


# --- the service itself --------------------------------------------------------

def test_a_missing_database_is_a_503_that_says_what_to_run(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.db"))
    for path, params in (("/games", {"date": "2019-10-22"}), ("/predict", {"home": "BOS", "away": "NYK"}),
                         ("/backtest/summary", {})):
        response = client.get(path, params=params)
        assert response.status_code == 503, path
        assert "db.py" in response.json()["detail"]


def test_the_docs_page_lists_the_four_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/games", "/predict", "/backtest/summary", "/admin/refresh"}
    assert client.get("/docs").status_code == 200


def test_run_refresh_runs_the_scripts_in_order_and_stops_at_a_required_failure(tmp_path, monkeypatch):
    import src.api.main as api

    for name, code in (("one.py", "print('one done')"), ("two.py", "import sys; print('two broke'); sys.exit(1)"),
                       ("three.py", "print('three done')")):
        (tmp_path / name).write_text(code)
    monkeypatch.setattr(api, "BASKETBALL_DIR", tmp_path)

    monkeypatch.setattr(api, "REFRESH_STEPS", [("one", "one.py", True), ("two", "two.py", False),
                                               ("three", "three.py", True)])
    optional_failure = api.run_refresh()
    assert [(step["step"], step["ok"]) for step in optional_failure] == [("one", True), ("two", False), ("three", True)]
    assert optional_failure[1]["output"] == "two broke"

    monkeypatch.setattr(api, "REFRESH_STEPS", [("one", "one.py", True), ("two", "two.py", True),
                                               ("three", "three.py", True)])
    required_failure = api.run_refresh()
    assert [step["step"] for step in required_failure] == ["one", "two"]   # three never ran
