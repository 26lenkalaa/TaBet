import pandas as pd
import pytest

from tabet.ingest import IngestError, implied_probabilities, load_season, season_from_filename

OK = "E0,16/08/2026,15:00,Arsenal,Fulham,2,1,H,1,0,H,A Ref,10,5,4,2,1.5,4.0,6.0"


def test_real_data_shape(real):
    matches, odds = real
    assert len(matches) == 1900
    assert set(matches["season"]) == {"2021-22", "2022-23", "2023-24", "2024-25", "2025-26"}
    assert matches["match_id"].is_unique
    assert (matches["division"] == "E0").all()  # BOM-prefixed files still yield "Div"
    assert len(odds) == len(matches)
    assert not {"home_goals", "result"} & set(odds.columns)


def test_kickoff_converted_from_uk_time_to_utc(real):
    m = real[0]
    first = m[m["match_id"] == "2021-22-20210813-brentford-arsenal"].iloc[0]
    assert first["kickoff_utc"] == pd.Timestamp("2021-08-13 19:00", tz="UTC")  # 20:00 BST
    assert (first["home_goals"], first["away_goals"], first["result"]) == (2, 0, "H")


def test_season_from_filename():
    assert season_from_filename("Premier League 21-22.csv") == "2021-22"
    assert season_from_filename("E0 99-00.csv") == "1999-00"
    with pytest.raises(ValueError):
        season_from_filename("Premier League 21-23.csv")


def test_bom_and_two_digit_year(write_csv):
    p = write_csv([OK.replace("16/08/2026", "16/08/26")], bom=True)
    m, _ = load_season(p)
    assert m.loc[0, "kickoff_utc"] == pd.Timestamp("2026-08-16 14:00", tz="UTC")
    assert m.loc[0, "division"] == "E0"


def test_missing_time_assumes_late_kickoff(write_csv):
    p = write_csv([OK.replace(",15:00,", ",,")])
    m, _ = load_season(p)
    assert not m.loc[0, "kickoff_time_known"]
    assert m.loc[0, "kickoff_utc"] == pd.Timestamp("2026-08-16 21:00", tz="UTC")


def test_scheduled_fixture(write_csv):
    p = write_csv(["E0,23/08/2026,15:00,Fulham,Arsenal,,,,,,,,,,,,2.5,3.4,2.8"])
    m, _ = load_season(p)
    assert m.loc[0, "status"] == "scheduled"
    assert pd.isna(m.loc[0, "home_goals"])


@pytest.mark.parametrize("row,msg", [
    (OK.replace("Fulham", "Fulham Reserves"), "unknown team"),
    (OK.replace(",2,1,H,", ",2,,H,"), "incomplete result"),
    (OK.replace(",2,1,H,", ",two,1,H,"), "non-numeric"),
    (OK.replace("16/08/2026", "2026-08-16"), "Date"),
])
def test_bad_rows_raise(write_csv, row, msg):
    with pytest.raises(IngestError, match=msg):
        load_season(write_csv([row]))


def test_zero_odds_treated_as_missing(write_csv):
    _, odds = load_season(write_csv([OK.replace("1.5,4.0,6.0", "0,4.0,6.0")]))
    assert pd.isna(odds.loc[0, "AvgH"])


def test_implied_probabilities_sum_to_one(real):
    p = implied_probabilities(real[1]).dropna()
    assert len(p) > 1800
    assert ((p[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs() < 1e-9).all()


def test_source_fix_applied(real):
    m = real[0]
    row = m[m["match_id"] == "2021-22-20210815-newcastle-west_ham"].iloc[0]
    assert pd.isna(row["away_shots"]) and pd.isna(row["away_shots_on_target"])
    assert row["away_goals"] == 4  # untouched fields kept
