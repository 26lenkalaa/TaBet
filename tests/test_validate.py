import pandas as pd

from tabet.validate import has_errors, validate


def checks(issues, level="error"):
    return {i.check for i in issues if i.level == level}


def test_real_data_has_no_errors(real):
    issues = validate(*real)
    assert not has_errors(issues), [str(i) for i in issues]


def test_detects_corrupted_rows(real):
    m = real[0].copy()
    m.loc[0, "result"] = "A" if m.loc[0, "result"] != "A" else "H"
    m.loc[1, "home_shots_on_target"] = m.loc[1, "home_shots"] + 1
    m.loc[2, "ht_home_goals"] = m.loc[2, "home_goals"] + 1
    m.loc[3, "home_goals"] = -1
    m = pd.concat([m, m.iloc[[10]]], ignore_index=True)
    found = checks(validate(m))
    assert {"result_mismatch", "sot_exceeds_shots", "ht_exceeds_ft", "negative_count", "duplicate_match"} <= found


def test_future_completed_result_is_error(real):
    m = real[0]
    assert "future_result" in checks(validate(m, now=pd.Timestamp("2024-01-01", tz="UTC")))


def test_partial_latest_season_is_warning_only(real):
    m = real[0]
    m = m[m["season"] <= "2025-26"]  # make 2025-26 the latest season
    latest = m[m["season"] == "2025-26"]
    trimmed = m.drop(latest.index[-50:])
    issues = validate(trimmed)
    assert "partial_season" in checks(issues, "warning")
    assert not has_errors(issues)


def test_partial_older_season_is_error(real):
    m = real[0]
    older = m[m["season"] == "2022-23"]
    assert "incomplete_season" in checks(validate(m.drop(older.index[:5])))


def test_repeated_fixture_is_error(real):
    m = real[0].copy()
    i = m.index[0]
    j = m[(m["season"] == m.loc[i, "season"]) & (m.index != i)].index[0]
    m.loc[j, ["home_id", "away_id"]] = m.loc[i, ["home_id", "away_id"]].values
    m.loc[j, "match_id"] = "dup-pair"
    assert "repeated_fixture" in checks(validate(m))


def test_invalid_odds(real):
    m, o = real
    o = o.copy()
    o.loc[0, "AvgH"] = 0.9
    assert "invalid_odds" in checks(validate(m, o))
