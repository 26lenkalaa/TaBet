import pandas as pd

from tabet import transfers as T

HTML = """
<table class="wikitable"><tr><th>Date</th><th>Player</th><th>Moving from</th><th>Moving to</th><th>Fee</th></tr>
<tr><td rowspan="2">1 July 2025</td><td>A One</td><td>Strasbourg</td><td>Sunderland</td><td>£30m<sup>[1]</sup></td></tr>
<tr><td>B Two</td><td>Unattached</td><td>Burnley</td><td>Free</td></tr>
<tr><td>3 July 2025</td><td>C Three</td><td>NEC</td><td>Sunderland</td><td>Undisclosed</td></tr>
<tr><td>4 July 2025</td><td>D Four</td><td>Ajax</td><td>Leeds United</td><td>£500k</td></tr></table>
<table class="wikitable"><tr><th>Start date</th><th>End date</th><th>Player</th><th>Moving from</th><th>Moving to</th></tr>
<tr><td>5 August 2025</td><td>30 June 2026</td><td>E Five</td><td>Chelsea</td><td>Sunderland</td></tr></table>
"""


def test_rows_spans_loans_and_fees_parse():
    df = T.parse_page({"html": HTML, "revision_id": 1, "revision_utc": "2025-09-03T00:00:00Z"}, "summer_2025")
    assert list(df["player"]) == ["A One", "B Two", "C Three", "D Four", "E Five"]
    assert df.loc[1, "date"] == pd.Timestamp("2025-07-01", tz="UTC")            # date carried down a row span
    assert list(df["kind"]) == ["fee", "free", "undisclosed", "fee", "loan"]
    assert df.loc[0, "fee_gbp_m"] == 30.0 and df.loc[3, "fee_gbp_m"] == 0.5
    assert df.loc[0, "to_id"] == "sunderland" and pd.isna(df.loc[0, "from_id"])


def test_fee_text_variants():
    assert T.parse_fee("£22.5m") == ("fee", 22.5)
    assert T.parse_fee("£1.2 million + add-ons") == ("fee", 1.2)
    assert T.parse_fee("£750,000") == ("fee", 0.75)
    assert T.parse_fee("Undisclosed") == ("undisclosed", None)
    assert T.parse_fee("Swap") == ("other", None)


def test_windows_only_open_ones_and_settling():
    w = T.windows(pd.Timestamp("2026-11-26", tz="UTC"))
    slugs = [x["slug"] for x in w]
    assert slugs[0] == "summer_2005" and slugs[-1] == "summer_2026" and "winter_2026_27" not in slugs
    summer26 = w[-1]
    assert T._settled(summer26, pd.Timestamp("2026-11-26", tz="UTC"))
    assert not T._settled(summer26, pd.Timestamp("2026-09-15", tz="UTC"))


def test_auto_refresh_triggers(tmp_path):
    calls = []

    def fake_build(refresh, today):
        calls.append(today)
        return pd.DataFrame({"window": ["summer_2026"], "player": ["x"]})

    st = tmp_path / "state.json"
    nov = pd.Timestamp("2026-11-26 12:00", tz="UTC")                     # no window active
    assert T.auto_refresh(nov, {1, 2}, st, fake_build)["reason"] == "first build"
    assert T.auto_refresh(nov + pd.Timedelta(hours=3), {1, 2}, st, fake_build) is None       # nothing new
    r = T.auto_refresh(nov + pd.Timedelta(hours=4), {1, 2, 3}, st, fake_build)               # a new signing
    assert r["reason"] == "1 new FPL player(s)"
    assert T.auto_refresh(nov + pd.Timedelta(hours=4, minutes=30), {1, 2, 3, 4}, st, fake_build) is None  # hourly cap
    jan = pd.Timestamp("2027-01-10 12:00", tz="UTC")                     # the winter window is open
    assert T.auto_refresh(jan, {1, 2, 3}, st, fake_build)["reason"] == "daily refresh during a window"
    assert T.auto_refresh(jan + pd.Timedelta(hours=5), {1, 2, 3}, st, fake_build) is None
    assert len(calls) == 3


def test_columns_are_found_by_header_and_style_text_is_ignored():
    html = """<table class="wikitable"><tr><th>Date</th><th>Name</th><th>Nat</th><th>Moving from</th><th>Moving to</th><th>Fee</th></tr>
    <tr><td>2 May 2006<style>.mw-parser-output .x{}</style></td><td>G R</td><td></td><td>Tottenham Hotspur</td><td>Southampton</td><td>£2m</td></tr></table>
    <table class="wikitable"><tr><th>Date</th><th>Name</th><th>Moving from</th><th>Moving to</th></tr>
    <tr><td>3 July 2017</td><td>L O</td><td>Chelsea</td><td>Hull City</td></tr></table>
    <table class="wikitable"><tr><th>Date</th><th>Name</th><th>New Club</th></tr><tr><td>1 July 2017</td><td>X</td><td>Retired</td></tr></table>"""
    df = T.parse_page({"html": html, "revision_id": 1, "revision_utc": ""}, "summer_2017")
    assert list(df["player"]) == ["G R", "L O"]
    assert df.loc[0, "date"] == pd.Timestamp("2006-05-02", tz="UTC") and df.loc[0, "fee_gbp_m"] == 2.0
    assert df.loc[1, "kind"] == "loan" and df.loc[1, "to_id"] == "hull"


def test_club_features_use_only_earlier_moves():
    moves = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-20", "2025-07-01", "2025-08-15", "2025-08-16", "2025-07-02"], utc=True),
        "to_id": ["sunderland", "sunderland", "sunderland", "sunderland", "leeds"],
        "from_id": [None, None, None, None, "sunderland"],
        "kind": ["fee", "fee", "undisclosed", "fee", "fee"], "fee_gbp_m": [99.0, 30.0, None, 10.0, 5.0]})
    f = T.club_features(moves, "sunderland", 2025, pd.Timestamp("2025-08-16 15:00", tz="UTC"))
    # April is before the window; the 16 August move is on the cutoff's own day, so not yet known
    assert f["fees_in"] == 30.0 and f["signings"] == 2 and f["undisclosed_in"] == 1
    assert f["fees_out"] == 5.0 and f["net_spend"] == 25.0 and f["departures"] == 1


def test_squad_cost_keeps_current_players_and_adjusts_for_inflation():
    moves = pd.DataFrame({
        "date": pd.to_datetime(["2015-07-01", "2020-07-01", "2021-07-01", "2022-07-01", "2022-07-02"], utc=True),
        "player": ["Old", "Mid", "Mid", "New", "Gone"],
        "to_id": ["leeds", "leeds", "chelsea", "leeds", "leeds"], "from_id": [None, None, "leeds", None, None],
        "kind": ["fee", "fee", "fee", "fee", "fee"], "fee_gbp_m": [10.0, 20.0, 30.0, 40.0, 5.0]})
    idx = pd.Series({2015: 1.0, 2020: 2.0, 2021: 2.0, 2022: 2.0})
    f = T.squad_cost(moves, "leeds", pd.Timestamp("2022-07-02 15:00", tz="UTC"), idx)
    # Old (10 in 2015 money = 20 today) + New (40); Mid was sold; Gone joined on the cutoff's own day
    assert f["squad_cost"] == 60.0 and f["squad_players_recorded"] == 2


def test_fee_index_point_in_time_ignores_later_moves():
    moves = pd.DataFrame({"date": pd.to_datetime(["2010-07-01", "2011-07-01", "2012-07-01"], utc=True),
                          "kind": ["fee"] * 3, "to_id": ["leeds"] * 3, "fee_gbp_m": [1.0, 2.0, 100.0]})
    idx = T.fee_index(moves, before=pd.Timestamp("2012-01-01", tz="UTC"))
    assert list(idx.index) == [2010, 2011] and idx.loc[2011] == 1.5
