from tabet import api_football as A
from tabet.site_data import match_events


def _fx():
    ev = lambda m, team, typ, det, who, assist=None, extra=None: {
        "time": {"elapsed": m, "extra": extra}, "team": {"id": team}, "player": {"name": who},
        "assist": {"name": assist}, "type": typ, "detail": det}
    return {"fixture": {"id": 7, "date": "2025-08-16T14:00:00+00:00"},
            "teams": {"home": {"id": 1, "name": "Manchester United"}, "away": {"id": 2, "name": "Nottingham Forest"}},
            "goals": {"home": 2, "away": 1},
            "events": [ev(90, 1, "Goal", "Normal Goal", "B. Fernandes", "A. Diallo", extra=3),
                       ev(12, 1, "Goal", "Penalty", "B. Fernandes"),
                       ev(30, 2, "Card", "Yellow Card", "M. Gibbs-White"),
                       ev(55, 2, "Goal", "Normal Goal", "C. Wood", "A. Elanga"),
                       ev(60, 1, "subst", "Substitution 1", "M. Mount", "A. Garnacho"),
                       ev(70, 2, "Var", "Goal cancelled", "C. Wood")]}


def test_compact_events_keeps_goals_assists_and_cards_in_minute_order():
    c = A.compact_events(_fx())
    assert [(e["min"], e["type"]) for e in c["events"]] == [(12, "pen_goal"), (30, "yellow"), (55, "goal"), (90, "goal")]
    assert c["events"][3] == {"min": 90, "extra": 3, "side": "home", "type": "goal", "player": "B. Fernandes", "assist": "A. Diallo"}
    assert "assist" not in c["events"][0] and c["events"][1]["side"] == "away"
    assert c["check"] is True                      # 2 home + 1 away scoring events = the final score


def test_compact_events_flags_a_feed_that_misses_a_goal():
    fx = _fx()
    fx["goals"]["away"] = 2
    assert A.compact_events(fx)["check"] is False


def test_match_events_joins_on_the_home_away_pair():
    entries = [{"id": "m1", "season": "2025-26", "status": "played", "home": "Man United", "away": "Nott'm Forest"},
               {"id": "m2", "season": "2025-26", "status": "upcoming", "home": "Man United", "away": "Nott'm Forest"}]
    from tabet.teams import TEAMS
    entries[0]["home"], entries[0]["away"] = TEAMS["man_united"], TEAMS["nottm_forest"]
    got = match_events(entries, {"2025-26": {"fixtures": [A.compact_events(_fx())]}})
    assert list(got["2025-26"]) == ["m1"] and got["2025-26"]["m1"]["check"] is True
