"""Paper betting: pick, price, stake base, settlement, and what happens without the private bucket."""
import gzip
import json

import pandas as pd

from tabet import paper_bets as pb


def rec(fx, gw, logged, kickoff, p=(0.5, 0.3, 0.2)):
    return {"fpl_fixture": fx, "gameweek": gw, "logged_utc": logged, "kickoff_utc": kickoff, "home_id": f"h{fx}",
            "away_id": f"a{fx}", "candidates": {"pE": {"p_home": p[0], "p_draw": p[1], "p_away": p[2]}}}


def books(rows):
    return gzip.compress(pd.DataFrame(rows).to_csv(index=False).encode())


def setup(monkeypatch, snaps):
    monkeypatch.setattr(pb, "matched_snapshot", lambda r, o: snaps.get(r["fpl_fixture"]))


def test_week_of_bets(monkeypatch, tmp_path):
    snap1, snap2 = pd.Timestamp("2026-11-27T12:00Z"), pd.Timestamp("2026-12-02T12:00Z")
    setup(monkeypatch, {1: snap1, 2: snap1, 3: snap2, 4: None})
    store = {"oddsapi/books/20261127T120000Z.csv.gz": books([
                 {"fpl_fixture": 1, "book": "pinnacle", "odds_h": 2.0, "odds_d": 3.4, "odds_a": 4.0},
                 {"fpl_fixture": 1, "book": "bet365", "odds_h": 1.9, "odds_d": 3.3, "odds_a": 4.2},
                 {"fpl_fixture": 2, "book": "pinnacle", "odds_h": 3.0, "odds_d": 3.2, "odds_a": 2.5}]),
             "oddsapi/books/20261202T120000Z.csv.gz": books([
                 {"fpl_fixture": 3, "book": "pinnacle", "odds_h": 1.5, "odds_d": 4.0, "odds_a": 6.0}])}
    logged = {1: rec(1, 12, "2026-11-27T12:00Z", "2026-11-28T15:00Z"),
              2: rec(2, 12, "2026-11-27T12:00Z", "2026-11-28T15:00Z", p=(0.25, 0.3, 0.45)),
              3: rec(3, 13, "2026-12-02T12:00Z", "2026-12-03T20:00Z"),
              4: rec(4, 13, "2026-12-02T12:00Z", "2026-12-03T20:00Z"),          # no matched snapshot
              9: rec(9, 11, "2026-11-20T12:00Z", "2026-11-21T15:00Z")}          # before GW12: ignored
    out = tmp_path / "paper-bets.json"
    d = pb.build(logged, pd.DataFrame(), {1: "H", 2: "H", 3: "D"}, {}, get=store.__getitem__, out=out)
    b = {x["fixture"]: x for x in d["bets"]}
    assert 9 not in b
    assert b[1]["pick"] == "H" and b[1]["odds"] == [2.0, 3.4, 4.0]          # Pinnacle, not bet365
    assert b[1]["stake"] == 10.0 and b[1]["pnl"] == 10.0                    # won at 2.0
    assert b[2]["pick"] == "A" and b[2]["stake"] == 10.0 and b[2]["pnl"] == -10.0   # same base as fixture 1
    assert b[3]["stake"] == 10.0 and b[3]["pnl"] == -10.0                   # base 100 + 10 - 10
    assert b[4]["status"] == "no price" and b[4]["stake"] is None
    assert d["bankroll"] == 90.0 and d["n_settled"] == 3 and d["n_won"] == 1

    # without the private bucket the stored prices are reused, and a result arriving later settles
    again = pb.build(logged, pd.DataFrame(), {1: "H", 2: "H", 3: "H"}, {}, get=None, out=out)
    assert again["bankroll"] == 105.0


def test_pending_and_missing_bucket(monkeypatch, tmp_path):
    snap = pd.Timestamp("2026-11-27T12:00Z")
    setup(monkeypatch, {1: snap, 2: snap})
    logged = {1: rec(1, 12, "2026-11-27T12:00Z", "2026-11-28T15:00Z"),
              2: rec(2, 12, "2026-11-28T18:00Z", "2026-11-29T15:00Z")}
    store = {"oddsapi/books/20261127T120000Z.csv.gz": books([
        {"fpl_fixture": f, "book": "pinnacle", "odds_h": 2.0, "odds_d": 3.4, "odds_a": 4.0} for f in (1, 2)])}
    out = tmp_path / "paper-bets.json"
    d = pb.build(logged, pd.DataFrame(), {}, {}, get=store.__getitem__, out=out)
    b = {x["fixture"]: x for x in d["bets"]}
    assert b[1]["status"] == "pending" and b[1]["stake"] == 10.0
    assert b[2]["status"] == "pending" and b[2]["stake"] is None   # its base waits for fixture 1's result
    out.unlink()
    assert pb.build(logged, pd.DataFrame(), {}, {}, get=None, out=out) is None and not out.exists()
