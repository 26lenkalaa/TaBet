import json
import xml.dom.minidom

from tabet.digest import build


def _m(gw, home, away, result, ours, market, status="played", kickoff="2026-11-22T15:00:00+00:00"):
    fc = [{"key": "pE", "label": "x", "kind": "logged", "p": ours}] if ours else []
    mk = [{"key": "matched", "label": "m", "p": market}] if market else []
    return {"id": f"{gw}-{home}", "season": "2026-27", "gameweek": gw, "kickoff": kickoff, "home": home, "away": away,
            "status": status, "result": result, "score": "1-0", "forecasts": fc, "market": mk}


def _write(tmp_path, matches, bets=None):
    data = tmp_path / "data"
    data.mkdir()
    (data / "index.json").write_text(json.dumps({"seasons": ["2026-27"], "files": {"2026-27": "m.json"}}))
    (data / "m.json").write_text(json.dumps({"season": "2026-27", "matches": matches}))
    if bets:
        (data / "paper-bets.json").write_text(json.dumps(bets))
    return data


def test_one_item_per_finished_gameweek_with_a_logged_forecast(tmp_path):
    data = _write(tmp_path, [
        _m(10, "A", "B", "H", None, None),                                   # nothing logged: no item
        _m(11, "C", "D", "H", [0.6, 0.2, 0.2], [0.5, 0.25, 0.25]),
        _m(11, "E", "F", "A", [0.5, 0.3, 0.2], None),                        # no market: counted, not compared
        _m(12, "G", "H", "H", [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]),
        _m(12, "I", "J", None, None, None, status="upcoming"),               # unfinished: no item yet
    ])
    text = build(data)
    assert (data / "digest.xml").read_text() == text
    doc = xml.dom.minidom.parseString(text)
    items = doc.getElementsByTagName("item")
    assert [i.getElementsByTagName("guid")[0].firstChild.data for i in items] == ["tabet-2026-27-gw11"]
    title = items[0].getElementsByTagName("title")[0].firstChild.data
    assert "1/2 called" in title and "vs market" in title
    assert build(data) == text            # rebuilding changes nothing


def test_paper_bets_appear_from_their_first_gameweek(tmp_path):
    bets = {"first_gameweek": 12, "start": 100.0, "bankroll": 108.0, "roi": 0.4, "bets": [
        {"gameweek": 12, "status": "won", "pnl": 8.0}]}
    data = _write(tmp_path, [_m(12, "G", "H", "H", [0.5, 0.3, 0.2], [0.5, 0.3, 0.2])], bets)
    desc = xml.dom.minidom.parseString(build(data)).getElementsByTagName("description")[1].firstChild.data
    assert "Paper bets" in desc and "Bankroll $108.00" in desc
