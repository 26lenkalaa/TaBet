"""Weekly digest: one RSS item per finished gameweek, written to site/data/digest.xml (committed by the logger with the rest of site/data).

tabet/newsletter.py emails each new item to the newsletter list (sign-ups from the site's box). The feed
is also a public RSS file, so a reader or a Power Automate RSS flow can follow it.

Built from the site's own data files (matches-<season>.json and paper-bets.json), after site_data writes
them. Descriptive only: nothing here feeds a forecast. A gameweek appears once every match in it is played
and at least one has a logged forecast. Each item's date is fixed (4 h after the gameweek's last kickoff),
so rebuilding the site does not change the file unless a new gameweek finishes.
"""
from __future__ import annotations

import json
import math
from email.utils import format_datetime
from html import escape
from pathlib import Path

import pandas as pd

SITE = "https://ta-bet-ml-model-edition.vercel.app"
OUT_NAME = "digest.xml"
RESULT = ("H", "D", "A")
MAX_ITEMS = 20


def _ours(m: dict) -> dict | None:
    """The logged forecast the site headlines: pE (predicted lineups), else p0."""
    logged = {f["key"]: f for f in m["forecasts"] if f.get("kind") == "logged"}
    return logged.get("pE") or logged.get("p0")


def _matched(m: dict) -> dict | None:
    """The market at our forecast time: the only fair comparison (closing odds know more than we did)."""
    return next((x for x in m["market"] if x["key"] == "matched"), None)


def _loss(p: list[float], result: str) -> float:
    return -math.log(max(float(p[RESULT.index(result)]), 1e-12))


def _rows(matches: list[dict]) -> list[dict]:
    """Scored rows for logged, played matches."""
    out = []
    for m in matches:
        f = _ours(m)
        if f is None or m.get("status") != "played" or m.get("result") not in RESULT:
            continue
        mk = _matched(m)
        pick = RESULT[max(range(3), key=lambda i: f["p"][i])]
        out.append({"m": m, "p": f["p"], "key": f["key"], "pick": pick, "hit": pick == m["result"],
                    "loss": _loss(f["p"], m["result"]),
                    "mloss": _loss(mk["p"], m["result"]) if mk else None, "mp": mk["p"] if mk else None})
    return out


def _summary(rows: list[dict]) -> dict:
    both = [r for r in rows if r["mloss"] is not None]
    return {"n": len(rows), "hits": sum(r["hit"] for r in rows), "n_both": len(both),
            "ours": sum(r["loss"] for r in both) / len(both) if both else None,
            "market": sum(r["mloss"] for r in both) / len(both) if both else None,
            "beat": sum(r["loss"] < r["mloss"] for r in both)}


def _pct(p: list[float]) -> str:
    return " / ".join(f"{round(x * 100)}%" for x in p)


def _verdict(s: dict) -> str:
    if s["ours"] is None:
        return "No market price at our forecast time, so no comparison."
    d = s["ours"] - s["market"]
    side = "better than" if d < 0 else "worse than" if d > 0 else "level with"
    return (f"Log loss {s['ours']:.3f} vs the market's {s['market']:.3f} ({side} the market by {abs(d):.3f}; "
            f"lower is better), over {s['n_both']} match{'es' * (s['n_both'] != 1)}. "
            f"We beat the market's forecast in {s['beat']} of them.")


def _bets_html(bets: dict | None, gw: int) -> str:
    if not bets or gw < bets.get("first_gameweek", 10 ** 9):
        return ""
    week = [b for b in bets["bets"] if b["gameweek"] == gw and b["status"] in ("won", "lost")]
    if not week:
        return ""
    pnl = sum(b["pnl"] for b in week)
    roi = f", ROI {bets['roi'] * 100:+.1f}%" if bets.get("roi") is not None else ""
    return (f"<h3>Paper bets (imaginary money)</h3><p>{sum(b['status'] == 'won' for b in week)} of {len(week)} won "
            f"this week, {'+' if pnl >= 0 else '-'}${abs(pnl):.2f}. Bankroll ${bets['bankroll']:.2f} "
            f"(started at ${bets['start']:.0f}{roi}).</p>")


def _item_html(gw: int, rows: list[dict], season_rows: list[dict], bets: dict | None) -> str:
    s, t = _summary(rows), _summary(season_rows)
    trs = []
    for r in rows:
        m = r["m"]
        trs.append(
            f"<tr><td>{escape(m['home'])} {escape(m.get('score') or '')} {escape(m['away'])}</td>"
            f"<td>{_pct(r['p'])}</td><td>{_pct(r['mp']) if r['mp'] else '&ndash;'}</td>"
            f"<td>{r['pick']} {'&#10003;' if r['hit'] else '&#10007;'}</td></tr>")
    return (
        f"<p><b>{s['hits']} of {s['n']}</b> results called correctly (most likely outcome). {_verdict(s)}</p>"
        "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">"
        "<tr><th>Match</th><th>TaBet (H / D / A)</th><th>Market</th><th>Our pick</th></tr>"
        + "".join(trs) + "</table>"
        + _bets_html(bets, gw)
        + f"<h3>Season so far</h3><p>{t['hits']} of {t['n']} called. {_verdict(t)}</p>"
        f"<p><a href=\"{SITE}/matches.html\">All matches on the site</a></p>")


def build(site_data: Path) -> str | None:
    """Write site_data/digest.xml from the season files in site_data; returns the feed text."""
    index = json.loads((site_data / "index.json").read_text())
    season = index["seasons"][-1]
    matches = json.loads((site_data / index["files"][season]).read_text())["matches"]
    bets_file = site_data / "paper-bets.json"
    bets = json.loads(bets_file.read_text()) if bets_file.exists() else None

    by_gw: dict[int, list[dict]] = {}
    for m in matches:
        if m.get("gameweek") is not None:
            by_gw.setdefault(int(m["gameweek"]), []).append(m)
    items, season_rows = [], []
    for gw in sorted(by_gw):
        ms = by_gw[gw]
        rows = _rows(ms)
        season_rows = season_rows + rows
        if not rows or any(m.get("status") != "played" for m in ms):
            continue
        done = max(pd.Timestamp(m["kickoff"]) for m in ms) + pd.Timedelta(hours=4)
        s = _summary(rows)
        title = f"TaBet {season} gameweek {gw}: {s['hits']}/{s['n']} called" + (
            f", log loss {s['ours']:.3f} vs market {s['market']:.3f}" if s["ours"] is not None else "")
        items.append((done, gw, title, _item_html(gw, rows, season_rows, bets)))

    items = items[-MAX_ITEMS:][::-1]   # newest first
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<rss version="2.0"><channel>',
             "<title>TaBet weekly digest</title>", f"<link>{SITE}/</link>",
             "<description>Premier League forecasts against the market, one item per finished gameweek.</description>"]
    if items:
        parts.append(f"<lastBuildDate>{format_datetime(items[0][0].to_pydatetime())}</lastBuildDate>")
    for done, gw, title, html in items:
        parts += ["<item>", f"<title>{escape(title)}</title>", f"<link>{SITE}/matches.html</link>",
                  f'<guid isPermaLink="false">tabet-{season}-gw{gw}</guid>',
                  f"<pubDate>{format_datetime(done.to_pydatetime())}</pubDate>",
                  f"<description>{escape(html)}</description>", "</item>"]
    parts.append("</channel></rss>")
    text = "\n".join(parts) + "\n"
    (site_data / OUT_NAME).write_text(text)
    return text
