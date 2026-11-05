"""Predicted lineups for upcoming fixtures from public pages, plus our own projection.

Sources (checked 2026-11-05: plain HTTP, robots.txt allows these paths):
  ffs       fantasyfootballscout.co.uk/team-news : one predicted XI per team.
            Player photos use Premier League photo ids, which equal FPL `code`,
            so players join to our data without name matching.
  rotowire  rotowire.com/soccer/lineups.php : predicted or confirmed XI per
            match, with full names (matched to FPL by name within the team).
  fpl_proj  our own: the most-started GK + 10 outfielders by start probability
            from preceding selections, minus players FPL flags as unavailable.

FotMob is not used: its robots.txt disallows /api/* for all agents.

Every fetch is stored with its retrieval time under data/raw/lineups/, and
parsed snapshots are written to results/lineup_snapshots/, so each source's
accuracy can be scored against the confirmed XIs once matches are played.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .fpl import _get
from .teams import try_resolve_team

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "lineups"
SNAP = ROOT / "results" / "lineup_snapshots"
URLS = {
    "ffs": "https://www.fantasyfootballscout.co.uk/team-news",
    "rotowire": "https://www.rotowire.com/soccer/lineups.php",
}
POS_MAP = {"GK": "GK", "D": "DEF", "M": "MID", "F": "FWD", "W": "MID", "A": "MID", "S": "FWD", "C": "FWD"}


_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "đ": "d", "Đ": "D",
                           "ł": "l", "Ł": "L", "ı": "i"})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def fetch(source: str, stamp: str) -> str:
    path = RAW / source / f"{stamp}.html"
    _get(URLS[source], path, pause=2.0)
    return path.read_text(errors="replace")


def parse_ffs(page: str) -> pd.DataFrame:
    rows = []
    for blk in re.split(r'class="team-news-item"', page)[1:]:
        name = re.search(r"<h2[^>]*>\s*([^<]+?)\s*</h2>", blk)
        team = try_resolve_team(html.unescape(name.group(1))) if name else None
        upd = re.search(r"(?i)last updated:?\s*([^<]{4,40})", blk)
        upd_txt = upd.group(1).strip() if upd else None
        pitch = blk.split("scout-picks-pitch", 1)[-1].split("</div>", 1)[0] if "scout-picks-pitch" in blk else ""
        for row, ul in re.findall(r'<ul class="row-(\d)">(.*?)</ul>', pitch, re.S):
            for code, pname in re.findall(r"players/110x140/(\d+)\.png.*?player-name[^>]*>([^<]+)<", ul, re.S):
                rows.append({"team_id": team, "player_code": int(code), "source_name": html.unescape(pname).strip(),
                             "row": int(row), "source_updated": upd_txt})
    df = pd.DataFrame(rows)
    if len(df):
        df["position"] = df["row"].map(lambda r: "GK" if r == 1 else None)  # outfield roles come from FPL
        df["status"] = "predicted"
    return df


def parse_rotowire(page: str, fpl_players: pd.DataFrame) -> pd.DataFrame:
    """fpl_players: columns player_code, team_id, first_name, second_name, web_name."""
    rows = []
    for box in re.split(r'class="lineup__box"', page)[1:]:
        teams = re.findall(r'class="lineup__mteam is-(home|visit)">\s*([^<]+?)\s*<', box)
        for side, ul in re.findall(r'<ul class="lineup__list is-(home|visit)">(.*?)</ul>', box, re.S):
            tname = dict(teams).get(side)
            team = try_resolve_team(html.unescape(tname)) if tname else None
            status = "confirmed" if "is-confirmed" in ul else "predicted"
            starters = ul.split("lineup__title", 1)[0]  # injuries/bench follow the title
            for pos, full in re.findall(r'lineup__pos[^>]*>\s*([A-Z]+)\s*<.*?<a title="([^"]+)"', starters, re.S):
                rows.append({"team_id": team, "source_name": html.unescape(full), "status": status,
                             "position": "GK" if pos == "GK" else POS_MAP.get(pos[0])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["player_code"] = [_match_name(n, t, fpl_players) for n, t in zip(df["source_name"], df["team_id"])]
    return df


def _match_name(full: str, team: str, fp: pd.DataFrame) -> int | None:
    """Match a source name to one FPL player in the same team: exact full name,
    then exact web name, then the unique best token overlap (FPL often uses full
    legal names, e.g. 'David Raya Martin', or single names like 'Alisson')."""
    cand = fp[fp["team_id"] == team]
    if cand.empty:
        return None
    n = _norm(full)
    toks = set(n.split())
    full_names = cand["first_name"].map(_norm) + " " + cand["second_name"].map(_norm)
    for hit in (cand[full_names == n], cand[cand["web_name"].map(_norm) == n]):
        if len(hit) == 1:
            return int(hit["player_code"].iloc[0])
    score = [len(toks & (set(fn.split()) | set(_norm(w).split()))) for fn, w in zip(full_names, cand["web_name"])]
    best = max(score)
    if best >= 1 and score.count(best) == 1:
        return int(cand["player_code"].iloc[score.index(best)])
    return None


def snapshot(fpl_players: pd.DataFrame, stamp: str | None = None) -> pd.DataFrame:
    """Fetch and parse every source; save a combined snapshot CSV."""
    stamp = stamp or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%MZ")
    parts = []
    for src in URLS:
        try:
            page = fetch(src, stamp)
        except Exception as e:  # a source being down must not stop the others
            print(f"{src}: fetch failed ({e})")
            continue
        df = parse_ffs(page) if src == "ffs" else parse_rotowire(page, fpl_players)
        parts.append(df.assign(source=src, retrieved_utc=stamp))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    SNAP.mkdir(parents=True, exist_ok=True)
    out.to_csv(SNAP / f"lineups_{stamp}.csv", index=False)
    return out


def fpl_players_now(boot: dict) -> pd.DataFrame:
    teams = {t["id"]: try_resolve_team(t["name"]) for t in boot["teams"]}
    return pd.DataFrame([{"player_code": e["code"], "team_id": teams[e["team"]], "first_name": e["first_name"],
                          "second_name": e["second_name"], "web_name": e["web_name"],
                          "position": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(e["element_type"]),
                          "status": e["status"], "chance_next": e["chance_of_playing_next_round"],
                          "news": e["news"]} for e in boot["elements"]])


def project_xi(hist: pd.DataFrame, cutoff: pd.Timestamp, teams: list[str], half_life: float = 120.0,
               unavailable: set[int] | None = None, current_team: dict[int, str] | None = None) -> pd.DataFrame:
    """Our own projected XI per team: the most-started GK plus the 10 outfielders
    with the highest start probability from preceding selections. A player
    counts for a team only if his latest appearance (or `current_team`, when
    given) is for that team; `unavailable` player codes are skipped."""
    from .lineup import start_probabilities
    sp = start_probabilities(hist, cutoff, half_life).reset_index()
    latest = hist.sort_values("kickoff_utc").groupby("player_code")["team_id"].last()
    owner = current_team if current_team is not None else latest.to_dict()
    sp = sp[sp["player_code"].map(owner) == sp["team_id"]]
    if unavailable:
        sp = sp[~sp["player_code"].isin(unavailable)]
    rows = []
    for t in teams:
        s = sp[sp["team_id"] == t].sort_values("start_prob", ascending=False)
        xi = pd.concat([s[s["position"] == "GK"].head(1), s[s["position"] != "GK"].head(10)])
        rows += [{"team_id": t, "player_code": int(c), "position": pos, "start_prob": float(pr)}
                 for c, pos, pr in zip(xi["player_code"], xi["position"], xi["start_prob"])]
    return pd.DataFrame(rows)
