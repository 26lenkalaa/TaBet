"""API-Football (api-sports.io v3) collector: multi-league match data for model training.

Training data only. Nothing on the logging path imports this, and it never calls the odds or
predictions endpoints (they are refused below). It is run by .github/workflows/api-football.yml with
the key in the API_FOOTBALL_KEY repository secret, and its files live on the `api-football-data`
branch, not on main.

    python -m tabet.api_football probe  --data DIR      # plan, allowed seasons, one sample match
    python -m tabet.api_football fetch  --data DIR --budget 50

Budget (kept deliberately modest): the free plan allows 100 requests a day and about 10 a minute.
A run uses at most `--budget` requests (default 50, counting the /status check), never goes within
`RESERVE` of the daily limit, and waits 6.5 s between calls. Fixture details are
fetched 20 at a time with /fixtures?ids=..., which returns events, lineups, team statistics and
player statistics together. Everything is resumable: what has been saved is never requested again.

Layout under DIR:
    fixtures/<league>-<season>.json.gz      one season's fixture list
    details/<league>-<season>/<n>.json.gz   up to 20 full fixtures per file
    fetched_ids.txt                         fixture ids already saved in details/
    blocked.json                            {"<league>-<season>": reason} the plan does not allow
    fetch_log.csv                           one row per request (never the key)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://v3.football.api-sports.io"
RESERVE = 10            # requests left untouched each day
PAUSE_S = 6.5           # about 9 requests a minute, under the free plan's per-minute limit
BATCH = 20              # /fixtures?ids= accepts up to 20 ids
FINISHED = {"FT", "AET", "PEN"}
REFUSED = ("odds", "predictions")   # benchmark data must never become training data

# Order of collection: the Premier League first, then the leagues most useful for a cross-league model.
LEAGUES = [
    (39, "Premier League"), (40, "Championship"), (140, "La Liga"), (135, "Serie A"),
    (78, "Bundesliga"), (61, "Ligue 1"), (88, "Eredivisie"), (94, "Primeira Liga"),
    (144, "Belgian Pro League"), (179, "Scottish Premiership"),
]
SEASONS = list(range(2025, 2009, -1))   # API seasons are the starting year: 2025 = 2025-26


class PlanLimit(Exception):
    """The plan does not cover this request (e.g. a season outside the free range)."""


class DailyLimit(Exception):
    """The daily quota is used up."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Client:
    def __init__(self, key: str, data: Path, budget: int, opener=None, pause: float = PAUSE_S):
        self.key, self.data, self.budget, self.pause = key, data, budget, pause
        self.used = 0
        self._last = 0.0
        if opener is None:
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
            opener = lambda req: urllib.request.urlopen(req, timeout=60, context=ctx)  # noqa: E731
        self._open = opener

    def get(self, endpoint: str, counts: bool = True, **params) -> dict:
        if any(r in endpoint for r in REFUSED):
            raise ValueError(f"refusing /{endpoint}: odds and predictions are a benchmark, never training data")
        if counts and self.used >= self.budget:
            raise DailyLimit(f"run budget of {self.budget} requests reached")
        wait = self._last + self.pause - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        url = f"{BASE}/{endpoint}" + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"x-apisports-key": self.key, "Accept": "application/json"})
        body, status, err = {}, 0, ""
        try:
            with self._open(req) as r:
                status = r.status
                body = json.loads(r.read().decode())
        except Exception as e:  # network or HTTP error: logged, then re-raised
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._last = time.monotonic()
            if counts:
                self.used += 1
            errors = body.get("errors") if isinstance(body, dict) else None
            self._log(endpoint, params, status, body, err or (json.dumps(errors) if errors else ""))
        errors = body.get("errors")
        if errors:
            text = json.dumps(errors).lower()
            if "request limit" in text or "requests" in text and "limit" in text:
                raise DailyLimit(text)
            if "plan" in text or "season" in text or "access" in text:
                raise PlanLimit(json.dumps(errors))
            raise RuntimeError(f"API error on /{endpoint}: {errors}")
        return body

    def _log(self, endpoint, params, status, body, error):
        path = self.data / "fetch_log.csv"
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["utc", "endpoint", "params", "http", "results", "error"])
            shown = {k: v for k, v in params.items()}
            if "ids" in shown:
                shown["ids"] = f"{len(str(shown['ids']).split('-'))} ids"
            w.writerow([_now(), endpoint, json.dumps(shown), status,
                        body.get("results", "") if isinstance(body, dict) else "", error[:300]])


# ---------------------------------------------------------------- storage helpers

def _write_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def _read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _fetched(data: Path) -> set[int]:
    p = data / "fetched_ids.txt"
    return {int(x) for x in p.read_text().split()} if p.exists() else set()


def _blocked(data: Path) -> dict:
    p = data / "blocked.json"
    return json.loads(p.read_text()) if p.exists() else {}


def quota(client: Client) -> tuple[int, int]:
    """(requests used today, daily limit) from /status, which does not count against the quota."""
    s = client.get("status")["response"]
    req = s.get("requests", {}) if isinstance(s, dict) else {}
    return int(req.get("current", 0)), int(req.get("limit_day", 100))


# ---------------------------------------------------------------- modes

def probe(client: Client) -> dict:
    """A few requests that show what the plan allows. Prints and returns a summary, never the key."""
    out = {}
    s = client.get("status")["response"]
    out["plan"] = (s.get("subscription") or {}).get("plan") if isinstance(s, dict) else None
    out["requests_today"] = (s.get("requests") or {}) if isinstance(s, dict) else {}
    lg = client.get("leagues", id=39)["response"]
    out["premier_league_seasons"] = [
        {"season": x["year"], "lineups": x["coverage"]["fixtures"].get("lineups"),
         "team_stats": x["coverage"]["fixtures"].get("statistics_fixtures"),
         "player_stats": x["coverage"]["fixtures"].get("statistics_players")}
        for x in (lg[0]["seasons"] if lg else [])]
    allowed = []
    for season in (2023, 2021):        # at most 2 season lists + 1 sample match: about 5 calls in all
        if allowed:
            break
        try:
            fx = client.get("fixtures", league=39, season=season)
            allowed.append(season)
            out.setdefault("sample", {"season": season, "fixtures": fx.get("results")})
            if "sample_fixture" not in out:
                done = [f for f in fx["response"] if f["fixture"]["status"]["short"] in FINISHED]
                if done:
                    d = client.get("fixtures", ids=str(done[0]["fixture"]["id"]))["response"][0]
                    try:
                        players = ((d.get("players") or [{}])[0].get("players") or [{}])[0]
                        out["sample_fixture"] = {
                            "match": f'{d["teams"]["home"]["name"]} v {d["teams"]["away"]["name"]}',
                            "has_lineups": bool(d.get("lineups")),
                            "team_stat_names": [x["type"] for x in (d.get("statistics") or [{}])[0].get("statistics", [])],
                            "player_stat_groups": list(players.get("statistics", [{}])[0].keys()),
                        }
                    except (IndexError, KeyError, AttributeError) as e:
                        out["sample_fixture"] = {"unexpected_shape": str(e)[:200], "keys": list(d)}
        except PlanLimit as e:
            out.setdefault("blocked", {})[season] = str(e)[:200]
    out["allowed_seasons_tried"] = allowed
    out["requests_used_by_probe"] = client.used
    return out


def fetch(client: Client, leagues=LEAGUES, seasons=SEASONS) -> dict:
    data = client.data
    used, limit = quota(client)
    client.budget = min(client.budget, max(0, limit - RESERVE - used) + client.used)
    fetched, blocked = _fetched(data), _blocked(data)
    summary = {"start_used_today": used, "limit_day": limit, "budget": client.budget, "new_fixtures": 0, "lists": 0}
    try:
        for league, _name in leagues:
            for season in seasons:
                tag = f"{league}-{season}"
                if tag in blocked:
                    continue
                lst = data / "fixtures" / f"{tag}.json.gz"
                if not lst.exists():
                    try:
                        body = client.get("fixtures", league=league, season=season)
                    except PlanLimit as e:
                        blocked[tag] = str(e)[:300]
                        (data / "blocked.json").write_text(json.dumps(blocked, indent=1, sort_keys=True))
                        continue
                    _write_gz(lst, body)
                    summary["lists"] += 1
                todo = [f["fixture"]["id"] for f in _read_gz(lst)["response"]
                        if f["fixture"]["status"]["short"] in FINISHED and f["fixture"]["id"] not in fetched]
                for i in range(0, len(todo), BATCH):
                    ids = todo[i:i + BATCH]
                    body = client.get("fixtures", ids="-".join(map(str, ids)))
                    got = [f["fixture"]["id"] for f in body.get("response", [])]
                    n = len(list((data / "details" / tag).glob("*.json.gz"))) if (data / "details" / tag).exists() else 0
                    _write_gz(data / "details" / tag / f"{n:04d}.json.gz", body)
                    with (data / "fetched_ids.txt").open("a") as fh:
                        fh.writelines(f"{x}\n" for x in got)
                    fetched.update(got)
                    summary["new_fixtures"] += len(got)
    except DailyLimit as e:
        summary["stopped"] = str(e)[:200]
    summary["requests_used"] = client.used
    summary["fixtures_total"] = len(fetched)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["probe", "fetch"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=50)
    a = ap.parse_args(argv)
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        print("API_FOOTBALL_KEY is not set; nothing to do.")
        return 0
    client = Client(key, a.data, a.budget)
    result = probe(client) if a.mode == "probe" else fetch(client)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
