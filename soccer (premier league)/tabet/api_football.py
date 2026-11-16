"""API-Football (api-sports.io v3) collector: multi-league match data for model training.

Training data only. Nothing on the logging path imports this, and it never calls the odds or
predictions endpoints (they are refused below). It runs on GitHub Actions
(.github/workflows/api-football.yml). The API key and the Supabase secret key live only in
repository secrets, and everything collected goes straight to the backend:

  * raw responses, gzipped  -> Supabase Storage, private bucket `api-football`
  * what has been collected -> private tables apif_lists, apif_fixtures, apif_blocked, apif_requests
    (backend/supabase/migrations/0003_api_football.sql)

Nothing is written to git or to anyone's laptop.

    python -m tabet.api_football probe                # plan, allowed seasons, one sample match (~5 calls)
    python -m tabet.api_football fetch --budget 50

Budget (kept deliberately modest): the free plan allows 100 requests a day and about 10 a minute.
A run uses at most `--budget` requests (default 50, counting the /status check), never goes within
`RESERVE` of the daily limit, and waits 6.5 s between calls. Fixture details come 20 at a time from
/fixtures?ids=..., which returns events, lineups, team statistics and player statistics together.
Everything is resumable: a fixture already stored is never requested again.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://v3.football.api-sports.io"
RESERVE = 10            # requests left untouched each day
PAUSE_S = 6.5           # about 9 requests a minute, under the free plan's per-minute limit
BATCH = 20              # /fixtures?ids= accepts up to 20 ids
FINISHED = {"FT", "AET", "PEN"}
REFUSED = ("odds", "predictions")   # benchmark data must never become training data
BUCKET = "api-football"

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


def _ssl():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _default_opener():
    ctx = _ssl()
    return lambda req: urllib.request.urlopen(req, timeout=60, context=ctx)


# ---------------------------------------------------------------- where the data goes

class MemoryStore:
    """In-memory store with the same interface as SupabaseStore (tests, dry runs)."""

    def __init__(self):
        self.objects, self.lists, self.fixtures, self.blocked_, self.requests = {}, {}, {}, {}, []

    def put(self, path, obj):
        self.objects[path] = gzip.compress(json.dumps(obj, separators=(",", ":")).encode())

    def get(self, path):
        return json.loads(gzip.decompress(self.objects[path]))

    def list_path(self, league, season):
        return self.lists.get((league, season))

    def add_list(self, league, season, path, n):
        self.lists[(league, season)] = path

    def fetched_ids(self):
        return set(self.fixtures)

    def add_fixtures(self, league, season, path, ids):
        for i in ids:
            self.fixtures.setdefault(i, path)

    def blocked(self):
        return set(self.blocked_)

    def block(self, league, season, reason):
        self.blocked_[(league, season)] = reason

    def log(self, row):
        self.requests.append(row)


class SupabaseStore:
    """Supabase Storage (raw gzipped responses) + PostgREST tables (bookkeeping), via the secret key."""

    def __init__(self, url: str, key: str, opener=None):
        self.url, self.key = url.rstrip("/"), key
        self._open = opener or _default_opener()

    def _req(self, method, path, body=None, headers=None):
        h = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
        h.update(headers or {})
        req = urllib.request.Request(self.url + path, data=body, method=method, headers=h)
        try:
            with self._open(req) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase {method} {path.split('?')[0]}: {e.code} {e.read()[:300]!r}") from None

    def _rows(self, table, select, **filters):
        out, offset = [], 0
        while True:
            q = urllib.parse.urlencode({"select": select, **filters, "limit": 1000, "offset": offset})
            page = json.loads(self._req("GET", f"/rest/v1/{table}?{q}"))
            out += page
            if len(page) < 1000:
                return out
            offset += 1000

    def _insert(self, table, rows, upsert=False):
        if rows:
            self._req("POST", f"/rest/v1/{table}", json.dumps(rows).encode(), {
                "Content-Type": "application/json",
                "Prefer": "return=minimal," + ("resolution=merge-duplicates" if upsert else "resolution=ignore-duplicates")})

    def put(self, path, obj):
        data = gzip.compress(json.dumps(obj, separators=(",", ":")).encode())
        self._req("POST", f"/storage/v1/object/{BUCKET}/{urllib.parse.quote(path)}", data,
                  {"Content-Type": "application/gzip", "x-upsert": "true"})

    def get(self, path):
        return json.loads(gzip.decompress(self._req("GET", f"/storage/v1/object/{BUCKET}/{urllib.parse.quote(path)}")))

    def list_path(self, league, season):
        rows = self._rows("apif_lists", "object_path", league=f"eq.{league}", season=f"eq.{season}")
        return rows[0]["object_path"] if rows else None

    def add_list(self, league, season, path, n):
        self._insert("apif_lists", [{"league": league, "season": season, "object_path": path, "n_fixtures": n}], upsert=True)

    def fetched_ids(self):
        return {r["fixture_id"] for r in self._rows("apif_fixtures", "fixture_id")}

    def add_fixtures(self, league, season, path, ids):
        self._insert("apif_fixtures", [{"fixture_id": i, "league": league, "season": season, "object_path": path} for i in ids])

    def blocked(self):
        return {(r["league"], r["season"]) for r in self._rows("apif_blocked", "league,season")}

    def block(self, league, season, reason):
        self._insert("apif_blocked", [{"league": league, "season": season, "reason": reason[:500]}], upsert=True)

    def log(self, row):
        self._insert("apif_requests", [row])


# ---------------------------------------------------------------- the API client

class Client:
    def __init__(self, key: str, store, budget: int, opener=None, pause: float = PAUSE_S):
        self.key, self.store, self.budget, self.pause = key, store, budget, pause
        self.used = 0
        self._last = 0.0
        self._open = opener or _default_opener()

    def get(self, endpoint: str, **params) -> dict:
        if any(r in endpoint for r in REFUSED):
            raise ValueError(f"refusing /{endpoint}: odds and predictions are a benchmark, never training data")
        if self.used >= self.budget:
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
            self.used += 1
            errors = body.get("errors") if isinstance(body, dict) else None
            shown = dict(params)
            if "ids" in shown:
                shown["ids"] = f"{len(str(shown['ids']).split('-'))} ids"
            self.store.log({"endpoint": endpoint, "params": shown, "http": status,
                            "results": body.get("results") if isinstance(body, dict) else None,
                            "error": (err or (json.dumps(errors) if errors else ""))[:300] or None})
        errors = body.get("errors")
        if errors:
            text = json.dumps(errors).lower()
            if "request limit" in text or ("requests" in text and "limit" in text):
                raise DailyLimit(text)
            if "plan" in text or "season" in text or "access" in text:
                raise PlanLimit(json.dumps(errors))
            raise RuntimeError(f"API error on /{endpoint}: {errors}")
        return body


def quota(client: Client) -> tuple[int, int]:
    """(requests used today, daily limit) from /status."""
    s = client.get("status")["response"]
    req = s.get("requests", {}) if isinstance(s, dict) else {}
    return int(req.get("current", 0)), int(req.get("limit_day", 100))


# ---------------------------------------------------------------- modes

def probe(client: Client) -> dict:
    """About 5 requests that show what the plan allows. Returns a summary, never the key."""
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
    for season in (2023, 2021):        # at most 2 season lists + 1 sample match
        if allowed:
            break
        try:
            fx = client.get("fixtures", league=39, season=season)
        except PlanLimit as e:
            out.setdefault("blocked", {})[season] = str(e)[:200]
            continue
        allowed.append(season)
        out["sample"] = {"season": season, "fixtures": fx.get("results")}
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
    out["allowed_seasons_tried"] = allowed
    out["requests_used_by_probe"] = client.used
    return out


def fetch(client: Client, leagues=LEAGUES, seasons=SEASONS) -> dict:
    store = client.store
    used, limit = quota(client)
    client.budget = min(client.budget, max(0, limit - RESERVE - used) + client.used)
    fetched, blocked = store.fetched_ids(), store.blocked()
    summary = {"start_used_today": used, "limit_day": limit, "budget": client.budget, "new_fixtures": 0, "lists": 0}
    try:
        for league, _name in leagues:
            for season in seasons:
                if (league, season) in blocked:
                    continue
                path = store.list_path(league, season)
                if path is None:
                    try:
                        body = client.get("fixtures", league=league, season=season)
                    except PlanLimit as e:
                        store.block(league, season, str(e))
                        blocked.add((league, season))
                        continue
                    path = f"fixtures/{league}-{season}.json.gz"
                    store.put(path, body)
                    store.add_list(league, season, path, len(body.get("response", [])))
                    summary["lists"] += 1
                    listing = body
                else:
                    listing = store.get(path)
                todo = [f["fixture"]["id"] for f in listing["response"]
                        if f["fixture"]["status"]["short"] in FINISHED and f["fixture"]["id"] not in fetched]
                for i in range(0, len(todo), BATCH):
                    ids = todo[i:i + BATCH]
                    body = client.get("fixtures", ids="-".join(map(str, ids)))
                    got = [f["fixture"]["id"] for f in body.get("response", [])]
                    obj = f"details/{league}-{season}/{ids[0]}.json.gz"
                    store.put(obj, body)
                    store.add_fixtures(league, season, obj, got)
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
    ap.add_argument("--budget", type=int, default=50)
    a = ap.parse_args(argv)
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    url, skey = os.environ.get("SUPABASE_URL", "").strip(), os.environ.get("SUPABASE_SECRET_KEY", "").strip()
    if not key:
        print("API_FOOTBALL_KEY is not set; nothing to do.")
        return 0
    if a.mode == "fetch" and not (url and skey):
        print("SUPABASE_URL / SUPABASE_SECRET_KEY are not set; fetch stores everything in Supabase, so it stops here.")
        return 1
    store = SupabaseStore(url, skey) if url and skey else MemoryStore()
    client = Client(key, store, a.budget)
    result = probe(client) if a.mode == "probe" else fetch(client)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
