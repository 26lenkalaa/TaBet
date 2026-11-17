"""API-Football (api-sports.io v3) collector: multi-league match data for model training.

Training data only. Nothing on the logging path imports this, and it never calls the odds or
predictions endpoints (they are refused below). It runs on GitHub Actions
(.github/workflows/api-football.yml). The API key and the Supabase secret key live only in
repository secrets, and everything collected goes straight to the backend:

  * raw responses, gzipped  -> Supabase Storage, private bucket `api-football`
  * what has been collected -> private tables apif_lists, apif_fixtures, apif_blocked, apif_requests
    (backend/supabase/migrations/0003_api_football.sql)

Nothing is written to git or to anyone's laptop.

    python -m tabet.api_football probe                # plan, allowed seasons, one sample match (5 calls)
    python -m tabet.api_football coverage             # plan + what each league offers by season (1 + 10 calls)
    python -m tabet.api_football fetch --budget 50

Budget: a run uses at most `--budget` requests (counting the /status check) and never goes within
`RESERVE` of the daily limit. fetch reads the plan from /status:
  * free plan (100 a day, about 10 a minute): one match a request (/fixtures?id=), 6.5 s apart;
  * paid plan (daily limit above 100): 20 matches a request (/fixtures?ids=), PAID_PAUSE_S apart, and
    seasons the free plan refused are tried again (and unblocked if now allowed).
When a paid month ends the limit drops back to 100 and fetch returns to the free settings by itself.
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
PAID_PAUSE_S = 0.4      # paid plans allow 300 requests a minute; this stays well under it
FREE_LIMIT = 100        # a daily limit above this means a paid plan
BATCH = 20              # /fixtures?ids= accepts up to 20 ids
FINISHED = {"FT", "AET", "PEN"}
REFUSED = ("odds", "predictions")   # benchmark data must never become training data
BUCKET = "api-football"

# Order of collection. The Premier League comes last: 2022-26 is already covered by FPL, which has
# player-level xG. The other big leagues come first (team xG is the cross-league analogue of V1x),
# then the Championship (promoted teams, brief §5.21).
LEAGUES = [
    (140, "La Liga"), (135, "Serie A"), (78, "Bundesliga"), (61, "Ligue 1"),
    # addendum DXI's test leagues come before the Championship, which that test does not use
    (88, "Eredivisie"), (94, "Primeira Liga"), (144, "Belgian Pro League"), (179, "Scottish Premiership"),
    (40, "Championship"), (39, "Premier League"),
]
# API seasons are the starting year (2024 = 2024-25). Coverage check of 2026-11-17 (Pro plan): every
# league has lineups and team and player statistics from 2015 (Belgium and Scotland: player
# statistics from 2020); team expected goals start in 2022 (samples of 2015-2021 had none). The
# season in progress (2026) is not collected: E0 2026-27 is the live test. On the free plan (2022-2024
# only) the other seasons' lists are refused once each and remembered as blocked.
SEASONS = list(range(2025, 2014, -1))


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

    def unblock(self, league, season):
        self.blocked_.pop((league, season), None)

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

    def unblock(self, league, season):
        self._req("DELETE", f"/rest/v1/apif_blocked?league=eq.{int(league)}&season=eq.{int(season)}")

    def log(self, row):
        self._insert("apif_requests", [row])


# ---------------------------------------------------------------- the API client

class Client:
    def __init__(self, key: str, store, budget: int, opener=None, pause: float = PAUSE_S, batch: bool = False):
        # batch=True uses /fixtures?ids= (20 matches a request). The free plan refuses that parameter
        # ("Free plans do not have access to the Ids parameter"), so the default is one match a request.
        self.key, self.store, self.budget, self.pause, self.batch = key, store, budget, pause, batch
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
    """Five requests that show what the plan allows. Returns a summary, never the key.

    status; Premier League coverage by season; the current season's fixture list (its error message
    names the plan's allowed seasons if it is refused); the 2023 list (kept, so fetch never asks again);
    and one match by id, to check that a single request returns lineups and team and player stats."""
    out, store = {}, client.store
    s = client.get("status")["response"]
    out["plan"] = (s.get("subscription") or {}).get("plan") if isinstance(s, dict) else None
    out["requests_today"] = (s.get("requests") or {}) if isinstance(s, dict) else {}
    lg = client.get("leagues", id=39)["response"]
    out["premier_league_coverage"] = {
        x["year"]: {k: x["coverage"]["fixtures"].get(v) for k, v in
                    (("lineups", "lineups"), ("team_stats", "statistics_fixtures"), ("player_stats", "statistics_players"))}
        for x in (lg[0]["seasons"] if lg else []) if x["year"] >= 2015}
    out["seasons"] = {}
    listing = None
    for season in (2025, 2023):
        try:
            body = client.get("fixtures", league=39, season=season)
            out["seasons"][season] = f"allowed ({body.get('results')} fixtures)"
            if listing is None:
                listing = body
                path = f"fixtures/39-{season}.json.gz"
                store.put(path, body)
                store.add_list(39, season, path, len(body.get("response", [])))
        except PlanLimit as e:
            out["seasons"][season] = f"blocked: {e}"[:250]
            store.block(39, season, str(e))
    done = [f for f in (listing or {}).get("response", []) if f["fixture"]["status"]["short"] in FINISHED]
    if done:
        try:
            d = client.get("fixtures", id=done[0]["fixture"]["id"])["response"][0]
            players = ((d.get("players") or [{}])[0].get("players") or [{}])[0]
            out["sample_match"] = {
                "match": f'{d["teams"]["home"]["name"]} v {d["teams"]["away"]["name"]}',
                "has_lineups": bool(d.get("lineups")),
                "team_stats": [x["type"] for x in (d.get("statistics") or [{}])[0].get("statistics", [])],
                "player_stat_groups": list(players.get("statistics", [{}])[0].keys()),
            }
        except PlanLimit as e:
            out["sample_match"] = f"blocked: {e}"[:250]
        except (IndexError, KeyError, AttributeError) as e:
            out["sample_match"] = {"unexpected_shape": str(e)[:200]}
    out["requests_used_by_probe"] = client.used
    return out


def xgcheck(client: Client, league: int = 39, season: int = 2022) -> dict:
    """Two calls: does an older season's match data include team expected goals? Keeps the fixture
    list it downloads (fetch reuses it) and reports the team statistics one finished match returns."""
    store = client.store
    out = {"league": league, "season": season}
    body = client.get("fixtures", league=league, season=season)
    path = f"fixtures/{league}-{season}.json.gz"
    store.put(path, body)
    store.add_list(league, season, path, len(body.get("response", [])))
    done = [f for f in body.get("response", []) if f["fixture"]["status"]["short"] in FINISHED]
    out["fixtures"] = len(body.get("response", []))
    if done:
        d = client.get("fixtures", id=done[len(done) // 2]["fixture"]["id"])["response"][0]
        stats = {x["type"]: x.get("value") for x in (d.get("statistics") or [{}])[0].get("statistics", [])}
        out["sample_match"] = f'{d["teams"]["home"]["name"]} v {d["teams"]["away"]["name"]}, {d["fixture"].get("date", "")[:10]}'
        out["has_expected_goals"] = "expected_goals" in stats and stats["expected_goals"] is not None
        out["expected_goals_value"] = stats.get("expected_goals")
        out["team_stat_names"] = list(stats)
    out["requests_used"] = client.used
    return out


def fetch(client: Client, leagues=LEAGUES, seasons=SEASONS) -> dict:
    store = client.store
    used, limit = quota(client)
    client.budget = min(client.budget, max(0, limit - RESERVE - used) + client.used)
    paid = limit > FREE_LIMIT
    if paid:   # 20 matches a request, faster pace, and another try at seasons the free plan refused
        client.batch, client.pause = True, min(client.pause, PAID_PAUSE_S)
    fetched, blocked = store.fetched_ids(), store.blocked()
    retry = set(blocked) if paid else set()
    summary = {"start_used_today": used, "limit_day": limit, "plan": "paid" if paid else "free",
               "budget": client.budget, "new_fixtures": 0, "lists": 0, "unblocked": []}
    try:
        for league, _name in leagues:
            for season in seasons:
                if (league, season) in blocked and (league, season) not in retry:
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
                    if (league, season) in blocked:
                        store.unblock(league, season)
                        blocked.discard((league, season))
                        summary["unblocked"].append(f"{league}-{season}")
                    listing = body
                else:
                    listing = store.get(path)
                todo = [f["fixture"]["id"] for f in listing["response"]
                        if f["fixture"]["status"]["short"] in FINISHED and f["fixture"]["id"] not in fetched]
                group = BATCH if client.batch else 1
                for i in range(0, len(todo), group):
                    ids = todo[i:i + group]
                    body = (client.get("fixtures", ids="-".join(map(str, ids))) if client.batch
                            else client.get("fixtures", id=ids[0]))
                    got = [f["fixture"]["id"] for f in body.get("response", [])]
                    obj = f"details/{league}-{season}/{ids[0]}.json.gz"
                    store.put(obj, body)
                    store.add_fixtures(league, season, obj, got)
                    fetched.update(got)
                    summary["new_fixtures"] += len(got)
    except (DailyLimit, PlanLimit) as e:
        summary["stopped"] = f"{type(e).__name__}: {e}"[:250]
    summary["requests_used"] = client.used
    summary["fixtures_total"] = len(fetched)
    return summary


def coverage(client: Client, leagues=LEAGUES) -> dict:
    """1 + len(leagues) calls: the plan, and per league and season what API-Football says it has
    (lineups, team statistics, player statistics). Team xG is not a coverage flag; xgcheck samples it."""
    s = client.get("status")["response"]
    out = {"plan": (s.get("subscription") or {}).get("plan") if isinstance(s, dict) else None,
           "requests_today": (s.get("requests") or {}) if isinstance(s, dict) else {}, "leagues": {}}
    for league, name in leagues:
        lg = client.get("leagues", id=league)["response"]
        out["leagues"][f"{league} {name}"] = {
            x["year"]: "".join(c for c, k in (("L", "lineups"), ("T", "statistics_fixtures"), ("P", "statistics_players"))
                               if x["coverage"]["fixtures"].get(k)) or "-"
            for x in (lg[0]["seasons"] if lg else [])}
    out["key"] = "L = lineups, T = team statistics, P = player statistics"
    out["requests_used"] = client.used
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["probe", "fetch", "xgcheck", "coverage"])
    ap.add_argument("--league", type=int, default=39, help="xgcheck: league id")
    ap.add_argument("--season", type=int, default=2022, help="xgcheck: season (starting year)")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--batch", action="store_true", help="20 matches a request (paid plans only)")
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
    client = Client(key, store, a.budget, batch=a.batch)
    if a.mode == "xgcheck":
        result = xgcheck(client, a.league, a.season)
    else:
        result = {"probe": probe, "fetch": fetch, "coverage": coverage}[a.mode](client)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
