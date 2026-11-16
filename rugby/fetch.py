"""Download international rugby data from API-Rugby (api-sports.io).

Only talks to the rugby API (v1.rugby.api-sports.io). The key comes from the RUGBY_API_KEY environment
variable, which the GitHub workflow fills from the repo secret of the same name. The key is never
printed or written to a file.

    python fetch.py probe     # about 15 requests: plan, competitions, which seasons the key can read, odds

The free plan allows 100 requests a day and 10 a minute, so calls are spaced 6.5 s apart and a run
stops early if fewer than 10 requests would be left for the day.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("RUGBY_API_BASE", "https://v1.rugby.api-sports.io")   # overridable for tests
PAUSE = float(os.environ.get("RUGBY_API_PAUSE", "6.5"))   # seconds between calls (10 a minute)
KEEP = 10   # requests always left unused for the day

# Men's international XV competitions, as API-Rugby names them. Matched case-insensitively; sevens,
# women's and age-grade competitions are left out on purpose.
INTERNATIONAL = [
    "Six Nations", "Rugby Championship", "World Cup", "Friendly International", "Lions Tour",
    "Nations Cup", "Autumn Nations Cup", "Pacific Nations Cup", "Rugby Europe Championship",
    "Americas Championship", "Pacific Challenge", "Nations Championship",
]
SKIP_WORDS = ("women", "7's", "sevens", "u20", "u18", "u19", "trophy", "conference")


class Budget(Exception):
    """Raised when a call would eat into the requests we keep for the day."""


class Client:
    def __init__(self, key: str):
        self.key = key
        self.calls = 0
        self.remaining = None   # from the x-ratelimit-requests-remaining header
        self._last = 0.0

    def get(self, endpoint: str, **params) -> dict:
        if self.remaining is not None and self.remaining <= KEEP:
            raise Budget(f"only {self.remaining} requests left today; stopping to keep {KEEP} spare")
        wait = PAUSE - (time.monotonic() - self._last)
        if self._last and wait > 0:
            time.sleep(wait)
        url = BASE + "/" + endpoint + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"x-apisports-key": self.key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.load(resp)
                left = resp.headers.get("x-ratelimit-requests-remaining")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{endpoint}: HTTP {error.code}") from None
        finally:
            self._last = time.monotonic()
            self.calls += 1
        if left is not None and left.strip().lstrip("-").isdigit():
            self.remaining = int(left)
        return body


def errors_of(body: dict) -> str:
    """API-Sports puts problems in `errors`, as a list or a dict; turn either into one line."""
    errs = body.get("errors") or []
    if isinstance(errs, dict):
        return "; ".join(f"{k}: {v}" for k, v in errs.items())
    return "; ".join(str(e) for e in errs)


def is_international(name: str) -> bool:
    low = name.lower()
    if any(w in low for w in SKIP_WORDS):
        return False
    return any(low == want.lower() or low.startswith(want.lower()) for want in INTERNATIONAL)


def probe(client: Client) -> dict:
    report: dict = {"plan": None, "competitions": [], "season_access": [], "odds": None, "problems": []}

    # 1. the account: plan name and today's usage (no key or email in the report)
    status = client.get("status")
    acct = status.get("response") or {}
    if isinstance(acct, dict):
        sub = acct.get("subscription") or {}
        reqs = acct.get("requests") or {}
        report["plan"] = {"plan": sub.get("plan"), "active": sub.get("active"),
                          "used_today": reqs.get("current"), "limit_day": reqs.get("limit_day")}
    if errors_of(status):
        report["problems"].append("status: " + errors_of(status))

    # 2. every competition in one call; keep the men's international XV ones
    leagues = client.get("leagues")
    if errors_of(leagues):
        report["problems"].append("leagues: " + errors_of(leagues))
    comps = []
    for item in leagues.get("response") or []:
        lg = item.get("league") or {}
        name = lg.get("name") or ""
        if not is_international(name):
            continue
        years = sorted(int(s["year"]) for s in item.get("seasons") or [] if str(s.get("year", "")).isdigit())
        comps.append({"id": lg.get("id"), "name": name, "country": (item.get("country") or {}).get("name"),
                      "seasons": f"{years[0]}-{years[-1]}" if years else "none", "n_seasons": len(years),
                      "_years": years})
    comps.sort(key=lambda c: c["name"])
    report["competitions"] = [{k: v for k, v in c.items() if k != "_years"} for c in comps]

    # 3. which seasons the key may read: try a spread of years on two busy competitions
    for want in ("Six Nations", "Friendly International"):
        comp = next((c for c in comps if c["name"].lower() == want.lower()), None)
        if not comp or not comp["_years"]:
            report["problems"].append(f"{want}: not found in the competition list")
            continue
        years = comp["_years"]
        tries = sorted({years[-1], years[-1] - 1, years[-1] - 2, years[-1] - 4, years[-1] - 8, years[0]} & set(years),
                       reverse=True)
        for year in tries:
            body = client.get("games", league=comp["id"], season=year)
            games = body.get("response") or []
            done = sum(1 for g in games if (g.get("status") or {}).get("short") in ("FT", "AET", "AP"))
            report["season_access"].append({"competition": want, "season": year, "games": len(games),
                                            "finished": done, "error": errors_of(body) or None})
            # 4. odds: once, on the first upcoming game we see
            if report["odds"] is None:
                upcoming = [g for g in games if (g.get("status") or {}).get("short") == "NS"]
                if upcoming:
                    g = upcoming[0]
                    odds = client.get("odds", game=g["id"])
                    books = []
                    for row in odds.get("response") or []:
                        books += [b.get("name") for b in row.get("bookmakers") or []]
                    report["odds"] = {"game": f'{g["teams"]["home"]["name"]} v {g["teams"]["away"]["name"]}',
                                      "date": g.get("date"), "bookmakers": len(books),
                                      "examples": books[:5], "error": errors_of(odds) or None}
    if report["odds"] is None:
        report["odds"] = "no upcoming game found in the seasons tried"
    report["requests_used_this_run"] = client.calls
    report["requests_left_today"] = client.remaining
    return report


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    key = os.environ.get("RUGBY_API_KEY", "").strip()
    if not key:
        print("RUGBY_API_KEY is not set. Add it under Settings > Secrets and variables > Actions.")
        return 1
    if mode != "probe":
        print(f"Unknown mode {mode!r}. Only 'probe' exists so far.")
        return 2
    client = Client(key)
    try:
        report = probe(client)
    except Budget as stop:
        report = {"stopped": str(stop), "requests_used_this_run": client.calls}
    except (RuntimeError, OSError, ValueError) as error:
        report = {"failed": str(error), "requests_used_this_run": client.calls}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if "failed" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
