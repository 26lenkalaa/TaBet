"""API-Football collector: budget, resume, plan limits and the odds refusal, against a fake API."""
import io
import json
import urllib.parse

import pytest

from tabet import api_football as af

KEY = "secret-key-123"


class FakeAPI:
    """Serves /status, /fixtures?league&season (lists) and /fixtures?ids (details)."""

    def __init__(self, n_fixtures=45, allowed=(2023,), used=0, limit=100):
        self.calls = []
        self.allowed, self.used, self.limit = set(allowed), used, limit
        self.fixtures = [{"fixture": {"id": 1000 + i, "status": {"short": "FT" if i < n_fixtures - 2 else "NS"}},
                          "teams": {"home": {"name": "A"}, "away": {"name": "B"}}} for i in range(n_fixtures)]

    def __call__(self, req):
        assert req.headers.get("X-apisports-key") == KEY
        url = urllib.parse.urlparse(req.full_url)
        q = dict(urllib.parse.parse_qsl(url.query))
        endpoint = url.path.strip("/")
        self.calls.append((endpoint, q))
        if endpoint == "status":
            body = {"errors": [], "response": {"subscription": {"plan": "Free"},
                                               "requests": {"current": self.used, "limit_day": self.limit}}}
        elif endpoint == "fixtures" and "ids" in q:
            ids = [int(x) for x in q["ids"].split("-")]
            assert len(ids) <= af.BATCH
            body = {"errors": [], "results": len(ids), "response": [
                {"fixture": {"id": i}, "lineups": [{}], "statistics": [], "players": []} for i in ids]}
        elif endpoint == "fixtures":
            if int(q["season"]) not in self.allowed:
                body = {"errors": {"plan": "Free plans do not have access to this season, try from 2021 to 2023."},
                        "response": []}
            else:
                body = {"errors": [], "results": len(self.fixtures), "response": self.fixtures}
        else:
            raise AssertionError(endpoint)
        self.used += 1

        class R(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R(json.dumps(body).encode())


def client(store, api, budget=50):
    return af.Client(KEY, store, budget, opener=api, pause=0)


LEAGUES = [(39, "Premier League")]


def test_refuses_odds_and_predictions():
    c = client(af.MemoryStore(), FakeAPI())
    for ep in ("odds", "odds/live", "predictions"):
        with pytest.raises(ValueError):
            c.get(ep, fixture=1)


def test_fetch_batches_skips_unfinished_and_blocked_seasons():
    api, store = FakeAPI(n_fixtures=45, allowed=(2023,)), af.MemoryStore()
    s = af.fetch(client(store, api), LEAGUES, seasons=[2024, 2023])
    assert s["new_fixtures"] == 43                      # 2 unfinished fixtures are left for later
    assert (39, 2024) in store.blocked()
    detail_calls = [q for e, q in api.calls if "ids" in q]
    assert len(detail_calls) == 3                       # 43 fixtures in batches of 20
    assert len(store.fetched_ids()) == 43
    assert {p.split("/")[0] for p in store.objects} == {"fixtures", "details"}


def test_resume_never_refetches():
    store = af.MemoryStore()
    af.fetch(client(store, FakeAPI(n_fixtures=45)), LEAGUES, seasons=[2023])
    api2 = FakeAPI(n_fixtures=45)
    s = af.fetch(client(store, api2), LEAGUES, seasons=[2023])
    assert s["new_fixtures"] == 0
    assert [e for e, q in api2.calls] == ["status"]     # list cached, nothing left to fetch, blocked season remembered


def test_run_budget_and_daily_reserve():
    api = FakeAPI(n_fixtures=400)
    s = af.fetch(client(af.MemoryStore(), api, budget=4), LEAGUES, seasons=[2023])
    assert len(api.calls) == 4 and s["requests_used"] == 4   # status + list + 2 batches
    api2 = FakeAPI(n_fixtures=400, used=88, limit=100)        # 12 left today, 10 kept in reserve
    s2 = af.fetch(client(af.MemoryStore(), api2, budget=50), LEAGUES, seasons=[2023])
    assert s2["requests_used"] <= 3


def test_log_never_contains_the_key():
    store = af.MemoryStore()
    af.fetch(client(store, FakeAPI()), LEAGUES, seasons=[2023])
    log = json.dumps(store.requests)
    assert KEY not in log and "ids" in log
