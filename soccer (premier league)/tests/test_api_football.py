"""API-Football collector: budget, resume, plan limits and the odds refusal, against a fake API."""
import io
import json
import urllib.parse

import pytest

from tabet import api_football as af

KEY = "secret-key-123"


class FakeAPI:
    """Serves /status, /fixtures?league&season (lists) and /fixtures?ids (details)."""

    def __init__(self, n_fixtures=45, allowed=(2023,), used=0, limit=100, free=True):
        self.calls, self.free = [], free
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
        elif endpoint == "fixtures" and "ids" in q and self.free:
            body = {"errors": {"plan": "Free plans do not have access to the Ids parameter."}, "response": []}
        elif endpoint == "fixtures" and "id" in q:
            i = int(q["id"])
            body = {"errors": [], "results": 1, "response": [
                {"fixture": {"id": i}, "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
                 "lineups": [{"startXI": []}], "statistics": [{"statistics": [{"type": "Shots on Goal"}]}],
                 "players": [{"players": [{"statistics": [{"games": {}, "shots": {}}]}]}]}]}
        elif endpoint == "leagues":
            body = {"errors": [], "results": 1, "response": [{"seasons": [
                {"year": 2023, "coverage": {"fixtures": {"lineups": True, "statistics_fixtures": True, "statistics_players": True}}}]}]}
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


def client(store, api, budget=50, batch=False):
    return af.Client(KEY, store, budget, opener=api, pause=0, batch=batch)


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
    detail_calls = [q for e, q in api.calls if "id" in q]
    assert len(detail_calls) == 43                      # free plan: one match a request
    assert len(store.fetched_ids()) == 43
    assert {p.split("/")[0] for p in store.objects} == {"fixtures", "details"}


def test_resume_never_refetches():
    store = af.MemoryStore()
    af.fetch(client(store, FakeAPI(n_fixtures=45)), LEAGUES, seasons=[2023])
    api2 = FakeAPI(n_fixtures=45)
    s = af.fetch(client(store, api2), LEAGUES, seasons=[2023])
    assert s["new_fixtures"] == 0
    assert [e for e, q in api2.calls] == ["status"]     # list cached, nothing left to fetch, blocked season remembered


def test_batches_of_20_on_a_paid_plan():
    api, store = FakeAPI(n_fixtures=45, free=False), af.MemoryStore()
    af.fetch(client(store, api, batch=True), LEAGUES, seasons=[2023])
    assert len([q for e, q in api.calls if "ids" in q]) == 3 and len(store.fetched_ids()) == 43


def test_probe_uses_five_calls_and_keeps_the_list():
    api, store = FakeAPI(allowed=(2023,)), af.MemoryStore()
    out = af.probe(client(store, api))
    assert out["requests_used_by_probe"] == 5
    assert out["seasons"][2025].startswith("blocked") and out["seasons"][2023].startswith("allowed")
    assert out["sample_match"]["has_lineups"] and out["sample_match"]["team_stats"] == ["Shots on Goal"]
    assert store.list_path(39, 2023) and (39, 2025) in store.blocked()   # fetch will not ask again


def test_run_budget_and_daily_reserve():
    api = FakeAPI(n_fixtures=400)
    s = af.fetch(client(af.MemoryStore(), api, budget=4), LEAGUES, seasons=[2023])
    assert len(api.calls) == 4 and s["requests_used"] == 4   # status + list + 2 matches
    api2 = FakeAPI(n_fixtures=400, used=88, limit=100)        # 12 left today, 10 kept in reserve
    s2 = af.fetch(client(af.MemoryStore(), api2, budget=50), LEAGUES, seasons=[2023])
    assert s2["requests_used"] <= 3


def test_log_never_contains_the_key():
    store = af.MemoryStore()
    af.fetch(client(store, FakeAPI()), LEAGUES, seasons=[2023])
    log = json.dumps(store.requests)
    assert KEY not in log and '"id"' in log


def test_xgcheck_uses_two_calls_and_reports_expected_goals():
    api, store = FakeAPI(allowed=(2022,)), af.MemoryStore()
    out = af.xgcheck(client(store, api), league=39, season=2022)
    assert out["requests_used"] == 2 and store.list_path(39, 2022)
    assert out["has_expected_goals"] is False and out["team_stat_names"] == ["Shots on Goal"]


def test_a_paid_plan_is_detected_batches_and_retries_blocked_seasons():
    store = af.MemoryStore()
    af.fetch(client(store, FakeAPI(n_fixtures=45, allowed=(2023,))), LEAGUES, seasons=[2024, 2023])
    assert (39, 2024) in store.blocked()                          # refused on the free plan
    api = FakeAPI(n_fixtures=45, allowed=(2023, 2024), free=False, limit=7500)
    s = af.fetch(client(store, api), LEAGUES, seasons=[2024, 2023])   # no --batch flag needed
    assert s["plan"] == "paid" and s["unblocked"] == ["39-2024"] and (39, 2024) not in store.blocked()
    assert store.list_path(39, 2024) and s["new_fixtures"] == 0     # the fake reuses 2023's ids, already stored


def test_a_paid_plan_batches_without_the_flag():
    api = FakeAPI(n_fixtures=45, free=False, limit=7500)
    s = af.fetch(client(af.MemoryStore(), api), LEAGUES, seasons=[2023])
    assert not [q for e, q in api.calls if "id" in q]               # never one match a request
    assert len([q for e, q in api.calls if "ids" in q]) == 3 and s["new_fixtures"] == 43


def test_a_season_still_refused_on_a_paid_plan_stays_blocked():
    store = af.MemoryStore()
    store.block(39, 2010, "refused")
    api = FakeAPI(n_fixtures=45, allowed=(2023,), free=False, limit=7500)
    s = af.fetch(client(store, api), LEAGUES, seasons=[2010])
    assert (39, 2010) in store.blocked() and s["unblocked"] == []


def test_free_plan_keeps_one_match_a_request_and_skips_blocked():
    store = af.MemoryStore()
    store.block(39, 2024, "refused")
    api = FakeAPI(n_fixtures=45, allowed=(2023, 2024))
    af.fetch(client(store, api), LEAGUES, seasons=[2024])
    assert [e for e, q in api.calls] == ["status"]


def test_coverage_reports_flags_per_season():
    api = FakeAPI(free=False, limit=7500)
    out = af.coverage(client(af.MemoryStore(), api), LEAGUES)
    assert out["leagues"]["39 Premier League"] == {2023: "LTP"} and out["requests_used"] == 2
