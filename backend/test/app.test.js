import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createApp } from "../src/app.js";

const row = {
  id: "premier-league:2026-27:arsenal-v-coventry-city", source_id: "2026-27-20260821-arsenal-coventry",
  competition_id: "premier-league", season: "2026-27", gameweek: 1, kickoff: "2026-08-21T19:00:00+00:00",
  home: "Arsenal", away: "Coventry City", status: "played", home_goals: 3, away_goals: 0, result: "H", note: null,
  forecasts: [{ key: "pE", kind: "logged", label: "Logged", note: null, p: ["0.6000", "0.2500", "0.1500"] }],
  market: [],
};

const calls = [];
const repo = {
  sports: async () => [{ id: "soccer", name: "Soccer" }],
  competitions: async () => [{ id: "premier-league", sport_id: "soccer", name: "Premier League", country: "England" }],
  competition: async (id) => (id === "premier-league" ? { id } : null),
  seasons: async () => [{ season: "2026-27", matches: 380, played: 110 }],
  matches: async (c, s, f) => { calls.push(f); return [row]; },
  match: async (id) => (id === row.id ? row : null),
  lineup: async () => ({ actual: {} }),
  lineups: async () => [{ id: row.id, source_id: row.source_id, lineup: { actual: {} } }],
  lastSync: async () => ({ season: "2026-27", source_built_at: "2026-11-14T02:22:00+00:00", ran_at: "2026-11-14T02:22:26+00:00" }),
};

let server;
let base;
before(async () => {
  server = createApp({ repo, log: { error() {} } }).listen(0);
  await new Promise((r) => server.once("listening", r));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => server.close());

const get = async (p, init) => {
  const res = await fetch(base + p, init);
  return { status: res.status, headers: res.headers, body: await res.json() };
};

test("index lists the endpoints", async () => {
  const r = await get("/api");
  assert.equal(r.status, 200);
  assert.ok(r.body.endpoints.length >= 5);
});

test("health is never cached", async () => {
  const r = await get("/api/health");
  assert.equal(r.body.status, "ok");
  assert.equal(r.headers.get("cache-control"), "no-store");
});

test("matches come back in the site's shape with numeric probabilities", async () => {
  const r = await get("/api/competitions/premier-league/seasons/2026-27/matches?gameweek=1&status=played");
  assert.equal(r.status, 200);
  assert.equal(r.body.count, 1);
  assert.equal(r.body.data[0].score, "3-0");
  assert.deepEqual(r.body.data[0].forecasts[0].p, [0.6, 0.25, 0.15]);
  assert.deepEqual(calls.at(-1), { gameweek: 1, status: "played" });
  assert.match(r.headers.get("cache-control"), /s-maxage=60/);
});

test("one match includes its lineup", async () => {
  const r = await get(`/api/matches/${encodeURIComponent(row.id)}`);
  assert.equal(r.status, 200);
  assert.deepEqual(r.body.data.lineup, { actual: {} });
});

test("bad input gets a clear 400", async () => {
  for (const p of [
    "/api/competitions/Premier%20League/seasons",
    "/api/competitions/premier-league/seasons/2026/matches",
    "/api/competitions/premier-league/seasons/2026-27/matches?gameweek=0",
    "/api/competitions/premier-league/seasons/2026-27/matches?status=live",
    "/api/matches/arsenal",
  ]) {
    const r = await get(p);
    assert.equal(r.status, 400, p);
    assert.equal(r.body.error.code, "bad_request");
  }
});

test("unknown things get a 404", async () => {
  assert.equal((await get("/api/competitions/la-liga/seasons")).status, 404);
  assert.equal((await get("/api/matches/premier-league:2026-27:a-v-b")).status, 404);
  assert.equal((await get("/api/nope")).status, 404);
});

test("the API is read-only", async () => {
  const r = await get("/api/sports", { method: "POST" });
  assert.equal(r.status, 405);
});

test("database failures do not leak details", async () => {
  const broken = createApp({ repo: { ...repo, sports: async () => { throw new Error("password=hunter2"); } }, log: { error() {} } }).listen(0);
  await new Promise((r) => broken.once("listening", r));
  const res = await fetch(`http://127.0.0.1:${broken.address().port}/api/sports`);
  const body = await res.json();
  broken.close();
  assert.equal(res.status, 500);
  assert.doesNotMatch(JSON.stringify(body), /hunter2/);
});

test("seasons carry when the data was built", async () => {
  const r = await get("/api/competitions/premier-league/seasons");
  assert.equal(r.status, 200);
  assert.equal(r.body.meta.built_at, "2026-11-14T02:22:00+00:00");
  assert.equal(r.body.data[0].season, "2026-27");
});

test("a season's lineups come keyed by both ids", async () => {
  const r = await get("/api/competitions/premier-league/seasons/2026-27/lineups");
  assert.equal(r.status, 200);
  assert.equal(r.body.data[0].source_id, row.source_id);
  assert.match(r.headers.get("cache-control"), /s-maxage=300/);
  assert.equal((await get("/api/competitions/premier-league/seasons/26-27/lineups")).status, 400);
});
