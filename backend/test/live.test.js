// /api/live against a fake FPL: team codes and player names, own goals credited to the side that
// benefits, match states, edge-cache length, name refresh, and upstream failures.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createApp } from "../src/app.js";
import { createLive, toLive, cacheSeconds } from "../src/live.js";

const BOOT = {
  teams: [{ id: 4, short_name: "BRE" }, { id: 9, short_name: "EVE" }, { id: 1, short_name: "ARS" }, { id: 6, short_name: "CHE" }],
  elements: [{ id: 86, web_name: "Mbeumo" }, { id: 98, web_name: "Wissa" }, { id: 300, web_name: "Tarkowski" }, { id: 301, web_name: "Gueye" }],
  events: [{ id: 11, is_current: false, is_next: false }, { id: 12, is_current: true, is_next: false }],
};
const fixture = (o) => ({ id: 1, team_h: 4, team_a: 9, kickoff_time: "2026-11-28T15:00:00Z", started: true, finished: false,
  finished_provisional: false, minutes: 67, team_h_score: 2, team_a_score: 0, stats: [], ...o });
const LIVE_FX = fixture({ stats: [
  { identifier: "goals_scored", h: [{ element: 86, value: 1 }], a: [] },
  { identifier: "own_goals", h: [], a: [{ element: 300, value: 1 }] },
  { identifier: "yellow_cards", h: [], a: [{ element: 301, value: 1 }] },
  { identifier: "bps", h: [{ element: 86, value: 40 }], a: [] },
] });

function fakeFPL(fixtures, calls = []) {
  return async (url) => {
    calls.push(url);
    const body = url.includes("bootstrap-static") ? BOOT : url.includes("fixtures/?event=12") ? fixtures : null;
    if (!body) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => body };
  };
}

test("an FPL fixture becomes the site's terms", () => {
  const teams = new Map(BOOT.teams.map((t) => [t.id, t.short_name]));
  const players = new Map(BOOT.elements.map((p) => [p.id, p.web_name]));
  const m = toLive(LIVE_FX, teams, players);
  assert.equal(m.home, "BRE"); assert.equal(m.away, "EVE");
  assert.equal(m.state, "live"); assert.equal(m.minute, 67); assert.deepEqual(m.score, [2, 0]);
  assert.deepEqual(m.events, [
    { type: "goal", side: "home", player: "Mbeumo", n: 1 },
    { type: "own_goal", side: "home", player: "Tarkowski", n: 1 },   // an Everton player's own goal counts for Brentford
    { type: "yellow", side: "away", player: "Gueye", n: 1 },
  ]);
  assert.equal(toLive(fixture({ started: false, team_h_score: null, team_a_score: null }), teams, players).state, "pre");
  assert.equal(toLive(fixture({ started: false, team_h_score: null }), teams, players).score, null);
  assert.equal(toLive(fixture({ finished_provisional: true }), teams, players).state, "ft");
  const done = toLive(fixture({ finished_provisional: true, finished: true }), teams, players);
  assert.equal(done.state, "final"); assert.equal(done.minute, null);
});

test("the edge cache is short around matches and long otherwise", () => {
  const now = Date.parse("2026-11-28T14:50:00Z");
  assert.equal(cacheSeconds([{ state: "pre", kickoff: "2026-11-28T15:00:00Z" }], now), 30);
  assert.equal(cacheSeconds([{ state: "live", kickoff: "2026-11-28T13:00:00Z" }], now), 30);
  assert.equal(cacheSeconds([{ state: "pre", kickoff: "2026-11-29T15:00:00Z" }, { state: "final", kickoff: "2026-11-27T20:00:00Z" }], now), 300);
});

test("/api/live serves the current gameweek and caches names", async () => {
  const calls = [];
  let t = Date.parse("2026-11-28T15:30:00Z");
  const app = createApp({ repo: {}, live: createLive({ fetchImpl: fakeFPL([LIVE_FX], calls), now: () => t }), log: { error() {} } });
  const server = app.listen(0);
  await new Promise((r) => server.once("listening", r));
  const base = `http://127.0.0.1:${server.address().port}/api`;
  try {
    const r = await fetch(`${base}/live/premier-league`);
    assert.equal(r.status, 200);
    assert.match(r.headers.get("cache-control"), /s-maxage=30/);
    const d = await r.json();
    assert.equal(d.gameweek, 12);
    assert.equal(d.matches[0].events[0].player, "Mbeumo");
    t += 60 * 1000;
    await fetch(`${base}/live/premier-league`);
    assert.equal(calls.filter((u) => u.includes("bootstrap")).length, 1, "names are fetched once, not on every request");
    assert.equal((await fetch(`${base}/live/la-liga`)).status, 404);
  } finally {
    server.close();
  }
});

test("an unknown player refreshes the names, at most every 10 minutes", async () => {
  const calls = [];
  let t = 0;
  const fx = [fixture({ stats: [{ identifier: "goals_scored", h: [{ element: 999, value: 1 }], a: [] }] })];
  const live = createLive({ fetchImpl: fakeFPL(fx, calls), now: () => t });
  assert.equal((await live()).matches[0].events[0].player, "Player 999");
  t = 5 * 60 * 1000; await live();
  assert.equal(calls.filter((u) => u.includes("bootstrap")).length, 1);
  t = 11 * 60 * 1000; await live();
  assert.equal(calls.filter((u) => u.includes("bootstrap")).length, 2);
});

test("FPL being down is a 502 with a readable message", async () => {
  const app = createApp({ repo: {}, live: createLive({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) }),
                          log: { error() {} } });
  const server = app.listen(0);
  await new Promise((r) => server.once("listening", r));
  try {
    const r = await fetch(`http://127.0.0.1:${server.address().port}/api/live/premier-league`);
    assert.equal(r.status, 502);
    assert.match((await r.json()).error.message, /unavailable right now: FPL/);
    assert.equal(r.headers.get("cache-control"), "no-store");
  } finally {
    server.close();
  }
});
