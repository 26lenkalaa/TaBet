import { test } from "node:test";
import assert from "node:assert/strict";
import { buildPayload, matchId, slug } from "../src/soccer-payload.js";

const file = {
  season: "2026-27",
  matches: [
    {
      id: "2026-27-20260821-arsenal-coventry", season: "2026-27", kickoff: "2026-08-21T19:00:00+00:00",
      home: "Arsenal", away: "Coventry City", score: "3-0", result: "H", status: "played", gameweek: 1,
      forecasts: [{ key: "pE", label: "Logged", kind: "logged", p: [0.6, 0.25, 0.15] }],
      market: [{ key: "close", label: "Closing", p: [0.62, 0.22, 0.16] }],
    },
    {
      id: "fpl-111", season: "2026-27", kickoff: "2026-11-29T16:30:00+00:00",
      home: "Brighton & Hove Albion", away: "Nott'm Forest", status: "upcoming", gameweek: 12,
      forecasts: [], market: [],
    },
  ],
};

test("slugs are stable and URL-safe", () => {
  assert.equal(slug("Brighton & Hove Albion"), "brighton-and-hove-albion");
  assert.equal(slug("Nott'm Forest"), "nott-m-forest");
  assert.equal(slug("Atlético Madrid"), "atletico-madrid");
  assert.equal(matchId("premier-league", "2026-27", "Arsenal", "Coventry City"), "premier-league:2026-27:arsenal-v-coventry-city");
});

test("a season file becomes rows", () => {
  const p = buildPayload({ matchesFile: file, lineupsFile: { lineups: { "fpl-111": { x: 1 } } } });
  assert.equal(p.matches.length, 2);
  assert.deepEqual([p.matches[0].home_goals, p.matches[0].away_goals], [3, 0]);
  assert.equal(p.matches[1].home_goals, null);
  assert.equal(p.forecasts[0].p_home, 0.6);
  assert.equal(p.teams.length, 4);
  assert.equal(p.lineups[0].match_id, "premier-league:2026-27:brighton-and-hove-albion-v-nott-m-forest");
});

test("bad data is rejected instead of synced", () => {
  const dup = structuredClone(file);
  dup.matches.push({ ...dup.matches[0], id: "other" });
  assert.throws(() => buildPayload({ matchesFile: dup }), /duplicate/);
  const kind = structuredClone(file);
  kind.matches[0].forecasts[0].kind = "odds";
  assert.throws(() => buildPayload({ matchesFile: kind }), /unknown forecast kind/);
  const p = structuredClone(file);
  p.matches[0].market[0].p = [0.5, 0.5];
  assert.throws(() => buildPayload({ matchesFile: p }), /bad probabilities/);
});
