// Runs the real migration in an in-memory Postgres (PGlite) and syncs the real site data into it.
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PGlite } from "@electric-sql/pglite";
import { buildPayload } from "../src/soccer-payload.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.resolve(here, "..", "..", "site", "data");
const readJSON = async (f) => JSON.parse(await readFile(path.join(DATA, f), "utf8"));

const MIGRATIONS = ["0001_init.sql", "0002_row_order.sql", "0005_awaiting_status.sql", "0008_logged_supersede.sql"];
let db;
let payload;

const sync = (p) => db.query("select public.sync_competition_season($1::jsonb) as r", [JSON.stringify(p)]);
const one = async (sql, params) => (await db.query(sql, params)).rows[0];

before(async () => {
  db = new PGlite();
  // Supabase creates these roles and, like here, gives them full table and function privileges by
  // default; row-level security and the explicit revokes in the migration are what keep them read-only.
  await db.exec(`
    create role anon nologin; create role authenticated nologin; create role service_role nologin;
    grant usage on schema public to anon, authenticated, service_role;
    alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
    alter default privileges in schema public grant all on functions to anon, authenticated, service_role;`);
  for (const f of MIGRATIONS) await db.exec(await readFile(path.join(here, "..", "supabase", "migrations", f), "utf8"));
  const index = await readJSON("index.json");
  const season = index.seasons.at(-1);
  payload = buildPayload({
    matchesFile: await readJSON(index.files[season]),
    lineupsFile: await readJSON(index.lineup_files[season]),
    builtAt: index.built_utc,
  });
  await sync(payload);
});

test("the migrations can run twice", async () => {
  for (const f of MIGRATIONS) await db.exec(await readFile(path.join(here, "..", "supabase", "migrations", f), "utf8"));
});

test("forecasts and odds keep the site's order", async () => {
  for (const m of payload.matches) {
    const want = payload.forecasts.filter((f) => f.match_id === m.id).map((f) => f.key);
    const wantOdds = payload.market.filter((o) => o.match_id === m.id).map((o) => o.key);
    const row = await one("select forecasts, market from match_cards where id = $1", [m.id]);
    assert.deepEqual(row.forecasts.map((f) => f.key), want, m.id);
    assert.deepEqual(row.market.map((o) => o.key), wantOdds, m.id);
  }
});

test("a season syncs completely", async () => {
  const n = await one("select count(*)::int n from matches where season = $1", [payload.season]);
  assert.equal(n.n, payload.matches.length);
  const f = await one("select count(*)::int n from forecasts");
  assert.equal(f.n, payload.forecasts.length);
  const o = await one("select count(*)::int n from market_odds");
  assert.equal(o.n, payload.market.length);
  const s = await one("select matches, played from season_summaries where season = $1", [payload.season]);
  assert.equal(s.matches, payload.matches.length);
});

test("match_cards returns the site's shape", async () => {
  const src = payload.matches.find((m) => m.status === "played");
  const row = await one("select * from match_cards where id = $1", [src.id]);
  assert.equal(row.result, src.result);
  assert.equal(row.home_goals, src.home_goals);
  assert.ok(Array.isArray(row.forecasts) && Array.isArray(row.market));
  const p = row.market[0].p.map(Number);
  assert.ok(Math.abs(p[0] + p[1] + p[2] - 1) < 0.001);
});

test("syncing the same data again changes nothing", async () => {
  const before = await one("select max(updated_at) t from matches");
  await sync(payload);
  const after = await one("select max(updated_at) t from matches");
  assert.deepEqual(after.t, before.t);
});

test("logged forecasts cannot be updated or deleted", async () => {
  const logged = payload.forecasts.find((f) => f.kind === "logged");
  assert.ok(logged, "the current season should have a logged forecast");
  await assert.rejects(db.query("update forecasts set p_home = 0.5 where match_id = $1 and key = $2", [logged.match_id, logged.key]), /append-only/);
  await assert.rejects(db.query("delete from forecasts where match_id = $1 and key = $2", [logged.match_id, logged.key]), /append-only/);
});

test("a sync that would change a logged forecast is refused as a whole", async () => {
  const tampered = structuredClone(payload);
  const f = tampered.forecasts.find((x) => x.kind === "logged");
  f.p_home = Math.round((f.p_home + 0.01) * 1e4) / 1e4;
  f.p_draw = Math.round((f.p_draw - 0.01) * 1e4) / 1e4;
  tampered.matches[0].gameweek = 59;
  await assert.rejects(sync(tampered), /logged forecast/);
  const m = await one("select gameweek from matches where id = $1", [tampered.matches[0].id]);
  assert.notEqual(m.gameweek, 59, "the failed sync must roll back");
});

test("previews that disappear from the site are removed, logged ones stay", async () => {
  const shrunk = structuredClone(payload);
  shrunk.forecasts = shrunk.forecasts.filter((f) => f.kind === "logged");
  await sync(shrunk);
  const n = await one("select count(*)::int n from forecasts where kind <> 'logged'");
  assert.equal(n.n, 0);
  const l = await one("select count(*)::int n from forecasts where kind = 'logged'");
  assert.equal(l.n, payload.forecasts.filter((f) => f.kind === "logged").length);
  await sync(payload);
});

test("the public role can read but not write", async () => {
  await db.exec("set role anon");
  try {
    const r = await one("select count(*)::int n from match_cards");
    assert.equal(r.n, payload.matches.length);
    await assert.rejects(db.query("insert into sports (id, name) values ('x', 'x')"), /row-level security/);
    const upd = await db.query("update matches set gameweek = 59");
    assert.equal(upd.affectedRows ?? 0, 0, "RLS hides every row from updates");
    await assert.rejects(db.query("select public.sync_competition_season('{}'::jsonb)"), /permission denied/);
  } finally {
    await db.exec("reset role");
  }
});

// Migration 0008: a logged fallback can be superseded by the primary record, nothing else changes.
test("the primary record supersedes a logged fallback; a primary never changes", async () => {
  const base = structuredClone(payload);
  const f = base.forecasts.find((x) => x.kind === "logged");
  const id = { match: f.match_id, key: f.key };
  // start from a stored fallback with slightly different probabilities (as on 2026-11-22)
  await db.exec("alter table forecasts disable trigger forecasts_logged_append_only");
  await db.query("update forecasts set record_role = 'fallback', p_draw = p_draw + 0.0001, p_away = p_away - 0.0001 where match_id = $1 and key = $2",
                 [id.match, id.key]);
  await db.exec("alter table forecasts enable trigger forecasts_logged_append_only");
  // the primary arrives: accepted, and the row now holds the primary
  f.record_role = "primary";
  await sync(base);
  const row = await one("select record_role, p_draw::float8 d from forecasts where match_id = $1 and key = $2", [id.match, id.key]);
  assert.equal(row.record_role, "primary");
  assert.equal(row.d, Number(f.p_draw));
  // a changed primary is refused, as is a fallback replacing it
  const changed = structuredClone(base);
  const g = changed.forecasts.find((x) => x.match_id === id.match && x.key === id.key);
  g.p_draw = Math.round((g.p_draw + 0.01) * 1e4) / 1e4;
  g.p_away = Math.round((g.p_away - 0.01) * 1e4) / 1e4;
  await assert.rejects(sync(changed), /logged forecast/);
  g.record_role = "fallback";
  await assert.rejects(sync(changed), /logged forecast/);
  await assert.rejects(db.query("update forecasts set record_role = 'fallback' where match_id = $1 and key = $2", [id.match, id.key]), /append-only/);
  await assert.rejects(db.query("delete from forecasts where match_id = $1 and key = $2", [id.match, id.key]), /append-only/);
});

test("a logged row stored without a role gets its role filled in, probabilities unchanged", async () => {
  const f = payload.forecasts.filter((x) => x.kind === "logged").at(-1);
  await db.exec("alter table forecasts disable trigger forecasts_logged_append_only");
  await db.query("update forecasts set record_role = null where match_id = $1 and key = $2", [f.match_id, f.key]);
  await db.exec("alter table forecasts enable trigger forecasts_logged_append_only");
  const withRole = structuredClone(payload);
  withRole.forecasts.find((x) => x.match_id === f.match_id && x.key === f.key).record_role = "fallback";
  await sync(withRole);
  const row = await one("select record_role from forecasts where match_id = $1 and key = $2", [f.match_id, f.key]);
  assert.equal(row.record_role, "fallback");
  // a fallback's probabilities change only when the primary replaces it
  await assert.rejects(db.query("update forecasts set p_home = 0.5 where match_id = $1 and key = $2", [f.match_id, f.key]), /append-only/);
});
