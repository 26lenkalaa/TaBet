// Runs the real migration in an in-memory Postgres (PGlite) and syncs the real site data into it.
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PGlite } from "@electric-sql/pglite";
import { buildPayload } from "../src/soccer-payload.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.resolve(here, "..", "..", "soccer (premier league)", "site", "data");
const readJSON = async (f) => JSON.parse(await readFile(path.join(DATA, f), "utf8"));

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
  await db.exec(await readFile(path.join(here, "..", "supabase", "migrations", "0001_init.sql"), "utf8"));
  const index = await readJSON("index.json");
  const season = index.seasons.at(-1);
  payload = buildPayload({
    matchesFile: await readJSON(index.files[season]),
    lineupsFile: await readJSON(index.lineup_files[season]),
    builtAt: index.built_utc,
  });
  await sync(payload);
});

test("the migration can run twice", async () => {
  await db.exec(await readFile(path.join(here, "..", "supabase", "migrations", "0001_init.sql"), "utf8"));
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
