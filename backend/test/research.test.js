// Migration 0007 on an in-memory Postgres (PGlite): the research tables move out of public with their
// rows, link to research.leagues, get a season label, and stay closed to the public roles.
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PGlite } from "@electric-sql/pglite";

const here = path.dirname(fileURLToPath(import.meta.url));
const sql = (f) => readFile(path.join(here, "..", "supabase", "migrations", f), "utf8");
const MIGRATIONS = ["0001_init.sql", "0002_row_order.sql", "0003_api_football.sql", "0004_odds_archive.sql",
  "0005_awaiting_status.sql", "0006_manager_changes.sql"];
let db;
const one = async (q, p) => (await db.query(q, p)).rows[0];

before(async () => {
  db = new PGlite();
  // Supabase provides these roles (service_role bypasses row-level security) and the Storage table;
  // PGlite needs stand-ins.
  await db.exec(`
    create role anon nologin; create role authenticated nologin; create role service_role nologin bypassrls;
    grant usage on schema public to anon, authenticated, service_role;
    alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
    create schema storage; create table storage.buckets (id text primary key, name text, public boolean);`);
  for (const f of MIGRATIONS) await db.exec(await sql(f));
  // rows like the live ones, written before the move
  await db.exec(`
    insert into competitions values ('premier-league', 'soccer', 'Premier League', 'England')
      on conflict do nothing;
    insert into public.apif_lists (league, season, object_path, n_fixtures) values (39, 2024, 'lists/39/2024.json.gz', 380);
    insert into public.apif_fixtures (fixture_id, league, season, object_path) values (1, 140, 2015, 'f/1.json.gz');
    insert into public.manager_changes (league, season, team_name, source) values ('E0', '2023-24', 'Chelsea', '2023-24 Premier League');`);
  await db.exec(await sql("0007_research_schema.sql"));
});

test("0007 can run twice", async () => {
  await db.exec(await sql("0007_research_schema.sql"));
});

test("research tables leave public with their rows", async () => {
  const pub = await db.query(`select table_name from information_schema.tables where table_schema = 'public'
                              and (table_name like 'apif%' or table_name = 'manager_changes')`);
  assert.equal(pub.rows.length, 0);
  assert.equal((await one("select count(*)::int n from research.apif_lists")).n, 1);
  assert.equal((await one("select count(*)::int n from research.manager_changes")).n, 1);
});

test("seasons get the shared label and leagues the lookup", async () => {
  assert.equal((await one("select season_label from research.apif_lists")).season_label, "2024-25");
  assert.equal((await one("select season_label from research.apif_fixtures")).season_label, "2015-16");
  const r = await one(`select l.code from research.apif_lists a join research.leagues l on l.apif_id = a.league`);
  assert.equal(r.code, "E0");
  const e0 = await one("select competition_id from research.leagues where code = 'E0'");
  assert.equal(e0.competition_id, "premier-league");
});

test("unknown leagues and malformed seasons are refused", async () => {
  await assert.rejects(db.exec(`insert into research.apif_lists (league, season, object_path, n_fixtures)
                                values (99999, 2024, 'x', 1)`), /foreign key/);
  await assert.rejects(db.exec(`insert into research.manager_changes (league, season, team_name, source)
                                values ('XX', '2023-24', 'A', 's')`), /foreign key/);
  await assert.rejects(db.exec(`insert into research.manager_changes (league, season, team_name, source)
                                values ('E0', '2023–24', 'A', 's')`), /season_format/);
});

test("the public roles cannot reach the research schema; the secret key can", async () => {
  for (const role of ["anon", "authenticated"]) {
    await db.exec(`set role ${role}`);
    await assert.rejects(db.query("select * from research.leagues"), /permission denied/);
    await db.exec("reset role");
  }
  await db.exec("set role service_role");
  assert.ok((await one("select count(*)::int n from research.leagues")).n > 50);
  await db.exec("reset role");
});

test("every table and view has a description", async () => {
  const r = await db.query(`
    select n.nspname || '.' || c.relname as name from pg_class c join pg_namespace n on n.oid = c.relnamespace
    where n.nspname in ('public', 'research') and c.relkind in ('r', 'v') and obj_description(c.oid) is null`);
  assert.deepEqual(r.rows.map((x) => x.name), []);
});
