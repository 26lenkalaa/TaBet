#!/usr/bin/env node
// Copies the Premier League site data (committed to git by the logger) into Supabase.
// Git stays the source of truth; this only refreshes the read copy the API serves.
//
//   SUPABASE_URL=... SUPABASE_SECRET_KEY=... node scripts/sync-soccer.mjs [--season 2026-27 | --latest] [--dry-run]
//
// Without the two variables it prints a notice and exits 0, so CI and the logger keep working
// before Supabase is set up.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildPayload } from "../src/soccer-payload.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.resolve(here, "..", "..", "soccer (premier league)", "site", "data");

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const only = args.includes("--season") ? args[args.indexOf("--season") + 1] : null;

const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

async function readJSON(file) {
  return JSON.parse(await readFile(path.join(DATA, file), "utf8"));
}

async function rpc(payload) {
  const headers = { apikey: key, "Content-Type": "application/json" };
  if (key.startsWith("eyJ")) headers.Authorization = `Bearer ${key}`; // legacy service_role JWT
  const res = await fetch(`${url.replace(/\/$/, "")}/rest/v1/rpc/sync_competition_season`, {
    method: "POST",
    headers,
    body: JSON.stringify({ payload }),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`Supabase ${res.status}: ${text.slice(0, 500)}`);
  return JSON.parse(text);
}

async function main() {
  if (!dryRun && (!url || !key)) {
    console.log("Supabase is not configured (SUPABASE_URL / SUPABASE_SECRET_KEY); nothing synced.");
    return;
  }
  const index = await readJSON("index.json");
  const seasons = only ? [only] : args.includes("--latest") ? index.seasons.slice(-1) : index.seasons;
  for (const season of seasons) {
    const file = index.files[season];
    if (!file) throw new Error(`no data file for season ${season}`);
    const matchesFile = await readJSON(file);
    const lf = index.lineup_files?.[season];
    const lineupsFile = lf ? await readJSON(lf).catch(() => null) : null;
    const payload = buildPayload({ matchesFile, lineupsFile, builtAt: index.built_utc ?? null });
    const counts = `${payload.matches.length} matches, ${payload.forecasts.length} forecasts, ` +
      `${payload.market.length} odds, ${payload.lineups.length} lineups`;
    if (dryRun) {
      console.log(`[dry run] ${season}: ${counts}`);
      continue;
    }
    await rpc(payload);
    console.log(`synced ${season}: ${counts}`);
  }
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
