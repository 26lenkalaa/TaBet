// Data layer over Supabase, using the publishable (read-only) key. Row-level security in
// supabase/migrations/0001_init.sql allows reads only, so this key can never change data.
import { createClient } from "@supabase/supabase-js";
import { HttpError } from "../app.js";

let client = null;

function db() {
  if (client) return client;
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_PUBLISHABLE_KEY || process.env.SUPABASE_ANON_KEY;
  if (!url || !key) {
    const err = new HttpError(503, "not_configured", "The database is not configured yet.");
    err.expose = true;
    throw err;
  }
  client = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
  return client;
}

function check({ data, error }) {
  if (error) {
    const err = new HttpError(502, "database_error", "The database did not answer correctly.");
    err.expose = true;
    err.cause = error;
    throw err;
  }
  return data;
}

export const supabaseRepo = {
  async sports() {
    return check(await db().from("sports").select("id, name").order("id"));
  },
  async competitions() {
    return check(await db().from("competitions").select("id, sport_id, name, country").order("id"));
  },
  async competition(id) {
    return check(await db().from("competitions").select("id, sport_id, name, country").eq("id", id).maybeSingle());
  },
  async seasons(competition) {
    return check(
      await db()
        .from("season_summaries")
        .select("season, matches, played, first_kickoff, last_kickoff")
        .eq("competition_id", competition)
        .order("season", { ascending: false }),
    );
  },
  async matches(competition, season, { gameweek, status } = {}) {
    let q = db().from("match_cards").select("*").eq("competition_id", competition).eq("season", season);
    if (gameweek !== undefined) q = q.eq("gameweek", gameweek);
    if (status !== undefined) q = q.eq("status", status);
    return check(await q.order("kickoff").order("id"));
  },
  async match(id) {
    return check(await db().from("match_cards").select("*").eq("id", id).maybeSingle());
  },
  async lineup(id) {
    const row = check(await db().from("lineups").select("data").eq("match_id", id).maybeSingle());
    return row?.data ?? null;
  },
  // [{id, source_id, lineup}] for every match in the season that has one
  async lineups(competition, season) {
    const rows = check(
      await db()
        .from("lineups")
        .select("match_id, data, matches!inner(source_id, competition_id, season)")
        .eq("matches.competition_id", competition)
        .eq("matches.season", season)
        .order("match_id"),
    );
    return rows.map((r) => ({ id: r.match_id, source_id: r.matches.source_id, lineup: r.data }));
  },
  async lastSync() {
    const rows = check(
      await db().from("sync_runs").select("competition_id, season, source_built_at, ran_at").order("ran_at", { ascending: false }).limit(1),
    );
    return rows[0] ?? null;
  },
};
