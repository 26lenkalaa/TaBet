// Turns one season of the static site's data (site/data/matches-<season>.json and
// lineups-<season>.json) into the payload that sync_competition_season() stores.
// Pure functions only, so they can be tested without a database.

export const SPORT = "soccer";
export const COMPETITION = "premier-league";

// Every forecast kind the site can show; logged ones are append-only in the database.
const KINDS = new Set(["logged", "saved", "preview", "backtest"]);

export function slug(name) {
  return String(name)
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export const teamId = (name) => `${SPORT}:${slug(name)}`;

// Stable for the whole season: in a league each home/away pairing happens once, whereas the site's
// own id changes when a fixture goes from upcoming ("fpl-111") to played (dated slug).
export const matchId = (competition, season, home, away) =>
  `${competition}:${season}:${slug(home)}-v-${slug(away)}`;

function probs(p, where) {
  if (!Array.isArray(p) || p.length !== 3 || p.some((x) => typeof x !== "number" || x < 0 || x > 1)) {
    throw new Error(`bad probabilities at ${where}: ${JSON.stringify(p)}`);
  }
  return { p_home: p[0], p_draw: p[1], p_away: p[2] };
}

function goals(score, where) {
  if (score == null) return [null, null];
  const m = /^(\d+)-(\d+)$/.exec(score);
  if (!m) throw new Error(`bad score at ${where}: ${score}`);
  return [Number(m[1]), Number(m[2])];
}

export function buildPayload({ matchesFile, lineupsFile = null, builtAt = null, competition = COMPETITION }) {
  const season = matchesFile.season;
  if (!/^\d{4}-\d{2}$/.test(season ?? "")) throw new Error(`bad season: ${season}`);

  const teams = new Map();
  const matches = [];
  const forecasts = [];
  const market = [];
  const bySource = new Map();
  const ids = new Set();

  for (const m of matchesFile.matches) {
    const where = `${season} ${m.id}`;
    const id = matchId(competition, season, m.home, m.away);
    if (bySource.has(m.id) || ids.has(id)) throw new Error(`duplicate match ${where}`);
    bySource.set(m.id, id);
    ids.add(id);
    teams.set(teamId(m.home), m.home);
    teams.set(teamId(m.away), m.away);

    const [hg, ag] = goals(m.score, where);
    matches.push({
      id,
      source_id: m.id,
      gameweek: m.gameweek ?? null,
      kickoff: m.kickoff,
      home_team_id: teamId(m.home),
      away_team_id: teamId(m.away),
      status: m.status,
      home_goals: hg,
      away_goals: ag,
      result: m.result ?? null,
      note: m.missing ?? null,
    });

    // `position` keeps the site's row order (e.g. logged before preview, pre-match before closing odds)
    (m.forecasts ?? []).forEach((f, position) => {
      if (!KINDS.has(f.kind)) throw new Error(`unknown forecast kind ${f.kind} at ${where}`);
      forecasts.push({ match_id: id, key: f.key, kind: f.kind, label: f.label, note: f.note ?? null, ...probs(f.p, where), position });
    });
    (m.market ?? []).forEach((o, position) => {
      market.push({ match_id: id, key: o.key, label: o.label, note: o.note ?? null, ...probs(o.p, where), position });
    });
  }

  const lineups = [];
  for (const [source, data] of Object.entries(lineupsFile?.lineups ?? {})) {
    const id = bySource.get(source);
    if (id) lineups.push({ match_id: id, data });
  }

  return {
    competition,
    season,
    built_at: builtAt,
    teams: [...teams].map(([id, name]) => ({ id, name })),
    matches,
    forecasts,
    market,
    lineups,
  };
}
