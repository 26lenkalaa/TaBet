// Live scores for the current Premier League gameweek, from the Fantasy Premier League API (free, no
// key). Browsers cannot call FPL directly (no CORS), so /api/live fetches it here and Vercel's edge
// cache serves the result for 30 seconds: FPL sees at most a couple of requests a minute however
// many people have the page open. Live data is display only: no model or logged forecast reads it.
//
// FPL gives the score, minutes played, and per-player goals, assists, own goals, cards and penalty
// saves/misses. It does not give the minute of each goal.

const FPL = "https://fantasy.premierleague.com/api";
const BOOTSTRAP_TTL_MS = 6 * 3600 * 1000;      // team codes and player names change rarely
const TIMEOUT_MS = 6000;

// FPL stat identifier -> our event type. Own goals are credited to the side that benefits.
const STATS = {
  goals_scored: "goal",
  own_goals: "own_goal",
  assists: "assist",
  yellow_cards: "yellow",
  red_cards: "red",
  penalties_saved: "pen_saved",
  penalties_missed: "pen_missed",
};

export class LiveError extends Error {}

function state(f) {
  if (f.finished) return "final";                         // FPL has confirmed the result
  if (f.finished_provisional) return "ft";                // full time, not yet confirmed
  if (f.started) return "live";
  return "pre";
}

// One FPL fixture in the site's terms: team codes (ARS, MUN, ...) match site/teams.js.
export function toLive(f, teams, players) {
  const events = [];
  for (const s of f.stats ?? []) {
    const type = STATS[s.identifier];
    if (!type) continue;
    for (const [side, list] of [["home", s.h ?? []], ["away", s.a ?? []]]) {
      for (const { element, value } of list) {
        const credited = type === "own_goal" ? (side === "home" ? "away" : "home") : side;
        events.push({ type, side: credited, player: players.get(element) ?? `Player ${element}`, n: value });
      }
    }
  }
  const st = state(f);
  return {
    fpl_id: f.id,
    home: teams.get(f.team_h) ?? null,
    away: teams.get(f.team_a) ?? null,
    kickoff: f.kickoff_time,
    state: st,
    minute: st === "live" ? f.minutes ?? null : null,
    score: f.team_h_score == null || f.team_a_score == null ? null : [f.team_h_score, f.team_a_score],
    events,
  };
}

// How long the edge may cache a reply: 30 s around live matches, 5 minutes when nothing is on.
export function cacheSeconds(matches, now = Date.now()) {
  const busy = matches.some((m) => {
    if (m.state === "live" || m.state === "ft") return true;
    const t = Date.parse(m.kickoff);
    return m.state === "pre" && t - now < 15 * 60 * 1000 && now - t < 3 * 3600 * 1000;
  });
  return busy ? 30 : 300;
}

export function createLive({ fetchImpl = fetch, now = () => Date.now() } = {}) {
  let boot = null;   // { at, teams: Map(id -> code), players: Map(id -> name), event }

  async function get(path) {
    let res;
    try {
      res = await fetchImpl(`${FPL}${path}`, {
        headers: { "User-Agent": "TaBet/1.0 (student research project)", Accept: "application/json" },
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch (e) {
      throw new LiveError(`FPL unreachable (${e.name})`);
    }
    if (!res.ok) throw new LiveError(`FPL ${path.split("?")[0]} returned ${res.status}`);
    return res.json();
  }

  async function bootstrap() {
    if (boot && now() - boot.at < BOOTSTRAP_TTL_MS) return boot;
    const d = await get("/bootstrap-static/");
    boot = {
      at: now(),
      teams: new Map(d.teams.map((t) => [t.id, t.short_name])),
      players: new Map(d.elements.map((p) => [p.id, p.web_name])),
      events: d.events.map((e) => ({ id: e.id, current: e.is_current, next: e.is_next })),
    };
    return boot;
  }

  // The current gameweek (it becomes current at its deadline, before its first kickoff).
  return async function premierLeague() {
    const b = await bootstrap();
    const ev = b.events.find((e) => e.current) ?? b.events.find((e) => e.next) ?? null;
    if (!ev) return { gameweek: null, updated_at: new Date(now()).toISOString(), matches: [] };
    const fixtures = await get(`/fixtures/?event=${ev.id}`);
    const unknown = fixtures.some((f) => !b.teams.has(f.team_h) || !b.teams.has(f.team_a)) ||
      fixtures.some((f) => (f.stats ?? []).some((s) => [...(s.h ?? []), ...(s.a ?? [])].some((x) => !b.players.has(x.element))));
    if (unknown && now() - b.at > 10 * 60 * 1000) {   // a new signing: refresh names, at most every 10 min
      boot = null;
      const fresh = await bootstrap();
      return build(ev.id, fixtures, fresh);
    }
    return build(ev.id, fixtures, b);
  };

  function build(gameweek, fixtures, b) {
    const matches = fixtures
      .map((f) => toLive(f, b.teams, b.players))
      .sort((x, y) => (x.kickoff < y.kickoff ? -1 : x.kickoff > y.kickoff ? 1 : x.fpl_id - y.fpl_id));
    return { source: "Fantasy Premier League", gameweek, updated_at: new Date(now()).toISOString(), matches };
  }
}
