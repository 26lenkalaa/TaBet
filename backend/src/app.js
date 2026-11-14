// TaBet API (Express). Read-only: every endpoint is a GET over the Supabase read copy.
// The data layer is passed in as `repo`, so tests can use an in-memory fake.
import express from "express";

const SLUG = /^[a-z0-9][a-z0-9-]{0,63}$/;
const SEASON = /^\d{4}-\d{2}$/;
const MATCH_ID = /^[a-z0-9-]+:\d{4}-\d{2}:[a-z0-9-]+-v-[a-z0-9-]+$/;
const STATUSES = new Set(["upcoming", "played"]);

export class HttpError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const bad = (message) => new HttpError(400, "bad_request", message);
const notFound = (message) => new HttpError(404, "not_found", message);

// Data changes at most every 30 minutes; let Vercel's edge cache absorb repeat requests.
function cached(res, seconds = 60) {
  res.set("Cache-Control", `public, max-age=0, s-maxage=${seconds}, stale-while-revalidate=300`);
}

// Same shape as a match in the static site's data files, plus stable ids.
export function toMatch(row, lineup) {
  const out = {
    id: row.id,
    source_id: row.source_id,
    competition: row.competition_id,
    season: row.season,
    gameweek: row.gameweek,
    kickoff: row.kickoff,
    home: row.home,
    away: row.away,
    status: row.status,
    score: row.home_goals == null ? null : `${row.home_goals}-${row.away_goals}`,
    result: row.result,
    note: row.note,
    forecasts: (row.forecasts ?? []).map((f) => ({ ...f, p: f.p.map(Number) })),
    market: (row.market ?? []).map((o) => ({ ...o, p: o.p.map(Number) })),
  };
  if (lineup !== undefined) out.lineup = lineup;
  return out;
}

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res)).catch(next);

export function createApp({ repo, log = console }) {
  const app = express();
  app.disable("x-powered-by");
  app.set("etag", "strong");

  app.use((req, res, next) => {
    res.set("Access-Control-Allow-Origin", "*"); // public, read-only data
    res.set("X-Content-Type-Options", "nosniff");
    if (req.method !== "GET" && req.method !== "HEAD" && req.method !== "OPTIONS") {
      return next(new HttpError(405, "method_not_allowed", "The TaBet API is read-only."));
    }
    next();
  });

  const api = express.Router();

  api.get("/", (req, res) => {
    cached(res, 3600);
    res.json({
      name: "TaBet API",
      version: 1,
      note: "Forecasts are TaBet's; market odds are a benchmark only.",
      endpoints: [
        "GET /api/health",
        "GET /api/sports",
        "GET /api/competitions",
        "GET /api/competitions/:competition/seasons",
        "GET /api/competitions/:competition/seasons/:season/matches?gameweek=&status=",
        "GET /api/matches/:id",
      ],
    });
  });

  api.get("/health", wrap(async (req, res) => {
    res.set("Cache-Control", "no-store");
    const last = await repo.lastSync();
    res.json({ status: "ok", time: new Date().toISOString(), last_sync: last });
  }));

  api.get("/sports", wrap(async (req, res) => {
    cached(res, 3600);
    res.json({ data: await repo.sports() });
  }));

  api.get("/competitions", wrap(async (req, res) => {
    cached(res, 3600);
    res.json({ data: await repo.competitions() });
  }));

  api.get("/competitions/:competition/seasons", wrap(async (req, res) => {
    const { competition } = req.params;
    if (!SLUG.test(competition)) throw bad("competition must be a lowercase slug, e.g. premier-league");
    if (!(await repo.competition(competition))) throw notFound(`unknown competition ${competition}`);
    cached(res);
    res.json({ data: await repo.seasons(competition) });
  }));

  api.get("/competitions/:competition/seasons/:season/matches", wrap(async (req, res) => {
    const { competition, season } = req.params;
    if (!SLUG.test(competition)) throw bad("competition must be a lowercase slug, e.g. premier-league");
    if (!SEASON.test(season)) throw bad("season must look like 2026-27");
    const filters = {};
    if (req.query.gameweek !== undefined) {
      const gw = Number(req.query.gameweek);
      if (!Number.isInteger(gw) || gw < 1 || gw > 60) throw bad("gameweek must be a whole number from 1 to 60");
      filters.gameweek = gw;
    }
    if (req.query.status !== undefined) {
      if (!STATUSES.has(req.query.status)) throw bad("status must be upcoming or played");
      filters.status = req.query.status;
    }
    if (!(await repo.competition(competition))) throw notFound(`unknown competition ${competition}`);
    const rows = await repo.matches(competition, season, filters);
    cached(res);
    res.json({ competition, season, count: rows.length, data: rows.map((r) => toMatch(r)) });
  }));

  api.get("/matches/:id", wrap(async (req, res) => {
    const { id } = req.params;
    if (!MATCH_ID.test(id)) throw bad("id must look like premier-league:2026-27:arsenal-v-chelsea");
    const row = await repo.match(id);
    if (!row) throw notFound(`unknown match ${id}`);
    const lineup = await repo.lineup(id);
    cached(res);
    res.json({ data: toMatch(row, lineup ?? null) });
  }));

  api.use((req, res, next) => next(notFound(`no endpoint ${req.method} ${req.originalUrl}; see GET /api`)));

  app.use("/api", api);
  app.use((req, res, next) => next(notFound("not found")));

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    const status = err.status ?? 500;
    if (status >= 500) log.error(err);
    res.set("Cache-Control", "no-store");
    res.status(status).json({
      error: {
        code: err.code ?? "internal_error",
        message: status >= 500 && !err.expose ? "Something went wrong on our side." : err.message,
      },
    });
  });

  return app;
}
