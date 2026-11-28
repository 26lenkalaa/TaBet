// TaBet API (Express). Read-only over the Supabase read copy, except the newsletter sign-up and
// leave endpoints, which can only call two narrow database functions (migration 0009).
// The data layer is passed in as `repo`, so tests can use an in-memory fake.
import express from "express";
import { cacheSeconds, LiveError } from "./live.js";

const SLUG = /^[a-z0-9][a-z0-9-]{0,63}$/;
const SEASON = /^\d{4}-\d{2}$/;
const MATCH_ID = /^[a-z0-9-]+:\d{4}-\d{2}:[a-z0-9-]+-v-[a-z0-9-]+$/;
const STATUSES = new Set(["upcoming", "awaiting", "played"]);
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SITE = "https://ta-bet-ml-model-edition.vercel.app";
const WRITES = new Set(["/api/newsletter", "/api/newsletter/leave"]);   // the only POST endpoints

// a small standalone page for the leave link (opened from an email, so it can't rely on the site's CSS)
function page(title, body) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>${title} | TaBet</title>
<style>body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#F3F4F6;color:#15181C}
main{max-width:480px;margin:12vh auto;padding:32px 24px;background:#fff;border:1px solid #E3E5E8;border-radius:16px}
h1{font-size:22px;margin:0 0 8px}p{color:#5B6168;line-height:1.5}a{color:#1F7A4C}
button{min-height:44px;padding:0 18px;border:0;border-radius:12px;background:#1F7A4C;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
@media (prefers-color-scheme: dark){body{background:#111418;color:#F1F2F4}main{background:#1A1E23;border-color:#2A2F36}p{color:#A4AAB2}}</style>
</head><body><main><h1>${title}</h1>${body}</main></body></html>`;
}

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

export function createApp({ repo, live = null, log = console }) {
  const app = express();
  app.disable("x-powered-by");
  app.set("etag", "strong");

  app.use((req, res, next) => {
    res.set("Access-Control-Allow-Origin", "*"); // public, read-only data
    res.set("X-Content-Type-Options", "nosniff");
    if (req.method !== "GET" && req.method !== "HEAD" && req.method !== "OPTIONS" && !(req.method === "POST" && WRITES.has(req.path))) {
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
        "GET /api/competitions/:competition/seasons/:season/lineups",
        "GET /api/matches/:id",
        "GET /api/live/premier-league",
        "POST /api/newsletter {email}",
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
    const [seasons, last] = await Promise.all([repo.seasons(competition), repo.lastSync()]);
    cached(res);
    res.json({ meta: { built_at: last?.source_built_at ?? null, synced_at: last?.ran_at ?? null }, data: seasons });
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
      if (!STATUSES.has(req.query.status)) throw bad("status must be upcoming, awaiting or played");
      filters.status = req.query.status;
    }
    if (!(await repo.competition(competition))) throw notFound(`unknown competition ${competition}`);
    const rows = await repo.matches(competition, season, filters);
    cached(res);
    res.json({ competition, season, count: rows.length, data: rows.map((r) => toMatch(r)) });
  }));

  api.get("/competitions/:competition/seasons/:season/lineups", wrap(async (req, res) => {
    const { competition, season } = req.params;
    if (!SLUG.test(competition)) throw bad("competition must be a lowercase slug, e.g. premier-league");
    if (!SEASON.test(season)) throw bad("season must look like 2026-27");
    if (!(await repo.competition(competition))) throw notFound(`unknown competition ${competition}`);
    const rows = await repo.lineups(competition, season);
    cached(res, 300);
    res.json({ competition, season, count: rows.length, data: rows });
  }));

  // Live scores for the current gameweek (Fantasy Premier League; display only, never a model input).
  api.get("/live/:competition", wrap(async (req, res) => {
    const { competition } = req.params;
    if (competition !== "premier-league") throw notFound("live scores are available for premier-league only");
    if (!live) throw new HttpError(503, "unavailable", "live scores are not configured");
    let data;
    try {
      data = await live();
    } catch (e) {
      if (!(e instanceof LiveError)) throw e;
      const err = new HttpError(502, "upstream", `Live scores are unavailable right now: ${e.message}`);
      err.expose = true;
      throw err;
    }
    const s = cacheSeconds(data.matches);
    res.set("Cache-Control", `public, max-age=0, s-maxage=${s}, stale-while-revalidate=${s}`);
    res.json(data);
  }));

  // Newsletter sign-up from the site's box. Same answer whether or not the address was already on the
  // list. "website" is a hidden field people never fill in; bots that do are quietly ignored.
  api.post("/newsletter", express.json({ limit: "2kb" }), wrap(async (req, res) => {
    res.set("Cache-Control", "no-store");
    const email = String(req.body?.email ?? "").trim().toLowerCase();
    if (req.body?.website) return res.json({ ok: true });
    if (!email || email.length > 254 || !EMAIL.test(email)) throw bad("That doesn't look like an email address.");
    if (!repo.newsletterJoin) throw new HttpError(503, "unavailable", "Sign-ups are not configured.");
    await repo.newsletterJoin(email);
    res.json({ ok: true });
  }));

  // The link in each email opens a confirm page (GET); only the button (POST) unsubscribes, so link
  // scanners that open every URL in an email can't unsubscribe anyone. Mail apps' one-click
  // unsubscribe (RFC 8058) POSTs here directly.
  api.get("/newsletter/leave", (req, res) => {
    res.set("Cache-Control", "no-store");
    const t = String(req.query.t ?? "");
    if (!UUID.test(t)) return res.status(400).type("html").send(page("Link not recognised", `<p>This leave link is incomplete. Use the link at the bottom of any TaBet email.</p>`));
    res.type("html").send(page("Leave TaBet's weekly results?",
      `<p>You'll stop getting the email after each gameweek. You can rejoin any time on <a href="${SITE}/">the site</a>.</p>
<form method="post" action="/api/newsletter/leave?t=${t}"><button type="submit">Leave the list</button></form>`));
  });
  api.post("/newsletter/leave", express.urlencoded({ extended: false, limit: "1kb" }), wrap(async (req, res) => {
    res.set("Cache-Control", "no-store");
    const t = String(req.query.t ?? "");
    if (!UUID.test(t)) throw bad("missing or malformed token");
    if (!repo.newsletterLeave) throw new HttpError(503, "unavailable", "The newsletter is not configured.");
    await repo.newsletterLeave(t);
    res.type("html").send(page("You've left the list",
      `<p>No more weekly results emails. Changed your mind? Rejoin on <a href="${SITE}/">the site</a>.</p>`));
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
