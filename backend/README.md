# TaBet backend

An Express API on Vercel, reading from a Supabase (Postgres) database. It is shared by every sport.

```
git (source of truth) ──logger──▶ site data files ──sync──▶ Supabase ──▶ Express API (/api/*)
```

- **Git stays the official record.** The logger commits forecasts to git exactly as before. After each push it copies the current season into Supabase (`scripts/sync-soccer.mjs`). If that copy fails, the log is unaffected.
- **Logged forecasts are append-only in the database too.** A trigger blocks updates and deletes. A sync that would change one fails as a whole and rolls back.
- **The public can only read.** Row-level security allows `select` only. The sync function needs the secret key, which lives only in GitHub secrets.
- **Market odds are a benchmark only.** No model reads them.

## Endpoints

All endpoints are `GET` and return JSON. Data responses are cached at Vercel's edge for 60 s.

| Endpoint | Returns |
|---|---|
| `/api` | the list of endpoints |
| `/api/health` | `status`, and when the database was last synced |
| `/api/sports` | sports |
| `/api/competitions` | competitions, e.g. `premier-league` |
| `/api/competitions/:competition/seasons` | seasons, with match counts |
| `/api/competitions/:competition/seasons/:season/matches?gameweek=&status=` | matches, with forecasts and odds |
| `/api/matches/:id` | one match plus its lineup; ids look like `premier-league:2026-27:arsenal-v-chelsea` |

Errors look like `{"error": {"code": "bad_request", "message": "..."}}`, with status 400, 404, 405, 502 or 503.

## Layout

| Path | What it is |
|---|---|
| `supabase/migrations/0001_init.sql` | tables, views, security rules and the sync function |
| `src/app.js` | the Express app (routes, validation, errors, caching) |
| `src/repo/supabase.js` | database reads, using the publishable (read-only) key |
| `src/soccer-payload.js` | turns a season of site data into database rows |
| `scripts/sync-soccer.mjs` | copies site data into Supabase (`--latest`, `--season`, `--dry-run`) |
| `src/server.js` | local dev server |
| `../api/index.mjs` | Vercel entry point; `vercel.json` sends `/api/*` here |
| `test/` | tests, including the real migration run in an in-memory Postgres (PGlite) |

## Setup (one time)

1. **Create the project.** At supabase.com, make a free project and save the database password.
2. **Create the tables.** In the dashboard, open **SQL Editor → New query**, paste all of `supabase/migrations/0001_init.sql`, and click **Run**.
3. **Find your keys.** Under **Project Settings → API Keys**, note the **Project URL**, the **Publishable key** and a **Secret key**.
4. **Add GitHub secrets.** In the repo, go to **Settings → Secrets and variables → Actions** and add `SUPABASE_URL` and `SUPABASE_SECRET_KEY`.
5. **Add Vercel variables.** In the project, go to **Settings → Environment Variables** and add `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. Then redeploy.
6. **Load the history.** Run one full sync locally: `cd backend && cp .env.example .env`, fill in the values, then `npm run sync`. After that, the logger keeps the current season up to date.

## Local development

```sh
cd backend
npm install
npm test            # 19 tests: API, data conversion, and the SQL in PGlite
npm run dev         # http://localhost:3001/api (needs .env)
npm run sync -- --dry-run
```

## Adding a sport or league

1. Add rows to `sports` and `competitions` in a new migration (`0002_...sql`).
2. Write a converter like `src/soccer-payload.js` that produces the same payload shape.
3. Call `sync_competition_season` with that payload.

The API serves the new data with no route changes.
