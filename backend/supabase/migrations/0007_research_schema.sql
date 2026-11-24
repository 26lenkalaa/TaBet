-- Version 7: tidy the database for the system-design review. Safe to run more than once.
--
-- 1. Two schemas with one job each:
--      public    what the site's API reads (matches, forecasts, market_odds, lineups, teams, ...);
--      research  private research data (the API-Football bookkeeping and managerial changes).
--    Nothing is deleted or rewritten; the research tables move as they are. The research schema is
--    reachable through the Data API only with the secret key (service_role); anon and authenticated
--    get no access to it at all.
-- 2. One league lookup, research.leagues, linking the three id systems in use: football-data
--    codes (E0, SP1, ...; our own codes for cups), API-Football ids (39, 140, ...) and the site's
--    competition ids ('premier-league'). The research tables now reference it. API-Football's
--    integer seasons (2024 = 2024-25) get a season_label in the "2024-25" form used everywhere else.
-- 3. Every table, view and function gets a description, shown in the Supabase dashboard.

-- ---------------------------------------------------------------- 1. the research schema

create schema if not exists research;
comment on schema research is
  'Private research data: never read by the site. Reachable only with the secret key (service_role).';

revoke all on schema research from public;
grant usage on schema research to service_role;

alter table if exists public.apif_lists      set schema research;
alter table if exists public.apif_fixtures   set schema research;
alter table if exists public.apif_blocked    set schema research;
alter table if exists public.apif_requests   set schema research;
alter table if exists public.manager_changes set schema research;

-- ---------------------------------------------------------------- 2. one league lookup

create table if not exists research.leagues (
  code           text primary key,                -- football-data code, or our own for cups
  name           text not null,
  country        text,                            -- null for European competitions
  kind           text not null check (kind in ('league', 'cup', 'europe')),
  tier           smallint check (tier between 1 and 5),   -- leagues only
  calendar       text not null default 'aug-may' check (calendar in ('aug-may', 'calendar')),
  apif_id        int unique,                      -- API-Football league id, where collected
  competition_id text references public.competitions (id)  -- the site's id, where shown on the site
);

insert into research.leagues (code, name, country, kind, tier, calendar, apif_id, competition_id) values
  -- leagues we model or have used (football-data codes)
  ('E0',   'Premier League',        'England',     'league', 1, 'aug-may',   39, 'premier-league'),
  ('E1',   'Championship',          'England',     'league', 2, 'aug-may',   40, null),
  ('E2',   'League One',            'England',     'league', 3, 'aug-may', null, null),
  ('E3',   'League Two',            'England',     'league', 4, 'aug-may', null, null),
  ('EC',   'National League',       'England',     'league', 5, 'aug-may', null, null),
  ('SP1',  'La Liga',               'Spain',       'league', 1, 'aug-may',  140, null),
  ('SP2',  'Segunda División',      'Spain',       'league', 2, 'aug-may',  141, null),
  ('I1',   'Serie A',               'Italy',       'league', 1, 'aug-may',  135, null),
  ('I2',   'Serie B',               'Italy',       'league', 2, 'aug-may',  136, null),
  ('D1',   'Bundesliga',            'Germany',     'league', 1, 'aug-may',   78, null),
  ('D2',   '2. Bundesliga',         'Germany',     'league', 2, 'aug-may',   79, null),
  ('F1',   'Ligue 1',               'France',      'league', 1, 'aug-may',   61, null),
  ('F2',   'Ligue 2',               'France',      'league', 2, 'aug-may',   62, null),
  ('N1',   'Eredivisie',            'Netherlands', 'league', 1, 'aug-may',   88, null),
  ('P1',   'Primeira Liga',         'Portugal',    'league', 1, 'aug-may',   94, null),
  ('B1',   'Belgian Pro League',    'Belgium',     'league', 1, 'aug-may',  144, null),
  ('SC0',  'Scottish Premiership',  'Scotland',    'league', 1, 'aug-may',  179, null),
  ('SC1',  'Scottish Championship', 'Scotland',    'league', 2, 'aug-may', null, null),
  ('SC2',  'Scottish League One',   'Scotland',    'league', 3, 'aug-may', null, null),
  ('SC3',  'Scottish League Two',   'Scotland',    'league', 4, 'aug-may', null, null),
  ('G1',   'Super League Greece',   'Greece',      'league', 1, 'aug-may',  197, null),
  ('T1',   'Süper Lig',             'Turkey',      'league', 1, 'aug-may',  203, null),
  ('AUT',  'Austrian Bundesliga',   'Austria',     'league', 1, 'aug-may',  218, null),
  ('SWZ',  'Swiss Super League',    'Switzerland', 'league', 1, 'aug-may',  207, null),
  ('DNK',  'Danish Superliga',      'Denmark',     'league', 1, 'aug-may',  119, null),
  -- the second sealed holdout (ledger note of 2026-11-24): registered here, not collected
  ('ARG',  'Liga Profesional',      'Argentina',   'league', 1, 'calendar', null, null),
  ('BRA',  'Série A',               'Brazil',      'league', 1, 'calendar', null, null),
  ('CHN',  'Super League',          'China',       'league', 1, 'calendar', null, null),
  ('FIN',  'Veikkausliiga',         'Finland',     'league', 1, 'calendar', null, null),
  ('IRL',  'Premier Division',      'Ireland',     'league', 1, 'calendar', null, null),
  ('JPN',  'J1 League',             'Japan',       'league', 1, 'calendar', null, null),
  ('MEX',  'Liga MX',               'Mexico',      'league', 1, 'aug-may',  null, null),
  ('NOR',  'Eliteserien',           'Norway',      'league', 1, 'calendar', null, null),
  ('POL',  'Ekstraklasa',           'Poland',      'league', 1, 'aug-may',  null, null),
  ('ROU',  'Liga I',                'Romania',     'league', 1, 'aug-may',  null, null),
  ('RUS',  'Premier League',        'Russia',      'league', 1, 'aug-may',  null, null),
  ('SWE',  'Allsvenskan',           'Sweden',      'league', 1, 'calendar', null, null),
  ('USA',  'MLS',                   'USA',         'league', 1, 'calendar', null, null),
  -- European competitions
  ('UCL',  'Champions League',      null,          'europe', null, 'aug-may',   2, null),
  ('UEL',  'Europa League',         null,          'europe', null, 'aug-may',   3, null),
  ('UECL', 'Conference League',     null,          'europe', null, 'aug-may', 848, null),
  -- domestic cups (our own codes: league code + C, or the cup's usual short name)
  ('FAC',  'FA Cup',                'England',     'cup', null, 'aug-may',   45, null),
  ('EFLC', 'League Cup',            'England',     'cup', null, 'aug-may',   48, null),
  ('SP1C', 'Copa del Rey',          'Spain',       'cup', null, 'aug-may',  143, null),
  ('I1C',  'Coppa Italia',          'Italy',       'cup', null, 'aug-may',  137, null),
  ('D1C',  'DFB Pokal',             'Germany',     'cup', null, 'aug-may',   81, null),
  ('F1C',  'Coupe de France',       'France',      'cup', null, 'aug-may',   66, null),
  ('N1C',  'KNVB Beker',            'Netherlands', 'cup', null, 'aug-may',   90, null),
  ('P1C',  'Taça de Portugal',      'Portugal',    'cup', null, 'aug-may',   96, null),
  ('B1C',  'Belgian Cup',           'Belgium',     'cup', null, 'aug-may',  147, null),
  ('SC0C', 'Scottish FA Cup',       'Scotland',    'cup', null, 'aug-may',  181, null),
  ('SC0L', 'Scottish League Cup',   'Scotland',    'cup', null, 'aug-may',  185, null),
  ('AUTC', 'Austrian Cup',          'Austria',     'cup', null, 'aug-may',  220, null),
  ('SWZC', 'Schweizer Cup',         'Switzerland', 'cup', null, 'aug-may',  209, null),
  ('DNKC', 'DBU Pokalen',           'Denmark',     'cup', null, 'aug-may',  121, null),
  ('G1C',  'Greek Cup',             'Greece',      'cup', null, 'aug-may',  199, null),
  ('T1C',  'Türkiye Kupası',        'Turkey',      'cup', null, 'aug-may',  206, null)
on conflict (code) do update set
  name = excluded.name, country = excluded.country, kind = excluded.kind, tier = excluded.tier,
  calendar = excluded.calendar, apif_id = excluded.apif_id, competition_id = excluded.competition_id;

-- one season format: "2024-25" beside API-Football's starting year
alter table research.apif_lists add column if not exists season_label text
  generated always as (season::text || '-' || lpad(((season + 1) % 100)::text, 2, '0')) stored;
alter table research.apif_fixtures add column if not exists season_label text
  generated always as (season::text || '-' || lpad(((season + 1) % 100)::text, 2, '0')) stored;

alter table research.manager_changes drop constraint if exists manager_changes_season_format;
alter table research.manager_changes add constraint manager_changes_season_format
  check (season ~ '^\d{4}-\d{2}$');

-- links to the lookup (every value in use was checked against it before this was written)
alter table research.apif_lists drop constraint if exists apif_lists_league_fkey;
alter table research.apif_lists add constraint apif_lists_league_fkey
  foreign key (league) references research.leagues (apif_id);
alter table research.apif_fixtures drop constraint if exists apif_fixtures_league_fkey;
alter table research.apif_fixtures add constraint apif_fixtures_league_fkey
  foreign key (league) references research.leagues (apif_id);
alter table research.apif_blocked drop constraint if exists apif_blocked_league_fkey;
alter table research.apif_blocked add constraint apif_blocked_league_fkey
  foreign key (league) references research.leagues (apif_id);
alter table research.manager_changes drop constraint if exists manager_changes_league_fkey;
alter table research.manager_changes add constraint manager_changes_league_fkey
  foreign key (league) references research.leagues (code);

-- the lookup is private like the rest of the schema
alter table research.leagues enable row level security;

-- ---------------------------------------------------------------- access to the research schema

grant all on all tables in schema research to service_role;
grant usage, select on all sequences in schema research to service_role;
alter default privileges in schema research grant all on tables to service_role;
alter default privileges in schema research grant usage, select on sequences to service_role;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    execute 'revoke all on schema research from anon, authenticated';
    execute 'revoke all on all tables in schema research from anon, authenticated';
  end if;
end $$;

-- ---------------------------------------------------------------- 3. descriptions

comment on table public.sports is 'Sports shown on the site. Read copy of git (site/data), public read.';
comment on table public.competitions is 'Competitions shown on the site, e.g. premier-league. Public read.';
comment on table public.teams is 'Teams shown on the site; id is <sport>:<slug>. Public read.';
comment on table public.matches is
  'One row per fixture shown on the site. Synced from site/data by sync_competition_season; git is the source of truth.';
comment on column public.matches.id is '<competition>:<season>:<home>-v-<away>, stable all season';
comment on column public.matches.status is 'upcoming, awaiting (kicked off, no result yet) or played';
comment on column public.matches.source_id is 'The id the static site uses for this match';
comment on table public.forecasts is
  'Home/draw/away probabilities per match. kind = logged rows are the live test record: append-only (trigger).';
comment on column public.forecasts.key is 'Which forecast: pE, p0, saved, preview, V0, ...';
comment on column public.forecasts.kind is 'logged (frozen live forecast), saved, preview or backtest';
comment on column public.forecasts.position is 'Order the site lists forecasts in';
comment on table public.market_odds is
  'Margin-free market probabilities. A benchmark only: no model reads this table.';
comment on column public.market_odds.key is 'pre, close, latest or matched (same time as our forecast)';
comment on table public.lineups is 'Predicted and confirmed lineups per match, as the site shows them (JSON).';
comment on table public.sync_runs is 'One row per sync of a competition-season from git, for monitoring.';
comment on view public.match_cards is 'What the API serves: a match with its forecasts and market odds.';
comment on view public.season_summaries is 'Match counts and date range per competition-season.';
comment on function public.sync_competition_season(jsonb) is
  'Replaces one competition-season from a site/data payload in one transaction; refuses to change logged forecasts.';

comment on table research.leagues is
  'Every league and cup we use, linking football-data codes, API-Football ids and the site''s competition ids.';
comment on column research.leagues.code is 'football-data code (E0, SP1, ...) or our own for cups and Europe';
comment on column research.leagues.calendar is 'aug-may, or calendar (season 2015 is labelled 2015-16)';
comment on table research.apif_lists is
  'API-Football fixture lists collected, one per league-season. The raw JSON is in Storage bucket api-football at object_path.';
comment on column research.apif_lists.season is 'API-Football season: the starting year (2024 = 2024-25)';
comment on table research.apif_fixtures is
  'API-Football fixtures whose full details (lineups, team and player stats) are stored in Storage at object_path.';
comment on column research.apif_fixtures.season is 'API-Football season: the starting year (2024 = 2024-25)';
comment on table research.apif_blocked is 'League-seasons the API plan refused, so they are not requested again.';
comment on table research.apif_requests is 'One row per API-Football request (never the key), for quota tracking.';
comment on table research.manager_changes is
  'Managerial changes from Wikipedia season articles (CC BY-SA 4.0), for research feature F5.';
comment on column research.manager_changes.source is 'The Wikipedia article the row comes from';
