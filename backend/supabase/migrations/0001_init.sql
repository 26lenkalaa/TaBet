-- TaBet database, version 1.
--
-- Git stays the source of truth: the forecast log and site data are committed by the logger, and
-- backend/scripts/sync-soccer.mjs copies them here. The database is a read copy for the API.
--
-- Rules enforced here:
--   * logged forecasts (kind = 'logged') are append-only: no update, no delete, like the git log;
--   * the public (anon / authenticated roles) can only read; writes need the secret key;
--   * one sync per competition-season runs in a single transaction (sync_competition_season).

-- ---------------------------------------------------------------- reference tables

create table if not exists public.sports (
  id   text primary key,                      -- 'soccer', 'basketball'
  name text not null
);

create table if not exists public.competitions (
  id       text primary key,                  -- 'premier-league'
  sport_id text not null references public.sports (id),
  name     text not null,
  country  text
);

create table if not exists public.teams (
  id       text primary key,                  -- '<sport>:<slug>', e.g. 'soccer:arsenal'
  sport_id text not null references public.sports (id),
  name     text not null,
  unique (sport_id, name)
);

-- ---------------------------------------------------------------- matches

create table if not exists public.matches (
  id             text primary key,            -- '<competition>:<season>:<home>-v-<away>' (stable all season)
  competition_id text not null references public.competitions (id),
  season         text not null check (season ~ '^\d{4}-\d{2}$'),
  gameweek       int check (gameweek between 1 and 60),
  kickoff        timestamptz not null,
  home_team_id   text not null references public.teams (id),
  away_team_id   text not null references public.teams (id),
  status         text not null check (status in ('upcoming', 'played')),
  home_goals     int check (home_goals >= 0),
  away_goals     int check (away_goals >= 0),
  result         text check (result in ('H', 'D', 'A')),
  source_id      text not null,               -- the id the static site uses
  note           text,
  updated_at     timestamptz not null default now(),
  unique (competition_id, season, home_team_id, away_team_id),
  check (home_team_id <> away_team_id),
  check ((status = 'played') = (result is not null))
);
create index if not exists matches_comp_season_kickoff on public.matches (competition_id, season, kickoff);

-- ---------------------------------------------------------------- probabilities

create table if not exists public.forecasts (
  match_id   text not null references public.matches (id) on delete cascade,
  key        text not null,                   -- 'pE', 'p0', 'saved', 'preview', 'V0', ...
  kind       text not null check (kind in ('logged', 'saved', 'preview', 'backtest')),
  label      text not null,
  note       text,
  p_home     numeric(6, 4) not null check (p_home between 0 and 1),
  p_draw     numeric(6, 4) not null check (p_draw between 0 and 1),
  p_away     numeric(6, 4) not null check (p_away between 0 and 1),
  created_at timestamptz not null default now(),
  primary key (match_id, key),
  check (abs(p_home + p_draw + p_away - 1) < 0.001)
);

-- Market odds are a benchmark only; no model reads this table.
create table if not exists public.market_odds (
  match_id   text not null references public.matches (id) on delete cascade,
  key        text not null,                   -- 'pre', 'close', 'latest', 'matched'
  label      text not null,
  note       text,
  p_home     numeric(6, 4) not null check (p_home between 0 and 1),
  p_draw     numeric(6, 4) not null check (p_draw between 0 and 1),
  p_away     numeric(6, 4) not null check (p_away between 0 and 1),
  updated_at timestamptz not null default now(),
  primary key (match_id, key),
  check (abs(p_home + p_draw + p_away - 1) < 0.001)
);

create table if not exists public.lineups (
  match_id   text primary key references public.matches (id) on delete cascade,
  data       jsonb not null,
  updated_at timestamptz not null default now()
);

create table if not exists public.sync_runs (
  id              bigint generated always as identity primary key,
  competition_id  text not null references public.competitions (id),
  season          text not null,
  source_built_at timestamptz,
  n_matches       int not null,
  ran_at          timestamptz not null default now()
);

-- ---------------------------------------------------------------- append-only logged forecasts

create or replace function public.forbid_logged_forecast_change() returns trigger
language plpgsql as $$
begin
  if old.kind = 'logged' then
    raise exception 'logged forecasts are append-only (match %, key %)', old.match_id, old.key
      using errcode = 'P0001';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end $$;

drop trigger if exists forecasts_logged_append_only on public.forecasts;
create trigger forecasts_logged_append_only
  before update or delete on public.forecasts
  for each row execute function public.forbid_logged_forecast_change();

-- ---------------------------------------------------------------- read views for the API

create or replace view public.match_cards with (security_invoker = true) as
select
  m.id,
  m.source_id,
  m.competition_id,
  c.sport_id,
  m.season,
  m.gameweek,
  m.kickoff,
  ht.name as home,
  aw.name as away,
  m.status,
  m.home_goals,
  m.away_goals,
  m.result,
  m.note,
  coalesce((
    select jsonb_agg(jsonb_build_object(
             'key', f.key, 'kind', f.kind, 'label', f.label, 'note', f.note,
             'p', jsonb_build_array(f.p_home, f.p_draw, f.p_away)) order by f.key)
    from public.forecasts f where f.match_id = m.id), '[]'::jsonb) as forecasts,
  coalesce((
    select jsonb_agg(jsonb_build_object(
             'key', o.key, 'label', o.label, 'note', o.note,
             'p', jsonb_build_array(o.p_home, o.p_draw, o.p_away)) order by o.key)
    from public.market_odds o where o.match_id = m.id), '[]'::jsonb) as market
from public.matches m
join public.competitions c on c.id = m.competition_id
join public.teams ht on ht.id = m.home_team_id
join public.teams aw on aw.id = m.away_team_id;

create or replace view public.season_summaries with (security_invoker = true) as
select
  competition_id,
  season,
  count(*)::int                                  as matches,
  count(*) filter (where status = 'played')::int as played,
  min(kickoff)                                   as first_kickoff,
  max(kickoff)                                   as last_kickoff
from public.matches
group by competition_id, season;

-- ---------------------------------------------------------------- row-level security: public read only

alter table public.sports       enable row level security;
alter table public.competitions enable row level security;
alter table public.teams        enable row level security;
alter table public.matches      enable row level security;
alter table public.forecasts    enable row level security;
alter table public.market_odds  enable row level security;
alter table public.lineups      enable row level security;
alter table public.sync_runs    enable row level security;

do $$
declare t text;
begin
  foreach t in array array['sports','competitions','teams','matches','forecasts','market_odds','lineups','sync_runs']
  loop
    execute format('drop policy if exists "public read" on public.%I', t);
    execute format('create policy "public read" on public.%I for select to anon, authenticated using (true)', t);
  end loop;
end $$;

-- ---------------------------------------------------------------- sync (secret key only)

-- Replaces one competition-season with the payload built by sync-soccer.mjs, in one transaction.
-- Logged forecasts are only ever inserted; if the payload disagrees with a stored logged forecast
-- the whole sync fails, because that would mean the git log changed.
create or replace function public.sync_competition_season(payload jsonb) returns jsonb
language plpgsql security definer set search_path = public as $$
declare
  comp  text := payload ->> 'competition';
  ssn   text := payload ->> 'season';
  sport text;
  n_conflicts int;
  n_matches int;
begin
  select sport_id into sport from competitions where id = comp;
  if sport is null then
    raise exception 'unknown competition %', comp;
  end if;

  insert into teams (id, sport_id, name)
  select t.id, sport, t.name
  from jsonb_to_recordset(payload -> 'teams') as t(id text, name text)
  on conflict (id) do update set name = excluded.name where teams.name is distinct from excluded.name;

  insert into matches (id, competition_id, season, gameweek, kickoff, home_team_id, away_team_id,
                       status, home_goals, away_goals, result, source_id, note)
  select x.id, comp, ssn, x.gameweek, x.kickoff, x.home_team_id, x.away_team_id,
         x.status, x.home_goals, x.away_goals, x.result, x.source_id, x.note
  from jsonb_to_recordset(payload -> 'matches') as x(
         id text, gameweek int, kickoff timestamptz, home_team_id text, away_team_id text,
         status text, home_goals int, away_goals int, result text, source_id text, note text)
  on conflict (id) do update set
    gameweek = excluded.gameweek, kickoff = excluded.kickoff, status = excluded.status,
    home_goals = excluded.home_goals, away_goals = excluded.away_goals, result = excluded.result,
    source_id = excluded.source_id, note = excluded.note, updated_at = now()
  where (matches.gameweek, matches.kickoff, matches.status, matches.home_goals, matches.away_goals,
         matches.result, matches.source_id, matches.note)
        is distinct from
        (excluded.gameweek, excluded.kickoff, excluded.status, excluded.home_goals, excluded.away_goals,
         excluded.result, excluded.source_id, excluded.note);

  drop table if exists incoming_forecasts;
  create temp table incoming_forecasts on commit drop as
  select * from jsonb_to_recordset(payload -> 'forecasts') as f(
    match_id text, key text, kind text, label text, note text,
    p_home numeric, p_draw numeric, p_away numeric);

  -- a stored logged forecast must never change
  select count(*) into n_conflicts
  from forecasts f join incoming_forecasts i using (match_id, key)
  where f.kind = 'logged'
    and (i.kind, i.p_home, i.p_draw, i.p_away) is distinct from ('logged', f.p_home, f.p_draw, f.p_away);
  if n_conflicts > 0 then
    raise exception '% logged forecast(s) differ from the database; refusing to sync', n_conflicts;
  end if;

  delete from forecasts f
  using matches m
  where f.match_id = m.id and m.competition_id = comp and m.season = ssn and f.kind <> 'logged'
    and not exists (select 1 from incoming_forecasts i where i.match_id = f.match_id and i.key = f.key);

  insert into forecasts (match_id, key, kind, label, note, p_home, p_draw, p_away)
  select match_id, key, kind, label, note, p_home, p_draw, p_away from incoming_forecasts
  on conflict (match_id, key) do update set
    kind = excluded.kind, label = excluded.label, note = excluded.note,
    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away
  where forecasts.kind <> 'logged'
    and (forecasts.kind, forecasts.label, forecasts.note, forecasts.p_home, forecasts.p_draw, forecasts.p_away)
        is distinct from
        (excluded.kind, excluded.label, excluded.note, excluded.p_home, excluded.p_draw, excluded.p_away);

  drop table if exists incoming_odds;
  create temp table incoming_odds on commit drop as
  select * from jsonb_to_recordset(payload -> 'market') as o(
    match_id text, key text, label text, note text, p_home numeric, p_draw numeric, p_away numeric);

  delete from market_odds o
  using matches m
  where o.match_id = m.id and m.competition_id = comp and m.season = ssn
    and not exists (select 1 from incoming_odds i where i.match_id = o.match_id and i.key = o.key);

  insert into market_odds (match_id, key, label, note, p_home, p_draw, p_away)
  select match_id, key, label, note, p_home, p_draw, p_away from incoming_odds
  on conflict (match_id, key) do update set
    label = excluded.label, note = excluded.note,
    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away, updated_at = now()
  where (market_odds.label, market_odds.note, market_odds.p_home, market_odds.p_draw, market_odds.p_away)
        is distinct from (excluded.label, excluded.note, excluded.p_home, excluded.p_draw, excluded.p_away);

  insert into lineups (match_id, data)
  select l.match_id, l.data from jsonb_to_recordset(payload -> 'lineups') as l(match_id text, data jsonb)
  on conflict (match_id) do update set data = excluded.data, updated_at = now()
  where lineups.data is distinct from excluded.data;

  select jsonb_array_length(payload -> 'matches') into n_matches;
  insert into sync_runs (competition_id, season, source_built_at, n_matches)
  values (comp, ssn, (payload ->> 'built_at')::timestamptz, n_matches);

  return jsonb_build_object('competition', comp, 'season', ssn, 'matches', n_matches);
end $$;

revoke all on function public.sync_competition_season(jsonb) from public;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    execute 'revoke all on function public.sync_competition_season(jsonb) from anon, authenticated';
  end if;
  if exists (select 1 from pg_roles where rolname = 'service_role') then
    execute 'grant execute on function public.sync_competition_season(jsonb) to service_role';
  end if;
end $$;

-- ---------------------------------------------------------------- seed

insert into public.sports (id, name) values ('soccer', 'Soccer')
on conflict (id) do nothing;
insert into public.competitions (id, sport_id, name, country)
values ('premier-league', 'soccer', 'Premier League', 'England')
on conflict (id) do nothing;
