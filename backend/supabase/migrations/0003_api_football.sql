-- Version 3: storage for the API-Football collector (training data only).
-- Raw API responses go to a private Storage bucket; these tables track what has been collected.
-- Nothing here is public: row-level security is on with no policies, so only the secret key
-- (used by the GitHub workflow) can read or write. Safe to run more than once.

-- private bucket for the gzipped raw responses
insert into storage.buckets (id, name, public)
values ('api-football', 'api-football', false)
on conflict (id) do nothing;

-- one row per league-season fixture list that has been downloaded
create table if not exists public.apif_lists (
  league      int  not null,
  season      int  not null,
  object_path text not null,
  n_fixtures  int  not null,
  fetched_at  timestamptz not null default now(),
  primary key (league, season)
);

-- one row per fixture whose full details (lineups, team and player stats) are stored
create table if not exists public.apif_fixtures (
  fixture_id  bigint primary key,
  league      int  not null,
  season      int  not null,
  object_path text not null,
  fetched_at  timestamptz not null default now()
);
create index if not exists apif_fixtures_league_season on public.apif_fixtures (league, season);

-- league-seasons the plan does not cover, so they are never requested again
create table if not exists public.apif_blocked (
  league     int  not null,
  season     int  not null,
  reason     text not null,
  noted_at   timestamptz not null default now(),
  primary key (league, season)
);

-- one row per API request (never the key)
create table if not exists public.apif_requests (
  id        bigint generated always as identity primary key,
  utc       timestamptz not null default now(),
  endpoint  text not null,
  params    jsonb,
  http      int,
  results   int,
  error     text
);

alter table public.apif_lists    enable row level security;
alter table public.apif_fixtures enable row level security;
alter table public.apif_blocked  enable row level security;
alter table public.apif_requests enable row level security;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    execute 'revoke all on public.apif_lists, public.apif_fixtures, public.apif_blocked, public.apif_requests from anon, authenticated';
  end if;
end $$;
