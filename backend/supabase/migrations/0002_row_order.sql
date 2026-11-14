-- Version 2: keep forecasts and odds in the order the site lists them (e.g. logged before preview,
-- pre-match odds before closing odds) instead of alphabetical. Safe to run more than once.

alter table public.forecasts   add column if not exists position smallint not null default 0;
alter table public.market_odds add column if not exists position smallint not null default 0;

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
             'p', jsonb_build_array(f.p_home, f.p_draw, f.p_away)) order by f.position, f.key)
    from public.forecasts f where f.match_id = m.id), '[]'::jsonb) as forecasts,
  coalesce((
    select jsonb_agg(jsonb_build_object(
             'key', o.key, 'label', o.label, 'note', o.note,
             'p', jsonb_build_array(o.p_home, o.p_draw, o.p_away)) order by o.position, o.key)
    from public.market_odds o where o.match_id = m.id), '[]'::jsonb) as market
from public.matches m
join public.competitions c on c.id = m.competition_id
join public.teams ht on ht.id = m.home_team_id
join public.teams aw on aw.id = m.away_team_id;

-- The sync function, now also storing each row's position.
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
  select match_id, key, kind, label, note, p_home, p_draw, p_away, coalesce(position, 0)::smallint as position
  from jsonb_to_recordset(payload -> 'forecasts') as f(
    match_id text, key text, kind text, label text, note text,
    p_home numeric, p_draw numeric, p_away numeric, position int);

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

  insert into forecasts (match_id, key, kind, label, note, p_home, p_draw, p_away, position)
  select match_id, key, kind, label, note, p_home, p_draw, p_away, position from incoming_forecasts
  on conflict (match_id, key) do update set
    kind = excluded.kind, label = excluded.label, note = excluded.note,
    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away, position = excluded.position
  where forecasts.kind <> 'logged'
    and (forecasts.kind, forecasts.label, forecasts.note, forecasts.p_home, forecasts.p_draw, forecasts.p_away, forecasts.position)
        is distinct from
        (excluded.kind, excluded.label, excluded.note, excluded.p_home, excluded.p_draw, excluded.p_away, excluded.position);

  drop table if exists incoming_odds;
  create temp table incoming_odds on commit drop as
  select match_id, key, label, note, p_home, p_draw, p_away, coalesce(position, 0)::smallint as position
  from jsonb_to_recordset(payload -> 'market') as o(
    match_id text, key text, label text, note text, p_home numeric, p_draw numeric, p_away numeric, position int);

  delete from market_odds o
  using matches m
  where o.match_id = m.id and m.competition_id = comp and m.season = ssn
    and not exists (select 1 from incoming_odds i where i.match_id = o.match_id and i.key = o.key);

  insert into market_odds (match_id, key, label, note, p_home, p_draw, p_away, position)
  select match_id, key, label, note, p_home, p_draw, p_away, position from incoming_odds
  on conflict (match_id, key) do update set
    label = excluded.label, note = excluded.note,
    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away,
    position = excluded.position, updated_at = now()
  where (market_odds.label, market_odds.note, market_odds.p_home, market_odds.p_draw, market_odds.p_away, market_odds.position)
        is distinct from
        (excluded.label, excluded.note, excluded.p_home, excluded.p_draw, excluded.p_away, excluded.position);

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
