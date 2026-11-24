-- Version 8: a logged fallback forecast may be superseded by the primary record. Safe to rerun.
--
-- The logger can log a match twice: a fallback record (e.g. 31.5 h before kickoff) and later the
-- primary record (24 h horizon), which supersedes it (results/prospective protocol). The site shows the
-- primary. The database stored the fallback as 'logged' and, since logged rows never change, refused
-- every sync from 2026-11-22 18:30 UTC (Brentford v Everton: pE away 0.2290 fallback, 0.2291 primary),
-- leaving the API stale. The git log was never affected.
--
-- Rules for a logged row (enforced by the trigger and by the sync function):
--   * never deleted;
--   * a row whose role is unknown (stored before this migration) may have its role filled in when the
--     probabilities are unchanged;
--   * a row that is not the primary record may be replaced once, by the primary record;
--   * a primary record never changes.

alter table public.forecasts add column if not exists record_role text;
alter table public.forecasts drop constraint if exists forecasts_record_role_check;
alter table public.forecasts add constraint forecasts_record_role_check
  check (record_role in ('primary', 'fallback') and kind = 'logged' or record_role is null);
comment on column public.forecasts.record_role is
  'Logged forecasts only: primary (24 h record) or fallback (logged earlier); the primary supersedes the fallback';

create or replace function public.logged_change_allowed(
  old_role text, old_h numeric, old_d numeric, old_a numeric,
  new_kind text, new_role text, new_h numeric, new_d numeric, new_a numeric) returns boolean
language sql immutable as $$
  select coalesce(new_kind = 'logged' and (
    (old_role is null and (old_h, old_d, old_a) = (new_h, new_d, new_a))      -- fill in the role
    or (old_role is distinct from 'primary' and new_role = 'primary')         -- primary supersedes
  ), false)                                                                   -- never null: unknown means no
$$;
comment on function public.logged_change_allowed(text, numeric, numeric, numeric, text, text, numeric, numeric, numeric) is
  'Whether a stored logged forecast may change: only to fill in its role, or when the primary record supersedes a fallback.';

create or replace function public.forbid_logged_forecast_change() returns trigger
language plpgsql as $$
begin
  if old.kind = 'logged' then
    if tg_op = 'UPDATE' and public.logged_change_allowed(old.record_role, old.p_home, old.p_draw, old.p_away,
                                                        new.kind, new.record_role, new.p_home, new.p_draw, new.p_away) then
      return new;
    end if;
    raise exception 'logged forecasts are append-only (match %, key %)', old.match_id, old.key
      using errcode = 'P0001';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end $$;

-- The sync function, now storing record_role and letting exactly those changes through.
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
  select match_id, key, kind, record_role, label, note, p_home, p_draw, p_away, coalesce(position, 0)::smallint as position
  from jsonb_to_recordset(payload -> 'forecasts') as f(
    match_id text, key text, kind text, record_role text, label text, note text,
    p_home numeric, p_draw numeric, p_away numeric, position int);

  -- a stored logged forecast never changes, except as public.logged_change_allowed permits
  select count(*) into n_conflicts
  from forecasts f join incoming_forecasts i using (match_id, key)
  where f.kind = 'logged'
    and (i.kind, i.p_home, i.p_draw, i.p_away) is distinct from ('logged', f.p_home, f.p_draw, f.p_away)
    and not public.logged_change_allowed(f.record_role, f.p_home, f.p_draw, f.p_away,
                                         i.kind, i.record_role, i.p_home, i.p_draw, i.p_away);
  if n_conflicts > 0 then
    raise exception '% logged forecast(s) differ from the database; refusing to sync', n_conflicts;
  end if;

  delete from forecasts f
  using matches m
  where f.match_id = m.id and m.competition_id = comp and m.season = ssn and f.kind <> 'logged'
    and not exists (select 1 from incoming_forecasts i where i.match_id = f.match_id and i.key = f.key);

  insert into forecasts (match_id, key, kind, record_role, label, note, p_home, p_draw, p_away, position)
  select match_id, key, kind, record_role, label, note, p_home, p_draw, p_away, position from incoming_forecasts
  on conflict (match_id, key) do update set
    kind = excluded.kind, record_role = excluded.record_role, label = excluded.label, note = excluded.note,
    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away, position = excluded.position
  where (forecasts.kind <> 'logged'
         or public.logged_change_allowed(forecasts.record_role, forecasts.p_home, forecasts.p_draw, forecasts.p_away,
                                         excluded.kind, excluded.record_role, excluded.p_home, excluded.p_draw, excluded.p_away))
    and (forecasts.kind, forecasts.record_role, forecasts.label, forecasts.note, forecasts.p_home, forecasts.p_draw,
         forecasts.p_away, forecasts.position)
        is distinct from
        (excluded.kind, excluded.record_role, excluded.label, excluded.note, excluded.p_home, excluded.p_draw,
         excluded.p_away, excluded.position);

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
comment on function public.sync_competition_season(jsonb) is
  'Replaces one competition-season from a site/data payload in one transaction; refuses to change logged forecasts beyond what logged_change_allowed permits.';
