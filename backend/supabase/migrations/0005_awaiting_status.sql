-- Version 5: a match that has kicked off but has no result yet is "awaiting" (tabet/site_data.py).
-- Without this the logger's database copy is refused from the first kickoff of a gameweek until its
-- result arrives (first seen 2026-11-21, GW11). "played" still means a result exists. Safe to rerun.
alter table public.matches drop constraint if exists matches_status_check;
alter table public.matches add constraint matches_status_check
  check (status in ('upcoming', 'awaiting', 'played'));
