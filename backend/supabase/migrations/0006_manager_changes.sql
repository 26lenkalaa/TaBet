-- Version 6: managerial changes for the leagues we model (feature F5, research only; tabet/managers.py).
-- Source: Wikipedia season articles' "Managerial changes" tables (CC BY-SA 4.0; credited in
-- docs/DATA_SOURCES.md). Private like the API-Football tables: row-level security on with no policies,
-- so only the secret key can read or write. Safe to run more than once.

create table if not exists public.manager_changes (
  league           text not null,          -- our code: E0, E1, SP1, I1, D1, F1, N1, P1, B1, SC0
  season           text not null,          -- "2023-24"
  team_name        text not null,          -- as Wikipedia writes it
  team_id          text,                   -- our id where we have one (English clubs), else null
  outgoing         text not null default '',
  manner           text not null default '',
  vacancy_date     date,
  incoming         text not null default '',
  interim          boolean not null default false,   -- caretaker or interim appointment
  appointment_date date,
  table_position   text not null default '',
  source           text not null,          -- the Wikipedia article, e.g. "2023-24 Premier League"
  retrieved_at     timestamptz not null default now(),
  primary key (league, season, team_name, outgoing, incoming)
);
create index if not exists manager_changes_team on public.manager_changes (league, team_name, appointment_date);

alter table public.manager_changes enable row level security;
