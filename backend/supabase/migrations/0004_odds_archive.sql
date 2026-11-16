-- Version 4: private bucket for The Odds API's raw responses and per-bookmaker prices
-- (amendment A8 item 6). Only the secret key (the logger) can read or write it.
-- The bucket was created on 2026-11-16 via the Storage API; this keeps a record. Safe to rerun.
insert into storage.buckets (id, name, public)
values ('odds-archive', 'odds-archive', false)
on conflict (id) do nothing;
