-- Weekly results newsletter: subscribers and the issues already sent.
--
-- The site's sign-up box posts to /api/newsletter, which calls newsletter_join() with the public
-- (publishable) key. Nobody outside the service role can read or change the tables directly: the
-- public key can only call the two functions below, and neither returns anyone's address. The
-- prospective logger (GitHub Action, secret key) reads the active list and emails each new digest
-- item from site/data/digest.xml, recording it in newsletter_sends so an issue goes out once.

create table if not exists public.newsletter_subscribers (
  id        bigint generated always as identity primary key,
  email     text not null unique
            check (email = lower(email) and length(email) <= 254 and email ~ '^[^@\s]+@[^@\s]+\.[^@\s]+$'),
  token     uuid not null unique default gen_random_uuid(),   -- the private key in each "leave" link
  status    text not null default 'active' check (status in ('active', 'left')),
  joined_at timestamptz not null default now(),
  left_at   timestamptz
);
comment on table public.newsletter_subscribers is
  'Weekly results email list. Private: service role only; the public key can only call newsletter_join/newsletter_leave.';

create table if not exists public.newsletter_sends (
  issue      text primary key,                 -- the digest item guid, e.g. tabet-2026-27-gw12
  sent_at    timestamptz not null default now(),
  recipients integer not null default 0
);
comment on table public.newsletter_sends is 'Digest issues already emailed, so each goes out once. Service role only.';

alter table public.newsletter_subscribers enable row level security;
alter table public.newsletter_sends enable row level security;
revoke all on table public.newsletter_subscribers, public.newsletter_sends from public, anon, authenticated;
grant all on table public.newsletter_subscribers, public.newsletter_sends to service_role;

-- Join (or rejoin). Returns nothing, so the endpoint never reveals whether an address was already on
-- the list. At most 100 new sign-ups an hour across everyone, to blunt scripted abuse.
create or replace function public.newsletter_join(p_email text)
returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  e text := lower(btrim(p_email));
begin
  if e is null or length(e) > 254 or e !~ '^[^@\s]+@[^@\s]+\.[^@\s]+$' then
    raise exception 'invalid email' using errcode = '22023';
  end if;
  if (select count(*) from newsletter_subscribers where joined_at > now() - interval '1 hour') >= 100 then
    raise exception 'too many sign-ups, try again later' using errcode = '53400';
  end if;
  insert into newsletter_subscribers (email) values (e)
  on conflict (email) do update
    set status = 'active', left_at = null,
        joined_at = case when newsletter_subscribers.status = 'left' then now() else newsletter_subscribers.joined_at end;
end;
$$;

-- Leave, by the token in the email's link. True if this unsubscribed someone.
create or replace function public.newsletter_leave(p_token uuid)
returns boolean
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
  update newsletter_subscribers set status = 'left', left_at = now()
   where token = p_token and status = 'active';
  return found;
end;
$$;

revoke all on function public.newsletter_join(text), public.newsletter_leave(uuid) from public;
grant execute on function public.newsletter_join(text), public.newsletter_leave(uuid) to anon, authenticated, service_role;
comment on function public.newsletter_join(text) is 'Add or reactivate a newsletter address; reveals nothing about the list.';
comment on function public.newsletter_leave(uuid) is 'Unsubscribe by the token in the leave link.';
