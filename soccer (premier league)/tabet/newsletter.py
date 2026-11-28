"""Email the newest weekly digest (site/data/digest.xml) to the newsletter list, once per issue.

Run by the prospective logger after it refreshes the site data. Subscribers sign up through the site's
box (/api/newsletter -> Supabase, migration 0009); this reads the active list with the secret key and
sends through Gmail's SMTP server with an app password. Each email carries the subscriber's own leave
link and a one-click List-Unsubscribe header. An issue is claimed in newsletter_sends before sending,
so a rerun or an overlapping job never sends it twice.

Environment: SUPABASE_URL, SUPABASE_SECRET_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD. Without them the
command says so and exits 0 (the logger must never fail because of the newsletter).
"""
from __future__ import annotations

import os
import smtplib
import ssl
import xml.etree.ElementTree as ET
from email.message import EmailMessage
from email.utils import formataddr, parsedate_to_datetime
from pathlib import Path

import pandas as pd

from .site_data import SITE_DATA

API = "https://ta-bet-ml-model-edition.vercel.app/api/newsletter/leave"
FRESH = pd.Timedelta(days=4)     # an issue older than this is never sent (e.g. when the list is first set up)
ENV = ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")


def newest_issue(feed: Path = SITE_DATA / "digest.xml") -> dict | None:
    if not feed.exists():
        return None
    item = ET.parse(feed).getroot().find("channel/item")     # the feed lists the newest first
    if item is None:
        return None
    return {"guid": item.findtext("guid"), "title": item.findtext("title"), "html": item.findtext("description"),
            "published": pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate")))}


def message(issue: dict, to: str, token: str, sender: str) -> EmailMessage:
    leave = f"{API}?t={token}"
    m = EmailMessage()
    m["From"] = formataddr(("TaBet weekly results", sender))
    m["To"] = to
    m["Subject"] = issue["title"]
    m["List-Unsubscribe"] = f"<{leave}>"
    m["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    foot = ('<p style="font-size:12px;color:#5B6168">You get this because this address joined TaBet\'s weekly '
            f'results on our site. <a href="{leave}">Leave the list</a>. Not betting advice.</p>')
    m.set_content(f"{issue['title']}\n\nThis email is best viewed as HTML.\nLeave the list: {leave}\n")
    m.add_alternative(f'<div style="font-family:system-ui,-apple-system,sans-serif">{issue["html"]}{foot}</div>',
                      subtype="html")
    return m


class Store:
    """The two newsletter tables, through PostgREST with the secret key."""

    def __init__(self, url: str, key: str):
        from .api_football import SupabaseStore
        self.s = SupabaseStore(url, key, schema="public")

    def sent(self, issue: str) -> bool:
        return bool(self.s._rows("newsletter_sends", "issue", issue=f"eq.{issue}"))

    def claim(self, issue: str) -> bool:
        """Record the issue before sending; False if another run already did."""
        try:
            self.s._req("POST", "/rest/v1/newsletter_sends", f'{{"issue": "{issue}"}}'.encode(),
                        {"Content-Type": "application/json", "Prefer": "return=minimal"})
            return True
        except RuntimeError as e:
            if " 409 " in str(e):
                return False
            raise

    def release(self, issue: str) -> None:
        """Undo a claim when nothing was sent, so the next run tries again."""
        self.s._req("DELETE", f"/rest/v1/newsletter_sends?issue=eq.{issue}", None, {"Prefer": "return=minimal"})

    def finish(self, issue: str, n: int) -> None:
        self.s._req("PATCH", f"/rest/v1/newsletter_sends?issue=eq.{issue}", f'{{"recipients": {n}}}'.encode(),
                    {"Content-Type": "application/json", "Prefer": "return=minimal"})

    def active(self) -> list[dict]:
        return self.s._rows("newsletter_subscribers", "email,token", status="eq.active", order="id")


def connect(sender: str, password: str):
    """Log in to Gmail: implicit TLS on 465, then STARTTLS on 587 if 465 drops the connection. App
    passwords are shown with spaces ("abcd efgh ..."); those are removed."""
    password = "".join(password.split())
    ctx = ssl.create_default_context()
    try:
        s = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=30)
        s.login(sender, password)
        return s
    except smtplib.SMTPServerDisconnected:
        s = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        s.starttls(context=ctx)
        s.login(sender, password)
        return s


def send(dry_run: bool = False, now: pd.Timestamp | None = None, store=None, smtp=None, feed: Path | None = None) -> int:
    missing = [k for k in ENV if not os.environ.get(k, "").strip()]
    if missing and store is None:
        print(f"newsletter: not configured ({', '.join(missing)} missing); nothing sent")
        return 0
    issue = newest_issue(feed) if feed else newest_issue()
    now = now or pd.Timestamp.now(tz="UTC")
    if issue is None:
        print("newsletter: no digest yet")
        return 0
    if now - issue["published"] > FRESH:
        print(f"newsletter: newest issue {issue['guid']} is older than {FRESH.days} days; not sent")
        return 0
    store = store or Store(os.environ["SUPABASE_URL"].strip(), os.environ["SUPABASE_SECRET_KEY"].strip())
    if store.sent(issue["guid"]):
        print(f"newsletter: {issue['guid']} already sent")
        return 0
    people = store.active()
    if dry_run:
        print(f"newsletter: would send {issue['guid']} to {len(people)} subscriber(s)")
        return 0
    if not people:
        print("newsletter: no subscribers yet")
        return 0
    sender = os.environ.get("GMAIL_ADDRESS", "").strip()
    server = smtp
    if server is None:
        try:     # log in before claiming the issue: a failed login must leave it to the next run
            server = connect(sender, os.environ["GMAIL_APP_PASSWORD"])
        except (smtplib.SMTPException, OSError) as e:
            print(f"newsletter: Gmail login failed ({type(e).__name__}: {getattr(e, 'smtp_code', '')}); "
                  "check GMAIL_ADDRESS and GMAIL_APP_PASSWORD. Nothing sent; the next run retries")
            return 1
    if not store.claim(issue["guid"]):
        print(f"newsletter: {issue['guid']} claimed by another run")
        server.quit()
        return 0
    n = 0
    try:
        for p in people:
            try:
                server.send_message(message(issue, p["email"], p["token"], sender))
                n += 1
            except smtplib.SMTPException as e:          # one bad address must not stop the rest
                print(f"newsletter: one address failed ({type(e).__name__})")
    finally:
        try:
            server.quit()
        except Exception:
            pass
        if n:
            store.finish(issue["guid"], n)
        else:
            store.release(issue["guid"])
    print(f"newsletter: sent {issue['guid']} to {n} of {len(people)} subscriber(s)")
    return 0
