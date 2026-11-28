import pandas as pd

from tabet import newsletter
from tabet.digest import build
from tests.test_digest import _m, _write


class FakeStore:
    def __init__(self, people, sent=()):
        self.people, self.done, self.claims, self.finished = people, set(sent), [], {}

    def sent(self, issue): return issue in self.done
    def claim(self, issue):
        if issue in self.done: return False
        self.done.add(issue); self.claims.append(issue); return True
    def finish(self, issue, n): self.finished[issue] = n
    def release(self, issue): self.done.discard(issue); self.released = issue
    def active(self): return self.people


class FakeSMTP:
    def __init__(self): self.msgs = []
    def send_message(self, m): self.msgs.append(m)
    def quit(self): pass


def _feed(tmp_path):
    data = _write(tmp_path, [_m(11, "C", "D", "H", [0.6, 0.2, 0.2], [0.5, 0.25, 0.25])])
    build(data)
    return data / "digest.xml"


def test_each_subscriber_gets_their_own_leave_link_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "tabet@example.com")
    feed = _feed(tmp_path)
    issue = newsletter.newest_issue(feed)
    store = FakeStore([{"email": "a@x.com", "token": "t-a"}, {"email": "b@x.com", "token": "t-b"}])
    smtp = FakeSMTP()
    now = issue["published"] + pd.Timedelta(hours=1)
    newsletter.send(now=now, store=store, smtp=smtp, feed=feed)
    assert [m["To"] for m in smtp.msgs] == ["a@x.com", "b@x.com"]
    assert "t-a" in smtp.msgs[0]["List-Unsubscribe"] and "t-b" in smtp.msgs[1].get_body(("html",)).get_content()
    assert store.finished == {issue["guid"]: 2}
    newsletter.send(now=now, store=store, smtp=smtp, feed=feed)          # second run: already sent
    assert len(smtp.msgs) == 2


def test_old_issues_and_missing_config_send_nothing(tmp_path, monkeypatch):
    feed = _feed(tmp_path)
    issue = newsletter.newest_issue(feed)
    store, smtp = FakeStore([{"email": "a@x.com", "token": "t"}]), FakeSMTP()
    newsletter.send(now=issue["published"] + pd.Timedelta(days=10), store=store, smtp=smtp, feed=feed)
    assert smtp.msgs == [] and store.claims == []
    for k in newsletter.ENV:
        monkeypatch.delenv(k, raising=False)
    assert newsletter.send(feed=feed) == 0


def test_a_failed_login_claims_nothing(tmp_path, monkeypatch):
    import smtplib
    monkeypatch.setenv("GMAIL_ADDRESS", "tabet@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    seen = {}
    def boom(sender, pw):
        seen["pw"] = "".join(pw.split())
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
    monkeypatch.setattr(newsletter, "connect", boom)
    feed = _feed(tmp_path)
    issue = newsletter.newest_issue(feed)
    store = FakeStore([{"email": "a@x.com", "token": "t"}])
    assert newsletter.send(now=issue["published"], store=store, feed=feed) == 1
    assert store.claims == [] and not store.sent(issue["guid"])
    assert seen["pw"] == "abcdefghijklmnop"


def test_nothing_delivered_releases_the_claim(tmp_path, monkeypatch):
    import smtplib
    class Refuses(FakeSMTP):
        def send_message(self, m): raise smtplib.SMTPRecipientsRefused({})
    feed = _feed(tmp_path)
    issue = newsletter.newest_issue(feed)
    store = FakeStore([{"email": "a@x.com", "token": "t"}])
    newsletter.send(now=issue["published"], store=store, smtp=Refuses(), feed=feed)
    assert not store.sent(issue["guid"]) and store.finished == {}
