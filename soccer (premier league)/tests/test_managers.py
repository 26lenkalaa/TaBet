"""Managerial-changes parser: header variants, footnotes, merged cells, interim flags, and the rows sent
to Supabase. Offline: a small hand-written table in Wikipedia's shape."""
import pandas as pd

from tabet import managers as M

HTML = """
<table><tr><th>Team</th><th>Manager</th><th>Captain</th></tr><tr><td>Chelsea</td><td>X</td><td>Y</td></tr></table>
<table class="wikitable">
<tr><th>Team</th><th>Outgoing manager</th><th>Manner of departure</th><th>Date of vacancy</th>
    <th>Position in table</th><th>Replaced by</th><th>Date of appointment</th></tr>
<tr><td>Chelsea</td><td>Frank Lampard<sup>[127]</sup></td><td>End of interim spell</td><td>28 May 2023</td>
    <td rowspan="2">Pre-season</td><td>Mauricio Pochettino<sup>[128]</sup></td><td>29 May 2023</td></tr>
<tr><td>Tottenham Hotspur</td><td>Ryan Mason</td><td>End of interim spell</td><td>28 May 2023</td>
    <td>Ange Postecoglou</td><td>6 June 2023</td></tr>
<tr><td>Wolverhampton Wanderers</td><td>Julen Lopetegui</td><td>Mutual consent</td><td>8 August 2023</td>
    <td>20th</td><td>Gary O'Neil (caretaker)</td><td>9 August 2023</td></tr>
</table>"""


def test_parses_variants_footnotes_rowspans_and_interims():
    rows = M.changes_from_html(HTML, "E0", "2023–24")
    assert [r["team_id"] for r in rows] == ["chelsea", "tottenham", "wolves"]
    assert rows[0]["outgoing"] == "Frank Lampard" and rows[0]["incoming"] == "Mauricio Pochettino"
    assert rows[1]["incoming"] == "Ange Postecoglou" and rows[1]["appointment_date"] == pd.Timestamp("2023-06-06")
    assert rows[1]["position"] == "Pre-season"                       # carried down by rowspan
    assert rows[2]["interim"] and rows[2]["incoming"] == "Gary O'Neil"


def test_rows_for_supabase():
    df = pd.DataFrame(M.changes_from_html(HTML, "E0", "2023–24"))
    rows = M.to_rows(pd.concat([df, df]))                            # duplicates collapse to one per key
    assert len(rows) == 3 and rows[0]["season"] == "2023-24" and rows[0]["appointment_date"] == "2023-05-29"
    sent = []

    class Store:
        def _insert(self, table, batch, upsert=False):
            sent.append((table, len(batch), upsert))
    assert M.upload(rows, store=Store()) == 3 and sent == [("manager_changes", 3, True)]


def test_build_tries_renamed_articles():
    seen = []

    def fetch(title):
        seen.append(title)
        return HTML if title.endswith("Belgian First Division A") else None
    df = M.build({"B1": ("Belgian Pro League", "Belgian First Division A")}, ["2018–19"], fetch=fetch, pause=0)
    assert seen == ["2018–19 Belgian Pro League", "2018–19 Belgian First Division A"] and len(df) == 3


def test_sealed_rows_drop_result_columns():
    df = pd.DataFrame(M.changes_from_html(HTML, "AUT", "2023–24"))
    rows = M.to_rows(df, sealed=True)
    assert all(r["manner"] == "" and r["table_position"] == "" for r in rows)
    assert rows[0]["incoming"] == "Mauricio Pochettino" and rows[0]["appointment_date"] == "2023-05-29"
