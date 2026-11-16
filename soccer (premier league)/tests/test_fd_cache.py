"""football-data cache in Supabase: Supabase first, else download once, clean and upload."""
import pytest

from tabet import fd_cache as fc

CSV = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HST,AST\n"
       b"E1,09/08/08,Reading,Cardiff,1,1,D,5,4,,EXTRA\n"          # a stray trailing field
       b"E1,09/08/08,Burnley,Sheff\xa0Wed,0,2,A,3,6\n")            # a Latin-1 byte


@pytest.fixture
def bucket(monkeypatch):
    store, downloads = {}, []
    monkeypatch.setattr(fc.ss, "configured", lambda: True)
    monkeypatch.setattr(fc.ss, "ensure_bucket", lambda b: None)

    def get(b, p):
        if p not in store:
            raise RuntimeError("Supabase storage GET x: HTTP 404 not_found")
        return store[p]
    monkeypatch.setattr(fc.ss, "get", get)
    monkeypatch.setattr(fc.ss, "put", lambda b, p, d, c: store.__setitem__(p, d) or f"supabase:{b}/{p}")
    monkeypatch.setattr(fc, "_download", lambda d, c: downloads.append((d, c)) or CSV)
    fc.raw_csv.cache_clear()
    yield store, downloads
    fc.raw_csv.cache_clear()


def test_clean_cuts_stray_fields_and_reencodes():
    out = fc.clean(CSV).decode("utf-8")
    lines = out.strip().splitlines()
    assert all(len(l.split(",")) == 9 for l in lines)
    assert "Sheff Wed" in out


def test_downloads_once_then_reads_from_supabase(bucket):
    store, downloads = bucket
    first = fc.raw_csv("E1", "0809")
    assert downloads == [("E1", "0809")] and "history/E1_0809.csv" in store
    fc.raw_csv.cache_clear()
    assert fc.raw_csv("E1", "0809") == first and downloads == [("E1", "0809")]   # served from the bucket


def test_season_parses_without_leaving_files(bucket, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = fc.season("E1", "0809")
    assert len(df) == 2 and set(df["home_id"]) == {"e1_reading", "e1_burnley"}
    assert list(tmp_path.iterdir()) == []
