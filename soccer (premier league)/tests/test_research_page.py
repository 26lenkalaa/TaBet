import json
import re
import shutil
from pathlib import Path

from tabet import research_page as R

SITE_DATA = Path(__file__).resolve().parents[2] / "site" / "data"


def test_builds_from_the_site_data_and_holds_no_addresses(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(SITE_DATA, data, ignore=shutil.ignore_patterns("research.json"))
    out = R.build(data)
    b = out["backtest"]
    assert b["n"] > 1000 and b["vs_close"]["ci_95"][0] < b["vs_close"]["mean"] < b["vs_close"]["ci_95"][1]
    assert {c["model"] for c in b["calibration"]} == {"TaBet (V2x)", "Closing odds"}
    assert out["live"]["logged"] <= out["live"]["played"]
    text = (data / "research.json").read_text()
    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)   # never an email address
    json.loads(text)
