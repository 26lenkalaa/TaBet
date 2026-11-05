from pathlib import Path

import pytest

from tabet.ingest import load_all

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "res"

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,AvgH,AvgD,AvgA"


@pytest.fixture(scope="session")
def real():
    return load_all(RES)


@pytest.fixture
def write_csv(tmp_path):
    """Write a small synthetic season file and return its path."""
    def _write(rows: list[str], name: str = "Premier League 26-27.csv", header: str = HEADER, bom: bool = False):
        p = tmp_path / name
        text = header + "\n" + "\n".join(rows) + "\n"
        p.write_text(("﻿" if bom else "") + text, encoding="utf-8")
        return p
    return _write
