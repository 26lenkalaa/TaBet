"""Embed the latest forecast into the dashboard page.

Usage: python dashboard/build.py OUT_HTML [FORECAST_JSON]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def track_record() -> dict:
    v = json.loads((ROOT / "results/xg_strength/results.json").read_text())
    base = json.loads((ROOT / "results/fpl_v0_v5/results.json").read_text())
    t, f = v["test"]["scores"], v["forward_2026_27"]["scores"]
    mk = base["test"]["market"]
    pre = next(x for k, x in mk.items() if k.startswith("pre-closing"))
    rows = [
        {"label": "Lineup-free model (shown)", "ll": t["V1x adjusted xG strength (replaces raw)"]["log_loss"], "ours": True},
        {"label": "With confirmed XIs", "ll": t["V2x XI strength with adjusted xG"]["log_loss"], "ours": True},
        {"label": "With XIs guessed from history", "ll": t["V1x adjusted xG strength (replaces raw)"]["log_loss"]
         + json.loads((ROOT / "results/projected_xi/results.json").read_text())["test"]["primary: V2x_projected - V1x"]["mean"],
         "ours": True},
        {"label": "Team ratings only", "ll": t["V0 Dixon-Coles"]["log_loss"], "ours": False},
        {"label": "Bookmakers (pre-match)", "ll": pre["log_loss"], "ours": False},
    ]
    note = (f"2024-25 and 2025-26 seasons, {t['V0 Dixon-Coles']['n']} matches. This season so far "
            f"({f['V0 Dixon-Coles']['n']} matches): lineup-free {f['V1x adjusted xG strength (replaces raw)']['log_loss']:.4f}, "
            f"with confirmed XIs {f['V2x XI strength with adjusted xG']['log_loss']:.4f}. "
            "XIs guessed from past selections alone add nothing; the gain needs the real lineup. Scraped predicted XIs "
            "(with team news) have no track record yet: each run is logged and will be scored against confirmed XIs.")
    return {"rows": rows, "note": note}


def main() -> None:
    out = Path(sys.argv[1])
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else max((ROOT / "results/forecasts").glob("forecast_*.json"))
    data = json.loads(src.read_text())
    page = (ROOT / "dashboard/template.html").read_text()
    page = page.replace("__DATA__", json.dumps(data, default=str)).replace("__TRACK__", json.dumps(track_record()))
    out.write_text(page)
    print(f"wrote {out} from {src.name}")


if __name__ == "__main__":
    main()
