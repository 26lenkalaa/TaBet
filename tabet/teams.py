"""Canonical team IDs and the alias table used to resolve source-specific names."""

from __future__ import annotations

# Canonical ID -> display name. Includes recent Premier League clubs and a few
# earlier ones, which will be needed for promoted-team priors later.
TEAMS: dict[str, str] = {
    "arsenal": "Arsenal",
    "aston_villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton & Hove Albion",
    "burnley": "Burnley",
    "cardiff": "Cardiff City",
    "chelsea": "Chelsea",
    "coventry": "Coventry City",
    "crystal_palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "huddersfield": "Huddersfield Town",
    "hull": "Hull City",
    "ipswich": "Ipswich Town",
    "leeds": "Leeds United",
    "leicester": "Leicester City",
    "liverpool": "Liverpool",
    "luton": "Luton Town",
    "man_city": "Manchester City",
    "man_united": "Manchester United",
    "middlesbrough": "Middlesbrough",
    "newcastle": "Newcastle United",
    "norwich": "Norwich City",
    "nottm_forest": "Nottingham Forest",
    "sheffield_united": "Sheffield United",
    "southampton": "Southampton",
    "stoke": "Stoke City",
    "sunderland": "Sunderland",
    "swansea": "Swansea City",
    "tottenham": "Tottenham Hotspur",
    "watford": "Watford",
    "west_brom": "West Bromwich Albion",
    "west_ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
}

# Extra spellings seen in sources (football-data.co.uk names first). Display
# names and IDs resolve automatically, so they aren't listed here.
_EXTRA_ALIASES: dict[str, str] = {
    "Brighton": "brighton",
    "Brighton and Hove Albion": "brighton",
    "Cardiff": "cardiff",
    "Coventry": "coventry",
    "Huddersfield": "huddersfield",
    "Hull": "hull",
    "Ipswich": "ipswich",
    "Leeds": "leeds",
    "Leicester": "leicester",
    "Luton": "luton",
    "Man City": "man_city",
    "Man United": "man_united",
    "Man Utd": "man_united",
    "Newcastle": "newcastle",
    "Norwich": "norwich",
    "Nott'm Forest": "nottm_forest",
    "Nottingham": "nottm_forest",
    "Sheffield Utd": "sheffield_united",
    "Spurs": "tottenham",
    "Stoke": "stoke",
    "Swansea": "swansea",
    "Tottenham": "tottenham",
    "West Brom": "west_brom",
    "West Ham": "west_ham",
    "Wolverhampton": "wolves",
    "AFC Bournemouth": "bournemouth",
}


def _key(name: str) -> str:
    return " ".join(name.strip().lower().split())


_ALIASES: dict[str, str] = {}
for _id, _name in TEAMS.items():
    _ALIASES[_key(_id)] = _id
    _ALIASES[_key(_name)] = _id
for _alias, _id in _EXTRA_ALIASES.items():
    if _id not in TEAMS:
        raise AssertionError(f"alias {_alias!r} points to unknown team id {_id!r}")
    _ALIASES[_key(_alias)] = _id


class UnknownTeamError(KeyError):
    pass


def resolve_team(name: str) -> str:
    """Return the canonical team ID for a source name (case/whitespace-insensitive)."""
    try:
        return _ALIASES[_key(name)]
    except KeyError:
        raise UnknownTeamError(
            f"unknown team name {name!r}; add it to tabet/teams.py"
        ) from None


def try_resolve_team(name: str) -> str | None:
    return _ALIASES.get(_key(name))
