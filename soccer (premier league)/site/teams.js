/* Team badges: a short code and the club's colours, used for the badges, the colour strip on each
   card and the probability bars. We use colours and codes instead of official crests, which are
   trademarked. Each entry: [code, main colour, text colour on main, second colour, short name]. */
(function () {
  var TEAMS = {
    "Arsenal": ["ARS", "#DB0007", "#FFFFFF", "#FFFFFF", "Arsenal"],
    "Aston Villa": ["AVL", "#670E36", "#FFFFFF", "#95BFE5", "Aston Villa"],
    "Bournemouth": ["BOU", "#DA291C", "#FFFFFF", "#111111", "Bournemouth"],
    "Brentford": ["BRE", "#E30613", "#FFFFFF", "#FFFFFF", "Brentford"],
    "Brighton & Hove Albion": ["BHA", "#0057B8", "#FFFFFF", "#FFFFFF", "Brighton"],
    "Burnley": ["BUR", "#6C1D45", "#FFFFFF", "#99D6EA", "Burnley"],
    "Chelsea": ["CHE", "#034694", "#FFFFFF", "#DBA111", "Chelsea"],
    "Coventry City": ["COV", "#6CABDD", "#0B1F3A", "#FFFFFF", "Coventry"],
    "Crystal Palace": ["CRY", "#1B458F", "#FFFFFF", "#C4122E", "Crystal Palace"],
    "Everton": ["EVE", "#003399", "#FFFFFF", "#FFFFFF", "Everton"],
    "Fulham": ["FUL", "#111111", "#FFFFFF", "#FFFFFF", "Fulham"],
    "Hull City": ["HUL", "#F5A12D", "#111111", "#111111", "Hull"],
    "Ipswich Town": ["IPS", "#0044A9", "#FFFFFF", "#DE2C37", "Ipswich"],
    "Leeds United": ["LEE", "#1D428A", "#FFCD00", "#FFCD00", "Leeds"],
    "Leicester City": ["LEI", "#003090", "#FFFFFF", "#FDBE11", "Leicester"],
    "Liverpool": ["LIV", "#C8102E", "#FFFFFF", "#00B2A9", "Liverpool"],
    "Luton Town": ["LUT", "#F78F1E", "#111111", "#002D62", "Luton"],
    "Manchester City": ["MCI", "#6CABDD", "#0B1F3A", "#1C2C5B", "Man City"],
    "Manchester United": ["MUN", "#DA291C", "#FFFFFF", "#FBE122", "Man Utd"],
    "Newcastle United": ["NEW", "#241F20", "#FFFFFF", "#FFFFFF", "Newcastle"],
    "Nottingham Forest": ["NFO", "#DD0000", "#FFFFFF", "#FFFFFF", "Nott\u2019m Forest"],
    "Sheffield United": ["SHU", "#EE2737", "#FFFFFF", "#111111", "Sheffield Utd"],
    "Southampton": ["SOU", "#D71920", "#FFFFFF", "#111111", "Southampton"],
    "Sunderland": ["SUN", "#EB172B", "#FFFFFF", "#FFFFFF", "Sunderland"],
    "Tottenham Hotspur": ["TOT", "#132257", "#FFFFFF", "#FFFFFF", "Tottenham"],
    "West Ham United": ["WHU", "#7A263A", "#FFFFFF", "#1BB1E7", "West Ham"],
    "Wolverhampton Wanderers": ["WOL", "#FDB913", "#231F20", "#231F20", "Wolves"]
  };

  function team(name) {
    var t = TEAMS[name];
    if (!t) t = [String(name).replace(/[^A-Za-z]/g, "").slice(0, 3).toUpperCase(), "#4B5563", "#FFFFFF", "#FFFFFF", name];
    return { name: name, code: t[0], bg: t[1], fg: t[2], ring: t[3], short: t[4] };
  }

  function rgb(hex) { var n = parseInt(hex.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function close(a, b) {   /* two colours that look alike in a bar next to each other */
    var x = rgb(a), y = rgb(b);
    return Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]) + Math.abs(x[2] - y[2]) < 140;
  }
  /* bar colours for a match: the away side switches to its second colour when the main colours clash */
  function barColours(h, a) {
    var away = a.bg;
    if (close(h.bg, a.bg)) away = a.ring.toUpperCase() === "#FFFFFF" ? "#1A1A1A" : a.ring;
    return [h.bg, away];
  }

  window.TaBetTeams = { team: team, barColours: barColours, all: Object.keys(TEAMS) };
})();
