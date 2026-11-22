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
  function lum(hex) { var c = rgb(hex); return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255; }
  /* two colours that look alike side by side */
  function alike(a, b) {
    var x = rgb(a), y = rgb(b);
    return Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]) + Math.abs(x[2] - y[2]) < 140;
  }
  /* Bar colours for a match: [home, away, draw]. Each team tries its main colour, then its second colour,
     then a neutral. A colour is skipped if it would vanish into the background (white on a white card,
     black on the dark highlight card) or look like the draw segment or the other team's colour. */
  function barColours(h, a, dark) {
    var draw = dark ? "#6B7280" : "#C3C7CD";
    function visible(c) { return dark ? lum(c) > 0.22 : lum(c) < 0.85; }
    function choose(list, avoid) {
      for (var i = 0; i < list.length; i++) {
        var c = list[i];
        if (visible(c) && avoid.every(function (x) { return !alike(c, x); })) return c;
      }
      return list[list.length - 1];
    }
    var spare = dark ? ["#FFFFFF", "#9AA0A8"] : ["#15181C", "#8A94A6"];
    var home = choose([h.bg, h.ring].concat(spare), [draw]);
    var away = choose([a.bg, a.ring].concat(spare, dark ? ["#FFD23F"] : ["#B45309"]), [draw, home]);
    return [home, away, draw];
  }

  window.TaBetTeams = { team: team, barColours: barColours, all: Object.keys(TEAMS) };
})();
