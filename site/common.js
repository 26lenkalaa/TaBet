/* Shared helpers for the TaBet pitch and match explorer: formatting and data loading. */
(function () {
  var KIND = { backtest: ["Backtest", "t-gray"], logged: ["Logged", "t-green"], saved: ["Pre-match", "t-yellow"], preview: ["Preview", "t-purple"], market: ["Market", "t-blue"] };
  var PRIORITY = ["pE", "p0", "saved_lineups", "saved", "V2x", "V1x", "preview_lineups", "preview", "V0"];

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function when(iso, mode) {
    var d = new Date(iso), o;
    if (mode === "long") o = { weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "Europe/London" };
    else if (mode === "card") o = { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Europe/London" };
    else o = { day: "numeric", month: "short", timeZone: "Europe/London" };
    return d.toLocaleString("en-GB", o);
  }
  /* The forecast to headline: the logged one if there is one, else the best backtest, else the preview. */
  function primary(m) {
    for (var i = 0; i < PRIORITY.length; i++) {
      var f = m.forecasts.find(function (x) { return x.key === PRIORITY[i]; });
      if (f) return f;
    }
    return m.forecasts[0] || null;
  }
  function market(m) {
    // the market at the same moment as our logged forecast is the fair comparison; then closing odds
    var order = ["matched", "close", "latest"];
    for (var i = 0; i < order.length; i++) {
      var x = m.market.find(function (y) { return y.key === order[i]; });
      if (x) return x;
    }
    return m.market[0] || null;
  }
  function bar(p) {
    return '<div class="pbar" aria-hidden="true"><span class="h" style="width:' + (p[0] * 100) + '%"></span><span class="d" style="width:' +
      (p[1] * 100) + '%"></span><span class="a" style="width:' + (p[2] * 100) + '%"></span></div>';
  }
  /* Data comes from the TaBet API (/api, backed by Supabase). If the API is slow or down, the page falls
     back to the static files in data/, which the logger still builds: data/index.json lists one file
     per season. */
  var API = "/api/competitions/premier-league";
  function getJSON(u, ms) {
    var ctl = ms && window.AbortController ? new AbortController() : null;
    var t = ctl ? setTimeout(function () { ctl.abort(); }, ms) : null;
    function done() { if (t) clearTimeout(t); }
    return fetch(u, ctl ? { signal: ctl.signal } : undefined)
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) { done(); return d; }, function (e) { done(); throw e; });
  }
  /* an API match in the shape the pages already use (the site's own id stays the page id) */
  function clean(f) {
    var o = { key: f.key, label: f.label, p: f.p };
    if (f.kind) o.kind = f.kind;
    if (f.note) o.note = f.note;
    return o;
  }
  function fromApi(m) {
    var o = { id: m.source_id, api_id: m.id, season: m.season, kickoff: m.kickoff, home: m.home, away: m.away,
              status: m.status, forecasts: m.forecasts.map(clean), market: m.market.map(clean) };
    if (m.gameweek != null) o.gameweek = m.gameweek;
    if (m.score != null) { o.score = m.score; o.result = m.result; }
    if (m.note) o.missing = m.note;
    return o;
  }

  var files = null;
  function fileIndex() {
    if (!files) files = getJSON("data/index.json");
    return files;
  }
  function fileSeason(s) {
    return fileIndex().then(function (ix) { return getJSON("data/" + ix.files[s]); });
  }
  function fileLineups(s) {
    return fileIndex().then(function (ix) {
      var f = ix.lineup_files && ix.lineup_files[s] ? ix.lineup_files[s] : "lineups-" + s + ".json";
      return getJSON("data/" + f).then(function (d) { return d.lineups || {}; });
    });
  }

  var idx = null, bySeason = {}, luBySeason = {};
  /* The API is used only when it is as fresh as the static files: if its last sync is older than the
     files' build (a refused or failed sync), the page reads the files instead. */
  function index() {
    if (!idx) idx = Promise.all([
      getJSON(API + "/seasons", 4000),
      fileIndex().catch(function () { return null; })
    ]).then(function (r) {
      var d = r[0], fx = r[1];
      var seasons = d.data.map(function (x) { return x.season; }).sort();
      if (!seasons.length) throw new Error("no seasons");
      var apiBuilt = d.meta && d.meta.built_at;
      if (fx && fx.built_utc && (!apiBuilt || Date.parse(apiBuilt) < Date.parse(fx.built_utc))) throw new Error("API is stale");
      return { built_utc: apiBuilt, seasons: seasons, source: "api" };
    }).catch(function () {
      return fileIndex().then(function (ix) { ix.source = "files"; return ix; });
    });
    return idx;
  }
  function season(s) {
    if (!bySeason[s]) bySeason[s] = index().then(function (ix) {
      if (ix.source !== "api") return fileSeason(s);
      return getJSON(API + "/seasons/" + s + "/matches", 6000)
        .then(function (d) { return { season: s, matches: d.data.map(fromApi) }; })
        .catch(function () { return fileSeason(s); });
    });
    return bySeason[s];
  }
  /* starting XIs for one season ({match id: {actual, pred, note, inferred}}), only for the match page */
  function lineups(s) {
    if (!luBySeason[s]) luBySeason[s] = index().then(function (ix) {
      if (ix.source !== "api") return fileLineups(s);
      return getJSON(API + "/seasons/" + s + "/lineups", 6000).then(function (d) {
        var out = {};
        d.data.forEach(function (r) { out[r.source_id] = r.lineup; });
        return out;
      }).catch(function () { return fileLineups(s); });
    }).catch(function () { return {}; });
    return luBySeason[s];
  }
  /* load(season?): {built_utc, n_log_records, seasons, matches} for one season (default: the latest) */
  function load(s) {
    return index().then(function (ix) {
      var want = s && ix.seasons.indexOf(s) >= 0 ? s : ix.seasons[ix.seasons.length - 1];
      return season(want).then(function (sd) {
        return { built_utc: ix.built_utc, n_log_records: ix.n_log_records, seasons: ix.seasons, season: want, matches: sd.matches, source: ix.source };
      });
    });
  }

  /* ---- live scores (current gameweek, from Fantasy Premier League via /api/live; display only) ----
     watchLive(cb) calls cb(L) now and after every refresh: every 30 s while a match is on or about to
     start, every 5 minutes otherwise, and not at all while the tab is hidden. L.find(m) returns the live
     entry for a site match ({state: pre|live|ft|final, minute, score: [h, a], events}), or null. */
  function liveFind(L, m) {
    var TM = window.TaBetTeams;
    if (!L || !TM) return null;
    return L.byPair[TM.team(m.home).code + "-" + TM.team(m.away).code] || null;
  }
  function liveBusy(d) {
    var now = Date.now();
    return d.matches.some(function (x) {
      var t = Date.parse(x.kickoff);
      return x.state === "live" || x.state === "ft" || (x.state === "pre" && t - now < 15 * 60000 && now - t < 3 * 3600000);
    });
  }
  function watchLive(cb) {
    var timer = null;
    function again(ms) { clearTimeout(timer); timer = setTimeout(go, ms); }
    function go() {
      if (document.hidden) return;                      /* resumes on visibilitychange */
      getJSON("/api/live/premier-league", 8000).then(function (d) {
        var L = { gameweek: d.gameweek, updated_at: d.updated_at, byPair: {} };
        d.matches.forEach(function (x) { if (x.home && x.away) L.byPair[x.home + "-" + x.away] = x; });
        L.find = function (m) { return liveFind(L, m); };
        try { cb(L); } catch (e) {}
        again(liveBusy(d) ? 30000 : 300000);
      }, function () { again(120000); });
    }
    document.addEventListener("visibilitychange", function () { if (!document.hidden) go(); });
    go();
  }
  /* "67′", "FT", or "" */
  function liveClock(x) {
    if (!x) return "";
    if (x.state === "live") return x.minute != null ? x.minute + "′" : "LIVE";
    if (x.state === "ft" || x.state === "final") return "FT";
    return "";
  }
  /* scorers for one side: "Mbeumo 2, Wissa, Tarkowski (OG)" */
  function liveScorers(x, side) {
    return (x ? x.events : []).filter(function (e) { return e.side === side && (e.type === "goal" || e.type === "own_goal"); })
      .map(function (e) { return e.player + (e.n > 1 ? " " + e.n : "") + (e.type === "own_goal" ? " (OG)" : ""); }).join(", ");
  }

  window.TaBet = { KIND: KIND, esc: esc, pct: pct, when: when, primary: primary, market: market, bar: bar, load: load, index: index, lineups: lineups,
                   watchLive: watchLive, liveClock: liveClock, liveScorers: liveScorers,
                   source: function () { return index().then(function (ix) { return ix.source; }); } };
})();
