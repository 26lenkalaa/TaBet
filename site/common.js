/* Shared helpers for the TaBet pitch and match explorer: formatting and data loading. */
(function () {
  var KIND = { backtest: ["Backtest", "t-gray"], logged: ["Logged", "t-green"], preview: ["Preview", "t-purple"], market: ["Market", "t-blue"] };
  var PRIORITY = ["pE", "p0", "V2x", "V1x", "preview_lineups", "preview", "V0"];

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
  var cache = null;
  function load() {
    if (!cache) cache = fetch("data/matches.json").then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    return cache;
  }

  window.TaBet = { KIND: KIND, esc: esc, pct: pct, when: when, primary: primary, market: market, bar: bar, load: load };
})();
