/* Shared helpers for the TaBet site: theme toggle, formatting, data loading. */
(function () {
  var root = document.documentElement;

  /* theme */
  function cur() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  try { var saved = localStorage.getItem("tabet-theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme");
    if (!btn) return;
    function label() { btn.textContent = cur() === "dark" ? "Light mode" : "Dark mode"; }
    label();
    btn.addEventListener("click", function () {
      var n = cur() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", n);
      try { localStorage.setItem("tabet-theme", n); } catch (e) {}
      label();
    });
  });

  var KIND = { backtest: ["Backtest", "t-gray"], logged: ["Logged", "t-green"], preview: ["Preview", "t-purple"], market: ["Market", "t-blue"] };
  var PRIORITY = ["pE", "p0", "V2x", "V1x", "preview", "V0"];

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
    return m.market.find(function (x) { return x.key === "close"; }) || m.market[0] || null;
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
