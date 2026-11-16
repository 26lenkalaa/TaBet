"use strict";

const PAGE_SIZE = 60;   // a season has about 1,300 games: show the newest first, the rest on request

const status = document.querySelector("#status");
const dashboard = document.querySelector("#dashboard");
const view = document.querySelector("#view");
const team = document.querySelector("#team");
const summary = document.querySelector("#summary");
const count = document.querySelector("#count");
const games = document.querySelector("#games");
const more = document.querySelector("#more");

let shown = PAGE_SIZE;

function addText(parent, tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    parent.append(node);
    return node;
}

function percent(probability) {
    return (probability * 100).toFixed(1) + "%";
}

function marginText(game, margin) {
    if (Math.abs(margin) < 0.05) return "even";
    return (margin > 0 ? game.home_team : game.away_team) + " by " + Math.abs(margin).toFixed(1);
}

function spreadText(game) {
    if (game.spread === 0) return "pick'em";
    const favorite = game.spread < 0 ? game.home_team : game.away_team;
    return favorite + " -" + Math.abs(game.spread).toFixed(1);
}

function restText(days) {
    if (days === null) return "fresh";
    if (days === 1) return "back-to-back";
    return (days - 1) + (days === 2 ? " day off" : " days off");
}

function card(game) {
    const article = document.createElement("article");
    article.className = "card chalk";
    const upcoming = game.home_score === undefined;
    const title = addText(article, "h3", game.away_team + " at " + game.home_team);
    if (game.game_type === "playoffs") addText(title, "span", "PLAYOFFS", "tag");
    if (game.game_type === "play_in") addText(title, "span", "PLAY-IN", "tag");
    addText(article, "p", game.game_date, "dim");
    addText(article, "p", "Model: " + game.home_team + " " + percent(game.p_home) +
        " · " + game.away_team + " " + percent(1 - game.p_home));
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = game.p_home;
    bar.setAttribute("aria-label", game.home_team + " win probability");
    article.append(bar);
    addText(article, "p", "Predicted margin: " + marginText(game, game.margin));
    if (upcoming) {
        addText(article, "p", "Predicted score: " + game.away_team + " " + game.away_pts +
            " · " + game.home_team + " " + game.home_pts);
        addText(article, "p", "Rest: " + game.away_team + " " + restText(game.away_rest_days) +
            " · " + game.home_team + " " + restText(game.home_rest_days), "dim");
    } else {
        addText(article, "p", "Final: " + game.away_team + " " + game.away_score +
            " · " + game.home_team + " " + game.home_score);
        addText(article, "p", game.market_p_home === null ? "Market odds unavailable" :
            "Market: " + game.home_team + " " + percent(game.market_p_home) +
            " · " + game.away_team + " " + percent(1 - game.market_p_home) +
            (game.spread === null ? "" : " · spread " + spreadText(game)), "dim");
    }
    return article;
}

function summaryText(data, stats) {
    if (!stats) {
        return data.has_schedule ? "Scheduled games · pregame forecasts from results through " + data.results_through :
            "No schedule in this snapshot, so there are no upcoming games to show.";
    }
    if (!stats.scored_games) {
        return stats.games + " games · no market odds for this season yet · model Brier " +
            stats.model_brier_all.toFixed(4);
    }
    return stats.scored_games + " of " + stats.games + " games scored · model Brier " +
        stats.model_brier.toFixed(4) + " · market Brier " + stats.market_brier.toFixed(4) +
        (stats.missing_odds ? " · " + stats.missing_odds + " missing odds" : "");
}

function show(data) {
    const report = data.reports.find(item => item.summary.season === view.value);
    const rows = report ? report.games.slice().reverse() : data.upcoming;
    const teams = [...new Set(rows.flatMap(game => [game.home_team, game.away_team]))].sort();
    const selected = team.value;
    team.replaceChildren(new Option("All teams", ""), ...teams.map(code => new Option(code, code)));
    if (teams.includes(selected)) team.value = selected;

    summary.textContent = summaryText(data, report && report.summary);
    const filtered = rows.filter(game => !team.value ||
        game.home_team === team.value || game.away_team === team.value);
    const visible = filtered.slice(0, shown);
    count.textContent = !filtered.length ? "No games in this view." :
        visible.length < filtered.length ? "Newest " + visible.length + " of " + filtered.length + " games shown" :
        filtered.length + " games shown";
    games.replaceChildren(...visible.map(card));
    more.hidden = visible.length >= filtered.length;
    more.textContent = "Show " + Math.min(PAGE_SIZE, filtered.length - visible.length) + " more";
}

async function load() {
    try {
        const response = await fetch("data/basketball.json");
        if (!response.ok) throw new Error("Forecast data unavailable.");
        const data = await response.json();
        view.append(new Option("Upcoming games", "upcoming"));
        for (const report of data.reports) {
            view.append(new Option(report.summary.season + " results", report.summary.season));
        }
        if (!data.upcoming.length && data.reports.length) view.selectedIndex = 1;
        status.textContent = "Snapshot: " + data.as_of + " UTC · results through " + data.results_through + ".";
        dashboard.hidden = false;
        const reset = () => { shown = PAGE_SIZE; show(data); };
        view.addEventListener("change", reset);
        team.addEventListener("change", reset);
        more.addEventListener("click", () => { shown += PAGE_SIZE; show(data); });
        show(data);
    } catch (error) {
        dashboard.hidden = true;
        status.textContent = "Could not load NBA forecasts. Try again later.";
    }
}

load();
