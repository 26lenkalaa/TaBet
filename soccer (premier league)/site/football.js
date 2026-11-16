"use strict";

const status = document.querySelector("#status");
const dashboard = document.querySelector("#dashboard");
const view = document.querySelector("#view");
const team = document.querySelector("#team");
const summary = document.querySelector("#summary");
const count = document.querySelector("#count");
const games = document.querySelector("#games");

function addText(parent, tag, text) {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.append(node);
}

function percent(probability) {
    return (probability * 100).toFixed(1) + "%";
}

function card(game) {
    const article = document.createElement("article");
    article.className = "card chalk";
    const upcoming = Boolean(game.outcome_probabilities);
    const home = upcoming ? game.outcome_probabilities.home : game.p_home;
    addText(article, "h3", game.away_team + " at " + game.home_team);
    addText(article, "p", game.game_date + (game.neutral ? " · Neutral site" : ""));
    addText(article, "p", "Elo: " + game.home_team + " " + percent(home) +
        " · " + game.away_team + " " + percent(1 - home));
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = home;
    bar.setAttribute("aria-label", game.home_team + " win probability");
    article.append(bar);
    if (upcoming) {
        addText(article, "p", "Results through " + game.history_through);
    } else {
        addText(article, "p", "Final: " + game.away_team + " " + game.away_score +
            " · " + game.home_team + " " + game.home_score);
        const market = game.market_p_home;
        addText(article, "p", market === null ? "Moneyline unavailable" :
            "Market: " + game.home_team + " " + percent(market) +
            " · " + game.away_team + " " + percent(1 - market));
        if (game.status === "tie") addText(article, "p", "Tie · excluded from scoring");
    }
    return article;
}

function show(data) {
    const report = data.reports.find(item => String(item.summary.season) === view.value);
    const rows = report ? report.games.slice().reverse() : data.upcoming;
    const stats = report && report.summary;
    const teams = [...new Set(rows.flatMap(game => [game.home_team, game.away_team]))].sort();
    const selected = team.value;
    team.replaceChildren(new Option("All teams", ""), ...teams.map(code => new Option(code, code)));
    if (teams.includes(selected)) team.value = selected;

    summary.textContent = !stats ? "Scheduled fixtures · pregame Elo" :
        stats.scored_games ? stats.scored_games + " of " + stats.games + " games scored · Elo Brier " +
        stats.elo_brier.toFixed(4) + " · Market Brier " + stats.market_brier.toFixed(4) +
        " · " + stats.ties + " ties · " + stats.missing_odds + " missing odds" :
        "No games with both a winner and valid market odds.";
    const filtered = rows.filter(game => !team.value ||
        game.home_team === team.value || game.away_team === team.value);
    count.textContent = filtered.length ? filtered.length + " games shown" : "No games in this view.";
    games.replaceChildren(...filtered.map(card));
}

async function load() {
    try {
        const response = await fetch("data/football.json");
        if (!response.ok) throw new Error("Forecast data unavailable.");
        const data = await response.json();
        view.append(new Option("Scheduled games", "upcoming"));
        for (const report of data.reports) {
            const season = report.summary.season;
            view.append(new Option(season + " backtest", season));
        }
        if (!data.upcoming.length && data.reports.length) view.selectedIndex = 1;
        status.textContent = "Snapshot: " + data.as_of + " UTC · Win probabilities assume no tie.";
        dashboard.hidden = false;
        view.addEventListener("change", () => show(data));
        team.addEventListener("change", () => show(data));
        show(data);
    } catch (error) {
        dashboard.hidden = true;
        status.textContent = "Could not load NFL forecasts. Try again later.";
    }
}

load();
