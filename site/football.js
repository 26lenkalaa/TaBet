"use strict";

const status = document.querySelector("#status");
const dashboard = document.querySelector("#dashboard");
const view = document.querySelector("#view");
const team = document.querySelector("#team");
const game = document.querySelector("#game");
const summary = document.querySelector("#summary");
const card = document.querySelector("#card");

let visibleGames = [];

function element(tag, text = "", className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
}

function percent(probability) {
    return (probability * 100).toFixed(1) + "%";
}

function matchup(row) {
    const separator = row.neutral ? " vs " : " at ";
    return row.away_team + separator + row.home_team;
}

function probabilityCell(label, probability, side, won) {
    const cell = element("div", "", won ? "cell winner" : "cell");
    cell.append(
        element("span", label),
        element("strong", percent(probability), "cell-value " + side + "-value")
    );
    return cell;
}

function renderGame() {
    card.replaceChildren();

    const row = visibleGames[game.selectedIndex];
    if (!row) {
        card.append(element("p", "No games in this view.", "note"));
        return;
    }

    const scheduled = Boolean(row.outcome_probabilities);
    const home = scheduled ? row.outcome_probabilities.home : row.p_home;
    const away = scheduled ? row.outcome_probabilities.away : row.p_away;
    const tied = !scheduled && row.home_score === row.away_score;
    const phase = scheduled ? "Scheduled · snapshot forecast" : tied ? "Final · tie" : "Final · backtest";
    const venue = row.neutral ? " · Neutral site" : "";

    card.append(
        element("h2", matchup(row)),
        element("p", row.game_date + " · " + phase + venue, "meta")
    );

    const board = element("div", "", "scoreboard");
    const score = scheduled ? "PREGAME" : row.away_score + " – " + row.home_score;
    board.append(
        element("span", row.away_team, "side"),
        element("span", score, scheduled ? "score pending" : "score"),
        element("span", row.home_team, "side")
    );

    const probabilities = element("div", "", "probabilities");
    probabilities.append(
        probabilityCell(row.away_team + " win", away, "away",
            !scheduled && row.away_score > row.home_score),
        probabilityCell(row.home_team + " win", home, "home",
            !scheduled && row.home_score > row.away_score)
    );

    card.append(board, element("h3", "TaBet forecast"), probabilities);

    if (scheduled) {
        card.append(element("p", "Results through " + row.history_through, "note"));
        return;
    }

    card.append(element("h3", "Betting market"));
    const market = row.market_p_home;
    const comparison = Number.isFinite(market)
        ? row.away_team + " " + percent(1 - market) + " · " +
          row.home_team + " " + percent(market)
        : "Moneyline unavailable for this game.";
    card.append(element("p", comparison, "market"));

    if (tied) {
        card.append(element("p", "Tie · excluded from backtest scoring.", "note"));
    }
}

function renderView(data) {
    const report = data.reports.find(item => String(item.summary.season) === view.value);
    const rows = report ? report.games.slice().reverse() : data.upcoming;
    const selectedTeam = team.value;
    const selectedGame = game.value;

    const teams = [...new Set(rows.flatMap(row => [row.home_team, row.away_team]))].sort();
    team.replaceChildren(
        new Option("All teams", ""),
        ...teams.map(code => new Option(code, code))
    );
    if (teams.includes(selectedTeam)) team.value = selectedTeam;

    visibleGames = rows.filter(row =>
        !team.value || row.home_team === team.value || row.away_team === team.value
    );
    game.replaceChildren(...visibleGames.map(row =>
        new Option(row.game_date + " · " + matchup(row), row.game_id)
    ));
    game.disabled = visibleGames.length === 0;
    if (visibleGames.some(row => row.game_id === selectedGame)) {
        game.value = selectedGame;
    }

    if (!report) {
        summary.textContent = "Scheduled games in this snapshot · " + visibleGames.length + " available to select";
    } else {
        const stats = report.summary;
        summary.textContent = stats.scored_games
            ? "Season backtest · " + stats.scored_games + " of " + stats.games + " games scored" +
              " · Elo Brier " + stats.elo_brier.toFixed(4) +
              " · Market Brier " + stats.market_brier.toFixed(4) +
              " · " + stats.ties + " ties · " + stats.missing_odds + " missing odds"
            : "No games in this season had both a winner and valid market odds.";
    }
    renderGame();
}

async function load() {
    try {
        const response = await fetch("data/football.json");
        if (!response.ok) throw new Error("Forecast data unavailable.");

        const data = await response.json();
        view.append(new Option("Scheduled games", "upcoming"));
        for (const report of data.reports) {
            const season = String(report.summary.season);
            view.append(new Option(season + " backtest", season));
        }
        if (!data.upcoming.length && data.reports.length) view.selectedIndex = 1;

        view.addEventListener("change", () => renderView(data));
        team.addEventListener("change", () => renderView(data));
        game.addEventListener("change", renderGame);

        renderView(data);
        status.textContent = "Snapshot: " + data.as_of + " UTC · Win probabilities assume no tie.";
        dashboard.hidden = false;
    } catch {
        dashboard.hidden = true;
        status.textContent = "Could not load NFL forecasts. Check that data/football.json is being served.";
    }
}

load();
