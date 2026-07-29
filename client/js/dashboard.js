const gameList = document.getElementById("gameList");

function renderGames(games) {
  gameList.replaceChildren();

  if (!games.length) {
    gameList.innerHTML = '<p class="status">No games are available.</p>';
    return;
  }

  for (const game of games) {
    const card = document.createElement(game.available === false ? "article" : "a");
    card.className = "game-card";
    if (game.available === false) {
      card.classList.add("unavailable");
      card.setAttribute("aria-disabled", "true");
    } else {
      card.href = `/games/${encodeURIComponent(game.id)}`;
    }

    const title = document.createElement("h2");
    title.textContent = game.title;

    const description = document.createElement("p");
    description.textContent = game.description;

    const capabilities = document.createElement("span");
    capabilities.className = "capabilities";
    capabilities.textContent =
      game.available === false
        ? "Unavailable"
        : (game.capabilities || []).join(" + ");

    card.append(title, description, capabilities);
    gameList.append(card);
  }
}

async function loadGames() {
  try {
    const response = await fetch("/api/games");
    if (!response.ok) throw new Error(`Game list returned ${response.status}`);
    renderGames(await response.json());
  } catch (error) {
    console.error("Could not load games", error);
    gameList.innerHTML =
      '<p class="status error">Could not load the game list. Please refresh.</p>';
  }
}

loadGames();
