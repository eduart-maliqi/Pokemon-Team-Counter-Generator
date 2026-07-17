// Kleine Hilfsschicht fuer die Aufrufe an die FastAPI (Phase 7).
// Die API laeuft standardmaessig auf Port 8000.

const API_BASE_URL = "http://127.0.0.1:8000";

export async function fetchAllPokemon() {
  const response = await fetch(`${API_BASE_URL}/pokemon`);
  if (!response.ok) {
    throw new Error("Could not load the Pokémon list.");
  }
  return response.json();
}

export async function fetchCounterTeam(opponentPokemonIds) {
  const response = await fetch(`${API_BASE_URL}/counter-team`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ opponent_pokemon_ids: opponentPokemonIds }),
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || "The counter team request failed.");
  }
  return response.json();
}
