import { useEffect, useMemo, useState } from "react";
import { fetchAllPokemon, fetchCounterTeam } from "./api.js";

const MAXIMUM_TEAM_SIZE = 6;

// pokemon-namen aus der API sind klein geschrieben ("mr-mime") - huebsch machen.
function formatName(name) {
  return name
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function App() {
  const [allPokemon, setAllPokemon] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [searchText, setSearchText] = useState("");
  const [counterTeam, setCounterTeam] = useState([]);
  const [loadError, setLoadError] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    fetchAllPokemon()
      .then(setAllPokemon)
      .catch((error) => setLoadError(error.message));
  }, []);

  const selectedPokemon = useMemo(
    () => selectedIds.map((id) => allPokemon.find((p) => p.pokemon_id === id)),
    [selectedIds, allPokemon]
  );

  const filteredPokemon = useMemo(() => {
    const needle = searchText.trim().toLowerCase();
    if (!needle) return allPokemon;
    return allPokemon.filter((p) => p.name.toLowerCase().includes(needle));
  }, [searchText, allPokemon]);

  function togglePokemon(pokemonId) {
    setCounterTeam([]);
    setSelectedIds((current) => {
      if (current.includes(pokemonId)) {
        return current.filter((id) => id !== pokemonId);
      }
      if (current.length >= MAXIMUM_TEAM_SIZE) {
        return current;
      }
      return [...current, pokemonId];
    });
  }

  async function handleGenerate() {
    setIsGenerating(true);
    setLoadError("");
    try {
      const result = await fetchCounterTeam(selectedIds);
      setCounterTeam(result.counter_team);
    } catch (error) {
      setLoadError(error.message);
    } finally {
      setIsGenerating(false);
    }
  }

  const isTeamFull = selectedIds.length >= MAXIMUM_TEAM_SIZE;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Pokémon Team Counter Generator</h1>
        <p className="subtitle">
          Choose up to six opposing Pokémon &ndash; the model suggests your
          counter team.
        </p>
      </header>

      {loadError && <div className="error-box">{loadError}</div>}

      <section className="panel">
        <div className="panel-title">
          Your opponent team &nbsp;({selectedIds.length}/{MAXIMUM_TEAM_SIZE})
        </div>

        <div className="selected-row">
          {Array.from({ length: MAXIMUM_TEAM_SIZE }).map((_, slotIndex) => {
            const pokemon = selectedPokemon[slotIndex];
            return (
              <div key={slotIndex} className="team-slot">
                {pokemon ? (
                  <button
                    className="slot-filled"
                    onClick={() => togglePokemon(pokemon.pokemon_id)}
                    title="Remove"
                  >
                    <img src={pokemon.sprite_url} alt={pokemon.name} />
                    <span>{formatName(pokemon.name)}</span>
                  </button>
                ) : (
                  <div className="slot-empty">?</div>
                )}
              </div>
            );
          })}
        </div>

        <button
          className="generate-button"
          disabled={selectedIds.length === 0 || isGenerating}
          onClick={handleGenerate}
        >
          {isGenerating ? "Calculating ..." : "Generate counter team"}
        </button>
      </section>

      {counterTeam.length > 0 && (
        <section className="panel">
          <div className="panel-title">Your counter team</div>
          <div className="counter-grid">
            {counterTeam.map((entry) => (
              <div key={entry.opponent.pokemon_id} className="counter-card">
                <div className="versus">
                  <span className="versus-label">against</span>
                  <strong>{formatName(entry.opponent.name)}</strong>
                </div>
                <img
                  className="counter-sprite"
                  src={entry.counter.sprite_url}
                  alt={entry.counter.name}
                />
                <div className="counter-name">{formatName(entry.counter.name)}</div>
                <div className="counter-types">
                  {entry.counter.types.map((type) => (
                    <span key={type} className={`type-badge type-${type}`}>
                      {type}
                    </span>
                  ))}
                </div>
                <div className="win-bar-track">
                  <div
                    className="win-bar-fill"
                    style={{ width: `${Math.round(entry.win_probability * 100)}%` }}
                  />
                </div>
                <div className="win-label">
                  {Math.round(entry.win_probability * 100)}% model confidence
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-title">Choose Pokémon</div>
        <input
          className="search-input"
          type="text"
          placeholder="Search ..."
          value={searchText}
          onChange={(event) => setSearchText(event.target.value)}
        />
        <div className="picker-grid">
          {filteredPokemon.map((pokemon) => {
            const isSelected = selectedIds.includes(pokemon.pokemon_id);
            const isDisabled = !isSelected && isTeamFull;
            return (
              <button
                key={pokemon.pokemon_id}
                className={
                  "picker-item" +
                  (isSelected ? " picker-selected" : "") +
                  (isDisabled ? " picker-disabled" : "")
                }
                onClick={() => togglePokemon(pokemon.pokemon_id)}
                disabled={isDisabled}
                title={formatName(pokemon.name)}
              >
                <img src={pokemon.sprite_url} alt={pokemon.name} loading="lazy" />
                <span className="picker-id">#{pokemon.pokemon_id}</span>
                <span className="picker-name">{formatName(pokemon.name)}</span>
              </button>
            );
          })}
        </div>
      </section>

      <footer className="app-footer">
        Random forest model on simulated Gen 1 battles &middot; voluntary summer
        project
      </footer>
    </div>
  );
}
