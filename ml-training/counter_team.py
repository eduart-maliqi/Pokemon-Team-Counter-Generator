"""
Konter-Team-Logik (Phase 6): benutzt das trainierte Modell, um zu einem
gegnerischen Team ein Konter-Team vorzuschlagen.

Kernidee (Greedy): Für jedes gegnerische Pokémon rechnet das Modell die
Gewinn-Wahrscheinlichkeit ALLER 151 möglichen Konter aus und nimmt das beste.
Team-Synergie der Konter wird bewusst nicht optimiert - nur die einzelnen
1-gegen-1-Matchups 

Zusatzregel: Jedes Konter-Pokémon darf nur einmal vorkommen. Ein schon
vergebenes faellt fuer die restlichen Gegner weg, damit das Konter-Team
nicht aus 6 gleichen Pokémon besteht.
Diese Datei liefert Funktionen, die die API in Phase 7 nur noch aufruft -
sie startet selbst keinen Server.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from build_features import FEATURE_COLUMN_NAMES
from simulate_battles import calculate_type_effectiveness

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE_PATH = Path(__file__).resolve().parent / "model.pkl"


def load_model_and_data() -> tuple:
    """Laedt das trainierte Modell, die Pokémon-Liste und die Typ-Tabelle."""
    saved_model = joblib.load(MODEL_FILE_PATH)
    model = saved_model["model"]

    pokemon_list = json.loads((DATA_DIRECTORY / "pokemon.json").read_text(encoding="utf-8"))
    type_effectiveness_chart = json.loads(
        (DATA_DIRECTORY / "type_effectiveness.json").read_text(encoding="utf-8")
    )
    return model, pokemon_list, type_effectiveness_chart


def calculate_best_type_advantage(
    attacking_pokemon: dict, defending_pokemon: dict, type_effectiveness_chart: dict) -> float:
    """Bester Typ-Multiplikator, den das angreifende Pokémon erreichen kann."""
    return max(
        calculate_type_effectiveness(
            move_type, defending_pokemon["types"], type_effectiveness_chart
        )
        for move_type in attacking_pokemon["types"]
    )


def build_feature_row(
    candidate_pokemon: dict, opponent_pokemon: dict, type_effectiveness_chart: dict) -> dict:
    """
    Baut fuer EIN Matchup (Kandidat gegen Gegner) genau die Feature-Spalten, die
    das Modell im Training gesehen hat - in derselben Bedeutung wie build_features.py.
    """
    own_type_advantage = calculate_best_type_advantage(
        candidate_pokemon, opponent_pokemon, type_effectiveness_chart
    )
    opponent_type_advantage = calculate_best_type_advantage(
        opponent_pokemon, candidate_pokemon, type_effectiveness_chart
    )

    return {
        "own_hit_points": candidate_pokemon["hit_points"],
        "own_attack": candidate_pokemon["attack"],
        "own_defense": candidate_pokemon["defense"],
        "own_special_attack": candidate_pokemon["special_attack"],
        "own_special_defense": candidate_pokemon["special_defense"],
        "own_speed": candidate_pokemon["speed"],
        "opponent_hit_points": opponent_pokemon["hit_points"],
        "opponent_attack": opponent_pokemon["attack"],
        "opponent_defense": opponent_pokemon["defense"],
        "opponent_special_attack": opponent_pokemon["special_attack"],
        "opponent_special_defense": opponent_pokemon["special_defense"],
        "opponent_speed": opponent_pokemon["speed"],
        "hit_points_difference": candidate_pokemon["hit_points"] - opponent_pokemon["hit_points"],
        "physical_attack_advantage": candidate_pokemon["attack"] - opponent_pokemon["defense"],
        "special_attack_advantage": candidate_pokemon["special_attack"] - opponent_pokemon["special_defense"],
        "physical_defense_advantage": candidate_pokemon["defense"] - opponent_pokemon["attack"],
        "special_defense_advantage": candidate_pokemon["special_defense"] - opponent_pokemon["special_attack"],
        "speed_difference": candidate_pokemon["speed"] - opponent_pokemon["speed"],
        "is_faster": int(candidate_pokemon["speed"] > opponent_pokemon["speed"]),
        "own_type_advantage": own_type_advantage,
        "opponent_type_advantage": opponent_type_advantage,
        "type_advantage_difference": own_type_advantage - opponent_type_advantage,
    }


def rank_counters_against(
    opponent_pokemon: dict,
    pokemon_list: list[dict],
    model,
    type_effectiveness_chart: dict,
) -> list[tuple[dict, float]]:
    """
    Rechnet fuer EIN gegnerisches Pokémon die Gewinn-Wahrscheinlichkeit aller 151
    Kandidaten aus und gibt sie absteigend sortiert zurueck.

    Alle 151 Feature-Zeilen werden in EINEM predict_proba-Aufruf verarbeitet -
    das ist deutlich schneller, als das Modell 151-mal einzeln zu fragen.
    """
    feature_rows = [
        build_feature_row(candidate_pokemon, opponent_pokemon, type_effectiveness_chart)
        for candidate_pokemon in pokemon_list
    ]
    feature_frame = pd.DataFrame(feature_rows)[FEATURE_COLUMN_NAMES]

    # Spalte 1 = Wahrscheinlichkeit fuer die Klasse "gewinnt" (wins_majority == 1).
    win_probabilities = model.predict_proba(feature_frame)[:, 1]

    ranked = list(zip(pokemon_list, win_probabilities))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def generate_counter_team(
    opponent_team: list[dict],
    pokemon_list: list[dict],
    model,
    type_effectiveness_chart: dict,
) -> list[dict]:
    """
    Baut zum gegnerischen Team ein Konter-Team mit 6 VERSCHIEDENEN Pokémon.

    Greedy pro Gegner: das beste noch freie Konter waehlen. Ein einmal vergebenes
    Konter ist fuer die restlichen Gegner gesperrt, damit kein Pokémon doppelt
    vorkommt.
    """
    already_chosen_ids: set[int] = set()
    counter_team = []

    for opponent_pokemon in opponent_team:
        ranked_counters = rank_counters_against(
            opponent_pokemon, pokemon_list, model, type_effectiveness_chart
        )

        for candidate_pokemon, win_probability in ranked_counters:
            if candidate_pokemon["pokemon_id"] in already_chosen_ids:
                continue

            already_chosen_ids.add(candidate_pokemon["pokemon_id"])
            counter_team.append(
                {
                    "opponent": {
                        "pokemon_id": opponent_pokemon["pokemon_id"],
                        "name": opponent_pokemon["name"],
                    },
                    "counter": {
                        "pokemon_id": candidate_pokemon["pokemon_id"],
                        "name": candidate_pokemon["name"],
                        "types": candidate_pokemon["types"],
                        "sprite_url": candidate_pokemon["sprite_url"],
                    },
                    "win_probability": round(float(win_probability), 3),
                }
            )
            break

    return counter_team


def main() -> None:
    """Kleiner Testlauf ueber die Kommandozeile, damit man die Logik ohne API sieht."""
    model, pokemon_list, type_effectiveness_chart = load_model_and_data()
    pokemon_by_name = {pokemon["name"]: pokemon for pokemon in pokemon_list}

    # Beispiel-Gegnerteam.
    opponent_team_names = ["articuno", "charizard", "venusaur", "alakazam", "starmie", "golem"]
    opponent_team = [pokemon_by_name[name] for name in opponent_team_names]

    counter_team = generate_counter_team(
        opponent_team, pokemon_list, model, type_effectiveness_chart
    )

    print("Gegnerisches Team  ->  bestes Konter (Gewinn-Wahrscheinlichkeit)\n")
    for entry in counter_team:
        print(
            f"  {entry['opponent']['name']:12s} -> "
            f"{entry['counter']['name']:12s} {entry['win_probability']:.1%}  "
            f"{'/'.join(entry['counter']['types'])}"
        )


if __name__ == "__main__":
    main()
