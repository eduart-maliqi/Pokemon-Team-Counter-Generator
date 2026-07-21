"""
Feature Engineering: uebersetzt jede Paarung aus battle_results.csv in Zahlen,
aus denen der Random Forest lernen kann. Ergebnis: data/features.csv

Ein Entscheidungsbaum prueft immer nur EINE Spalte gegen einen Schwellwert. Er
kann von sich aus nicht "mein Angriff minus gegnerische Verteidigung" rechnen.
Genau darum legen wir ihm solche Differenzen als fertige Spalte hin - das ist
der Kern von Feature Engineering.

Aufruf:  python ml-training/build_features.py
"""

from pathlib import Path

import pandas as pd

from simulate_battles import calculate_type_effectiveness, load_data

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"

# Genau diese Spalten sieht das Modell spaeter beim Training.
# Namen und IDs stehen zwar in der CSV, gehoeren aber NICHT dazu - sonst wuerde
# das Modell einzelne Pokémon auswendig lernen statt Muster zu erkennen.
FEATURE_COLUMN_NAMES = [
    "own_hit_points",
    "own_attack",
    "own_defense",
    "own_special_attack",
    "own_special_defense",
    "own_speed",
    "opponent_hit_points",
    "opponent_attack",
    "opponent_defense",
    "opponent_special_attack",
    "opponent_special_defense",
    "opponent_speed",
    "hit_points_difference",
    "physical_attack_advantage",
    "special_attack_advantage",
    "physical_defense_advantage",
    "special_defense_advantage",
    "speed_difference",
    "is_faster",
    "own_type_advantage",
    "opponent_type_advantage",
    "type_advantage_difference",
]
TARGET_COLUMN_NAME = "wins_majority"


def calculate_best_type_advantage(
    attacking_pokemon: dict, defending_pokemon: dict, type_effectiveness_chart: dict
) -> float:
    """
    Der beste Typ-Multiplikator, den das angreifende Pokémon erreichen kann.

    Zaehlt jetzt ueber die Typen der ECHTEN Attacken, nicht mehr nur ueber die
    eigenen Typen. So zaehlt auch Coverage: Relaxo (Normal) mit Erdbeben bekommt
    gegen Gestein/Boden den Boden-Vorteil, den es mit reinen Normal-Attacken nie
    haette.
    """
    return max(
        calculate_type_effectiveness(
            move["type"], defending_pokemon["types"], type_effectiveness_chart
        )
        for move in attacking_pokemon["moves"]
    )


def build_features(
    battle_results: pd.DataFrame, pokemon_list: list[dict], type_effectiveness_chart: dict
) -> pd.DataFrame:
    """Macht aus jeder Kampf-Paarung eine Zeile mit Features und Zielvariable."""
    # Nachschlagetabelle, damit wir pro Zeile nicht die ganze Liste durchsuchen muessen.
    pokemon_by_id = {pokemon["pokemon_id"]: pokemon for pokemon in pokemon_list}

    feature_rows = []

    for battle_result in battle_results.itertuples(index=False):
        own_pokemon = pokemon_by_id[battle_result.pokemon_id]
        opponent_pokemon = pokemon_by_id[battle_result.opponent_id]

        own_type_advantage = calculate_best_type_advantage(
            own_pokemon, opponent_pokemon, type_effectiveness_chart
        )
        opponent_type_advantage = calculate_best_type_advantage(
            opponent_pokemon, own_pokemon, type_effectiveness_chart
        )

        feature_rows.append(
            {
                # Nur zur Nachvollziehbarkeit in der CSV, fliesst nicht ins Training ein.
                "pokemon_id": battle_result.pokemon_id,
                "pokemon_name": battle_result.pokemon_name,
                "opponent_id": battle_result.opponent_id,
                "opponent_name": battle_result.opponent_name,

                # Rohwerte beider Seiten.
                "own_hit_points": own_pokemon["hit_points"],
                "own_attack": own_pokemon["attack"],
                "own_defense": own_pokemon["defense"],
                "own_special_attack": own_pokemon["special_attack"],
                "own_special_defense": own_pokemon["special_defense"],
                "own_speed": own_pokemon["speed"],
                "opponent_hit_points": opponent_pokemon["hit_points"],
                "opponent_attack": opponent_pokemon["attack"],
                "opponent_defense": opponent_pokemon["defense"],
                "opponent_special_attack": opponent_pokemon["special_attack"],
                "opponent_special_defense": opponent_pokemon["special_defense"],
                "opponent_speed": opponent_pokemon["speed"],

                # Differenzen: das eigentlich Aussagekraeftige. Positiv = Vorteil fuer uns.
                "hit_points_difference": own_pokemon["hit_points"] - opponent_pokemon["hit_points"],
                "physical_attack_advantage": own_pokemon["attack"] - opponent_pokemon["defense"],
                "special_attack_advantage": own_pokemon["special_attack"] - opponent_pokemon["special_defense"],
                "physical_defense_advantage": own_pokemon["defense"] - opponent_pokemon["attack"],
                "special_defense_advantage": own_pokemon["special_defense"] - opponent_pokemon["special_attack"],
                "speed_difference": own_pokemon["speed"] - opponent_pokemon["speed"],

                # Wer zuerst angreift, entscheidet in der Simulation oft den Kampf.
                "is_faster": int(own_pokemon["speed"] > opponent_pokemon["speed"]),

                # Typ-Vorteil in beide Richtungen, plus die Differenz als eigene Spalte.
                "own_type_advantage": own_type_advantage,
                "opponent_type_advantage": opponent_type_advantage,
                "type_advantage_difference": own_type_advantage - opponent_type_advantage,

                # Zielvariable: hat es die Mehrheit der 20 Kaempfe gewonnen?
                TARGET_COLUMN_NAME: battle_result.wins_majority,

                # Nur zur Analyse mitgefuehrt, wird ebenfalls nicht trainiert.
                "win_probability": battle_result.win_probability,
            }
        )

    return pd.DataFrame(feature_rows)


def main() -> None:
    pokemon_list, type_effectiveness_chart = load_data()
    battle_results = pd.read_csv(DATA_DIRECTORY / "battle_results.csv")

    print(f"Baue Features fuer {len(battle_results):,} Paarungen ...")

    features_data_frame = build_features(battle_results, pokemon_list, type_effectiveness_chart)

    output_file_path = DATA_DIRECTORY / "features.csv"
    features_data_frame.to_csv(output_file_path, index=False)

    won_share = features_data_frame[TARGET_COLUMN_NAME].mean()

    print(f"\n{len(features_data_frame):,} Zeilen mit {len(FEATURE_COLUMN_NAMES)} Features "
          f"gespeichert -> {output_file_path}")
    print(f"Verteilung der Zielvariable: {won_share:.1%} gewonnen, {1 - won_share:.1%} verloren")


if __name__ == "__main__":
    main()
