"""
Erzeugt die Trainingsdaten: laesst jedes Pokémon gegen jedes andere kaempfen und
schreibt die Gewinn-Wahrscheinlichkeit pro Paarung nach data/battle_results.csv.

Da es keine echten Kampfdaten gibt, sind das simulierte Kaempfe. Wichtig zu wissen:
Das Modell kann spaeter nur die Muster lernen, die in DIESER Simulation stecken.

Aufruf:  python ml-training/simulate_battles.py
"""

import json
import random
from itertools import permutations
from pathlib import Path

import pandas as pd

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
BATTLES_PER_PAIRING = 20
POKEMON_LEVEL = 50
STAB = 1.5
CRIT_CHANCE = 1 / 16  
CRIT_MULTIPLIER = 2.0
MAX_TURNS = 300  # Notbremse gegen Paarungen, die sich gegenseitig kaum schaden
RANDOM_SEED = 42  # feste Zahl -> jeder Lauf erzeugt dieselben Daten (reproduzierbar)

# In Gen 1 haengt es NICHT von der Attacke ab, ob sie physisch oder speziell ist,
# sondern allein vom Typ der Attacke. Genau das bilden wir hier nach.
PHYSICAL_TYPES = {
    "normal", "fighting", "flying", "ground", "rock", "bug", "ghost", "poison",
}
SPECIAL_TYPES = {
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
}


def load_data() -> tuple[list[dict], dict]:
    """Laedt die in Phase 2 gesammelten Pokémon- und Typ-Daten."""
    pokemon_list = json.loads((DATA_DIRECTORY / "pokemon.json").read_text(encoding="utf-8"))
    type_effectiveness_chart = json.loads(
        (DATA_DIRECTORY / "type_effectiveness.json").read_text(encoding="utf-8")
    )
    return pokemon_list, type_effectiveness_chart


def calculate_hit_points_at_level(base_hit_points: int) -> int:
    """
    KP-Formel der Spiele, ohne DVs/EVs.
    Auf Level 50 laeuft das auf "Basis-KP + 60" hinaus.
    """
    return (2 * base_hit_points * POKEMON_LEVEL) // 100 + POKEMON_LEVEL + 10


def calculate_stat_at_level(base_stat: int) -> int:
    """
    Formel fuer alle uebrigen Werte (Angriff, Verteidigung, ...), ohne DVs/EVs.
    Auf Level 50 laeuft das auf "Basiswert + 5" hinaus.
    """
    return (2 * base_stat * POKEMON_LEVEL) // 100 + 5


def calculate_type_effectiveness(
    attacking_type: str, defending_types: list[str], type_effectiveness_chart: dict
) -> float:
    """
    Multiplikator einer Attacke gegen ein (evtl. zweifach getyptes) Pokémon.
    Bei zwei Typen werden die Multiplikatoren multipliziert - aus 2.0 und 2.0 wird 4.0.
    """
    total_multiplier = 1.0
    for defending_type in defending_types:
        total_multiplier *= type_effectiveness_chart[attacking_type][defending_type]
    return total_multiplier


def calculate_base_damage(
    attacking_pokemon: dict,
    defending_pokemon: dict,
    move: dict,
    type_effectiveness_chart: dict,
) -> float:
    """
    Schadensformel der Spiele, OHNE Zufall. move ist eine echte Attacke mit
    {"name", "type", "power"} - die Staerke kommt jetzt aus der Attacke selbst,
    nicht mehr aus einer festen Konstante.
    """
    move_type = move["type"]
    type_multiplier = calculate_type_effectiveness(
        move_type, defending_pokemon["types"], type_effectiveness_chart
    )

    if type_multiplier == 0.0:
        return 0.0

    # In Gen 1 entscheidet allein der Typ der Attacke, ob sie physisch oder
    # speziell ist - und damit, welcher Angriffswert zaehlt.
    if move_type in PHYSICAL_TYPES:
        attack_value = calculate_stat_at_level(attacking_pokemon["attack"])
        defense_value = calculate_stat_at_level(defending_pokemon["defense"])
    else:
        attack_value = calculate_stat_at_level(attacking_pokemon["special_attack"])
        defense_value = calculate_stat_at_level(defending_pokemon["special_defense"])

    base_damage = (
        (2 * POKEMON_LEVEL / 5 + 2) * move["power"] * attack_value / defense_value / 50
    ) + 2

    if move_type in attacking_pokemon["types"]:
        base_damage *= STAB

    return base_damage * type_multiplier


def calculate_damage(
    attacking_pokemon: dict,
    defending_pokemon: dict,
    move: dict,
    type_effectiveness_chart: dict,
    random_generator: random.Random,
) -> int:
    """
    Der Schaden eines einzelnen Treffers, inklusive Zufall:
      - Kritischer Treffer: feste Chance von 1/16, verdoppelt den Schaden
      - Schadensstreuung: jeder Treffer macht 85-100 % des Grundschadens

    Die feste Krit-Chance ist bewusst die der neueren Spiele. In Gen 1 haette sie
    am Speed des Angreifers gehangen, was schnelle Pokémon zusaetzlich bevorteilt
    haette - Speed ist ohnehin schon ein Feature des Modells.
    """
    base_damage = calculate_base_damage(
        attacking_pokemon, defending_pokemon, move, type_effectiveness_chart
    )

    if base_damage == 0.0:
        return 0

    if random_generator.random() < CRIT_CHANCE:
        base_damage *= CRIT_MULTIPLIER

    base_damage *= random_generator.uniform(0.85, 1.0)

    # Ein Treffer, der ueberhaupt wirkt, macht immer mindestens 1 Schaden.
    return max(1, int(base_damage))


def choose_best_move(
    attacking_pokemon: dict, defending_pokemon: dict, type_effectiveness_chart: dict
) -> dict:
    """
    Waehlt aus den ECHTEN Attacken des Pokémon die, die gegen diesen Gegner am
    meisten Schaden macht - genau das, was ein Spieler auch tun wuerde.

    Bewusst nach tatsaechlichem SCHADEN und nicht nur nach Typ-Effektivitaet oder
    Staerke: Beruecksichtigt Attacken-Staerke, STAB, Typ-Vorteil und ob die
    Attacke ueber den physischen oder speziellen Angriff des Pokémon laeuft.
    So nimmt Relaxo gegen Gestein/Boden z.B. sein Erdbeben, gegen andere seinen
    staerksten Normal-Treffer.
    """
    return max(
        attacking_pokemon["moves"],
        key=lambda move: calculate_base_damage(
            attacking_pokemon, defending_pokemon, move, type_effectiveness_chart
        ),
    )


def simulate_single_battle(
    first_pokemon: dict,
    second_pokemon: dict,
    type_effectiveness_chart: dict,
    random_generator: random.Random,
) -> bool:
    """
    Simuliert EINEN Kampf. Gibt True zurueck, wenn first_pokemon gewinnt.

    Ablauf: Der Schnellere greift zuerst an, dann abwechselnd, bis einer bei 0 KP ist.
    """
    first_maximum_hp = calculate_hit_points_at_level(first_pokemon["hit_points"])
    second_maximum_hp = calculate_hit_points_at_level(second_pokemon["hit_points"])

    first_remaining_hp = first_maximum_hp
    second_remaining_hp = second_maximum_hp

    first_move = choose_best_move(first_pokemon, second_pokemon, type_effectiveness_chart)
    second_move = choose_best_move(second_pokemon, first_pokemon, type_effectiveness_chart)

    # Bei gleichem Speed entscheidet der Zufall, wer beginnt.
    if first_pokemon["speed"] > second_pokemon["speed"]:
        first_pokemon_attacks_first = True
    elif first_pokemon["speed"] < second_pokemon["speed"]:
        first_pokemon_attacks_first = False
    else:
        first_pokemon_attacks_first = random_generator.random() < 0.5

    for _ in range(MAX_TURNS):
        if first_pokemon_attacks_first:
            second_remaining_hp -= calculate_damage(
                first_pokemon, second_pokemon, first_move,
                type_effectiveness_chart, random_generator,
            )
            if second_remaining_hp <= 0:
                return True

            first_remaining_hp -= calculate_damage(
                second_pokemon, first_pokemon, second_move,
                type_effectiveness_chart, random_generator,
            )
            if first_remaining_hp <= 0:
                return False
        else:
            first_remaining_hp -= calculate_damage(
                second_pokemon, first_pokemon, second_move,
                type_effectiveness_chart, random_generator,
            )
            if first_remaining_hp <= 0:
                return False

            second_remaining_hp -= calculate_damage(
                first_pokemon, second_pokemon, first_move,
                type_effectiveness_chart, random_generator,
            )
            if second_remaining_hp <= 0:
                return True

    # Nach MAXIMUM_TURNS immer noch kein KO (beide sind gegen den anderen immun):
    # Es gewinnt, wer prozentual mehr KP uebrig hat.
    first_hit_points_fraction = first_remaining_hp / first_maximum_hp
    second_hit_points_fraction = second_remaining_hp / second_maximum_hp
    return first_hit_points_fraction >= second_hit_points_fraction


def simulate_all_battles(pokemon_list: list[dict], type_effectiveness_chart: dict) -> pd.DataFrame:
    """
    Laesst jedes Pokémon gegen jedes andere antreten, jeweils BATTLES_PER_PAIRING mal.
    Ergebnis: eine Zeile pro geordneter Paarung mit der Gewinn-Wahrscheinlichkeit.
    """
    random_generator = random.Random(RANDOM_SEED)
    battle_results = []

   
    all_pairings = list(permutations(pokemon_list, 2))

    for pairing_index, (first_pokemon, second_pokemon) in enumerate(all_pairings, start=1):
        wins_of_first_pokemon = 0
        for _ in range(BATTLES_PER_PAIRING):
            if simulate_single_battle(
                first_pokemon, second_pokemon, type_effectiveness_chart, random_generator
            ):
                wins_of_first_pokemon += 1

        win_probability = wins_of_first_pokemon / BATTLES_PER_PAIRING

        battle_results.append(
            {
                "pokemon_id": first_pokemon["pokemon_id"],
                "pokemon_name": first_pokemon["name"],
                "opponent_id": second_pokemon["pokemon_id"],
                "opponent_name": second_pokemon["name"],
                "wins": wins_of_first_pokemon,
                "battles": BATTLES_PER_PAIRING,
                "win_probability": win_probability,
                # Zielvariable fuer das Modell: gewinnt es die Mehrheit der Kaempfe?
                "wins_majority": int(win_probability > 0.5),
            }
        )

        if pairing_index % 2000 == 0:
            print(f"[{pairing_index}/{len(all_pairings)}] Paarungen simuliert")

    return pd.DataFrame(battle_results)


def main() -> None:
    pokemon_list, type_effectiveness_chart = load_data()

    total_battles = len(pokemon_list) * (len(pokemon_list) - 1) * BATTLES_PER_PAIRING
    print(f"Simuliere {total_battles:,} Kaempfe ...\n")

    battle_results_data_frame = simulate_all_battles(pokemon_list, type_effectiveness_chart)

    output_file_path = DATA_DIRECTORY / "battle_results.csv"
    battle_results_data_frame.to_csv(output_file_path, index=False)

    print(f"\n{len(battle_results_data_frame):,} Paarungen gespeichert -> {output_file_path}")
    print(f"Anteil mehrheitlich gewonnener Paarungen: {battle_results_data_frame['wins_majority'].mean():.1%}")


if __name__ == "__main__":
    main()
