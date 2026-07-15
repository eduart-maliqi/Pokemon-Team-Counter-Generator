"""
Holt die Basis-Daten der 151 Gen-1-Pokémon von der PokéAPI und speichert sie
zusammen mit der Typ-Effektivitaets-Tabelle als JSON im Ordner ../data/.
"""

import json
import time
from pathlib import Path

import requests

POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"
GEN_ONE_POKEMON_COUNT = 151
TARGET_GENERATION_NUMBER = 1  
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
REQUEST_DELAY_SECONDS = 0.2  # sonst kommt vllt sperre

# Reihenfolge der Generationen, um "generation-v" in eine Zahl zu uebersetzen.
GENERATION_NAMES_IN_ORDER = [
    "generation-i",
    "generation-ii",
    "generation-iii",
    "generation-iv",
    "generation-v",
    "generation-vi",
    "generation-vii",
    "generation-viii",
    "generation-ix",
]


def extract_generation_one_types(raw_pokemon: dict) -> list[str]:
    """
    Liefert die Typen, die das Pokémon in Gen 1 hatte.

    Das Feld "types" der PokéAPI enthaelt die HEUTIGEN Typen. Sieben Gen-1-Pokémon
    wurden spaeter umtypisiert (Piepi, Pixi, Pummeluff, Knuddeluff und Pantimos
    wurden zu Fee, Magnetilo und Magneton zu Stahl).

    Fuer diese Faelle gibt es "past_types". Wichtig: Ein Eintrag dort bedeutet
    "BIS EINSCHLIESSLICH dieser Generation galten diese Typen", nicht "in dieser
    Generation". Piepi steht z.B. unter generation-v (war also von Gen 1 bis Gen 5
    Normal), Magnetilo unter generation-i (wurde schon ab Gen 2 zu Stahl).

    Darum: von allen Eintraegen den mit der kleinsten Generation nehmen, die noch
    >= unserer Zielgeneration liegt. Gibt es keinen, gelten die heutigen Typen.
    """
    matching_entries = [
        past_type_entry
        for past_type_entry in raw_pokemon["past_types"]
        if GENERATION_NAMES_IN_ORDER.index(past_type_entry["generation"]["name"]) + 1
        >= TARGET_GENERATION_NUMBER
    ]

    if matching_entries:
        earliest_entry = min(
            matching_entries,
            key=lambda entry: GENERATION_NAMES_IN_ORDER.index(entry["generation"]["name"]),
        )
        return [type_entry["type"]["name"] for type_entry in earliest_entry["types"]]

    return [type_entry["type"]["name"] for type_entry in raw_pokemon["types"]]


def fetch_single_pokemon(pokemon_id: int) -> dict:
    """Lädt ein einzelnes Pokémon und reduziert es auf die Felder, die es braucht."""
    response = requests.get(f"{POKEAPI_BASE_URL}/pokemon/{pokemon_id}", timeout=30)
    response.raise_for_status()
    raw_pokemon = response.json()

    base_stats = {
        stat_entry["stat"]["name"]: stat_entry["base_stat"]
        for stat_entry in raw_pokemon["stats"]
    }

    return {
        "pokemon_id": raw_pokemon["id"],
        "name": raw_pokemon["name"],
        "types": extract_generation_one_types(raw_pokemon),
        "hit_points": base_stats["hp"],
        "attack": base_stats["attack"],
        "defense": base_stats["defense"],
        "special_attack": base_stats["special-attack"],
        "special_defense": base_stats["special-defense"],
        "speed": base_stats["speed"],
        "sprite_url": raw_pokemon["sprites"]["front_default"],
    }


def fetch_all_pokemon() -> list[dict]:
    """Laedt alle 151 Gen-1-Pokémon nacheinander."""
    all_pokemon = []
    for pokemon_id in range(1, GEN_ONE_POKEMON_COUNT + 1):
        pokemon = fetch_single_pokemon(pokemon_id)
        all_pokemon.append(pokemon)
        print(f"[{pokemon_id:3d}/{GEN_ONE_POKEMON_COUNT}] {pokemon['name']}")
        time.sleep(REQUEST_DELAY_SECONDS)
    return all_pokemon


def fetch_type_effectiveness_chart(all_pokemon: list[dict]) -> dict:
    """
    Baut die Typ-Effektivitaets-Tabelle aus der PokéAPI.

    Ergebnis-Form:  {angreifender_typ: {verteidigender_typ: multiplikator}}

    Hinweis: Die PokéAPI liefert die HEUTIGEN Typ-Beziehungen, nicht die von Gen 1.
    Bewusst akzeptiert. Bekannte Abweichungen gegenueber Gen 1:
      - Geist gegen Psycho: heute 2.0, in Gen 1 war es 0.0 (wegen einem coding fehler)
      - Kaefer gegen Gift:   heute 0.5, in Gen 1 war es 2.0
      - Gift gegen Kaefer:   heute 1.0, in Gen 1 war es 2.0
    """
    relevant_types = sorted(
        {type_name for pokemon in all_pokemon for type_name in pokemon["types"]}
    )
    type_effectiveness_chart: dict[str, dict[str, float]] = {}

    for attacking_type in relevant_types:
        response = requests.get(f"{POKEAPI_BASE_URL}/type/{attacking_type}", timeout=30)
        response.raise_for_status()
        damage_relations = response.json()["damage_relations"]

        multipliers = {defending_type: 1.0 for defending_type in relevant_types}
        for entry in damage_relations["double_damage_to"]:
            if entry["name"] in multipliers:
                multipliers[entry["name"]] = 2.0
        for entry in damage_relations["half_damage_to"]:
            if entry["name"] in multipliers:
                multipliers[entry["name"]] = 0.5
        for entry in damage_relations["no_damage_to"]:
            if entry["name"] in multipliers:
                multipliers[entry["name"]] = 0.0

        type_effectiveness_chart[attacking_type] = multipliers
        print(f"Typ-Beziehungen geladen: {attacking_type}")
        time.sleep(REQUEST_DELAY_SECONDS)

    return type_effectiveness_chart


def main() -> None:
    DATA_DIRECTORY.mkdir(exist_ok=True)

    all_pokemon = fetch_all_pokemon()
    type_effectiveness_chart = fetch_type_effectiveness_chart(all_pokemon)

    pokemon_file_path = DATA_DIRECTORY / "pokemon.json"
    type_chart_file_path = DATA_DIRECTORY / "type_effectiveness.json"

    pokemon_file_path.write_text(
        json.dumps(all_pokemon, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    type_chart_file_path.write_text(
        json.dumps(type_effectiveness_chart, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{len(all_pokemon)} Pokémon gespeichert  -> {pokemon_file_path}")
    print(f"{len(type_effectiveness_chart)} Typen gespeichert -> {type_chart_file_path}")


if __name__ == "__main__":
    main()
