"""
FastAPI-Server (Phase 7): nimmt ein gegnerisches Team entgegen und liefert das
vom Modell vorgeschlagene Konter-Team zurueck.

Der Server laedt beim Start EINMAL das trainierte Modell und die Daten und haelt
sie im Speicher - er trainiert nichts, sondern nutzt nur das fertige model.pkl
(so in der CLAUDE.md als Trennung von Training und API festgelegt).

Start:  uvicorn main:app --reload    (im Ordner prediction-api ausfuehren)
Docs:   http://127.0.0.1:8000/docs
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Der Ordner ml-training heisst mit Bindestrich und ist so kein gueltiger
# Modulname - darum haengen wir ihn direkt an den Import-Suchpfad.
ML_TRAINING_DIRECTORY = Path(__file__).resolve().parent.parent / "ml-training"
sys.path.insert(0, str(ML_TRAINING_DIRECTORY))

from counter_team import generate_counter_team, load_model_and_data  # noqa: E402

app = FastAPI(
    title="Pokémon Team Counter Generator",
    description="Schlaegt zu einem gegnerischen Gen-1-Team ein Konter-Team vor.",
    version="1.0.0",
)

# Erlaubt dem Frontend (Vite laeuft auf einem anderen Port), die API aufzurufen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Einmalig beim Start laden und im Modul-Zustand halten.
model, pokemon_list, type_effectiveness_chart = load_model_and_data()
pokemon_by_id = {pokemon["pokemon_id"]: pokemon for pokemon in pokemon_list}


class CounterTeamRequest(BaseModel):
    """Erwartet die IDs der 6 gegnerischen Pokémon (1-151)."""
    opponent_pokemon_ids: list[int]


@app.get("/pokemon")
def list_pokemon() -> list[dict]:
    """Alle 151 Pokémon fuer die Auswahl im Frontend (ohne Kampf-Details)."""
    return [
        {
            "pokemon_id": pokemon["pokemon_id"],
            "name": pokemon["name"],
            "types": pokemon["types"],
            "sprite_url": pokemon["sprite_url"],
        }
        for pokemon in pokemon_list
    ]


@app.post("/counter-team")
def counter_team(request: CounterTeamRequest) -> dict:
    """
    Nimmt ein gegnerisches Team (6 Pokémon-IDs) und gibt pro Gegner das beste
    Konter zurueck - 6 verschiedene Pokémon, per Greedy ueber das Modell gewaehlt.
    """
    if not 1 <= len(request.opponent_pokemon_ids) <= 6:
        raise HTTPException(
            status_code=400, detail="Please provide 1 to 6 opposing Pokémon."
        )

    opponent_team = []
    for opponent_id in request.opponent_pokemon_ids:
        if opponent_id not in pokemon_by_id:
            raise HTTPException(
                status_code=400, detail=f"Unknown Pokémon ID: {opponent_id} (valid: 1-151)."
            )
        opponent_team.append(pokemon_by_id[opponent_id])

    counter_team_result = generate_counter_team(
        opponent_team, pokemon_list, model, type_effectiveness_chart
    )
    return {"counter_team": counter_team_result}
