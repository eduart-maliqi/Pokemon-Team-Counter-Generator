# Pokémon Team Counter Generator

A machine learning project that automatically suggests a counter team for a given
Pokémon team. You pick up to six opposing Gen 1 Pokémon, and a trained random
forest model proposes the statistically best counter for each of them.

Voluntary summer project during the first year of a software development
apprenticeship.

## What the project does

1. Collects the data of all 151 Gen 1 Pokémon (stats, types) from the PokéAPI.
2. Simulates battles between all Pokémon, since no real battle data is publicly
   available.
3. Trains a random forest on these battles to learn who beats whom.
4. Serves the model through a FastAPI backend and a Game Boy styled React
   frontend.

## Architecture

```
pokemon-predictor/
├── data-collection/    # fetch data from the PokéAPI
│   └── fetch_pokemon.py
├── ml-training/        # generate training data, train model, counter logic
│   ├── simulate_battles.py
│   ├── build_features.py
│   ├── train_model.py
│   └── counter_team.py
├── prediction-api/     # FastAPI: serves counter team suggestions
│   └── main.py
├── frontend/           # React + Vite: team selection and result view
└── data/               # collected and generated data (JSON, CSV)
```

The ML training runs once/offline; the API only loads the finished model. This
keeps the training code and the API independent of each other.

## Tech stack

- **Data collection & ML:** Python (requests, pandas, scikit-learn)
- **Model:** Random forest (classification), since the data is tabular
- **API:** FastAPI
- **Frontend:** React + Vite
- **Data source:** PokéAPI (https://pokeapi.co), Gen 1 (151 Pokémon)

## Setup and usage

### 1. Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Generate data and model (in this order)

The model and intermediate data are deliberately not committed; they are
regenerated from code and raw data. Thanks to a fixed random seed, everyone gets
the exact same model.

```bash
python data-collection/fetch_pokemon.py    # fetches the Pokémon data (~1 min)
python ml-training/simulate_battles.py      # simulates the battles (~10 sec)
python ml-training/build_features.py        # builds the features
python ml-training/train_model.py           # trains the model (~20 sec)
```

### 3. Start the API

```bash
cd prediction-api
uvicorn main:app --reload
```

Runs on http://127.0.0.1:8000, interactive docs at `/docs`.

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on http://localhost:5173 (the API must run in parallel).

## Key design decisions

- **Greedy instead of a genetic algorithm:** For each opposing Pokémon, the best
  counter is searched among all 151. Chosen deliberately because it is realistic
  within a 1-2 week time budget. Downside: the synergy of the six counters is not
  considered, only the individual one-on-one matchups.
- **Simulated instead of real battle data:** No usable battle data is publicly
  available for training. The model therefore learns the patterns of the
  simulation itself, not "real" battle behaviour.
- **Split by pairing, not by row:** Each pairing appears twice in the data
  (A vs B and B vs A). Both directions go into training or into test together, so
  the accuracy is not inflated by data leakage.

## Result

The model reaches about **94 % accuracy** on held-out test data. The most
important features are the computed differences (above all the type advantage
difference), not the raw stats — evidence that the feature engineering works.

However, a high accuracy here does **not** mean the model understands real
Pokémon battles; it only means it reconstructs the simulation well.

## Known limitations

- **Moves are placeholders:** Each Pokémon has one move per own type with a fixed
  power, not real moves. Snorlax cannot use Earthquake, for example, and status
  moves are missing entirely.
- **Resistances are partly hidden:** The feature `opponent_type_advantage` only
  looks at the opponent's strongest move. A strong resistance against a different
  attack type is masked (for example Charizard's 4x resistance to Grass).
- **The displayed percentage is a model confidence, not a calibrated win chance:**
  It shows how much the trees of the forest agree, not the actual probability of
  winning.

## Possible extensions

- Real moves per Pokémon from the PokéAPI instead of the placeholders.
- A reinforcement learning approach where an agent learns to battle by playing,
  instead of learning from finished result tables.
