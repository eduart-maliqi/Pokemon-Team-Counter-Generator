"""
Trainiert den Random Forest auf data/features.csv und speichert ihn als model.pkl.

Aufruf:  python ml-training/train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from build_features import FEATURE_COLUMN_NAMES, TARGET_COLUMN_NAME

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE_PATH = Path(__file__).resolve().parent / "model.pkl"

TEST_SET_SHARE = 0.2  # 20 % der Paarungen werden zurueckgehalten
NUMBER_OF_TREES = 100
RANDOM_SEED = 42


def build_pairing_groups(features_data_frame: pd.DataFrame) -> pd.Series:
    """
    Gibt jeder Paarung eine gemeinsame Gruppen-ID, unabhaengig von der Reihenfolge.

    Hintergrund: Jede Paarung steht zweimal in den Daten - "Pikachu gegen Garados"
    und "Garados gegen Pikachu". Das sind gespiegelte Varianten desselben Beispiels.
   
    """
    smaller_id = features_data_frame[["pokemon_id", "opponent_id"]].min(axis=1)
    larger_id = features_data_frame[["pokemon_id", "opponent_id"]].max(axis=1)
    return smaller_id.astype(str) + "-" + larger_id.astype(str)


def main() -> None:
    features_data_frame = pd.read_csv(DATA_DIRECTORY / "features.csv")

    feature_values = features_data_frame[FEATURE_COLUMN_NAMES]
    target_values = features_data_frame[TARGET_COLUMN_NAME]
    pairing_groups = build_pairing_groups(features_data_frame)

    # Split nach Paarung, nicht nach Zeile (siehe build_pairing_groups).
    group_splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SET_SHARE, random_state=RANDOM_SEED
    )
    train_indexes, test_indexes = next(
        group_splitter.split(feature_values, target_values, groups=pairing_groups)
    )

    training_features = feature_values.iloc[train_indexes]
    training_target = target_values.iloc[train_indexes]
    test_features = feature_values.iloc[test_indexes]
    test_target = target_values.iloc[test_indexes]

    print(f"Training: {len(training_features):,} Zeilen")
    print(f"Test:     {len(test_features):,} Zeilen (haelt das Modell nie zu Gesicht bekommen)")
    print(f"\nTrainiere Random Forest mit {NUMBER_OF_TREES} Baeumen ...\n")

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        random_state=RANDOM_SEED,
        n_jobs=-1,  # alle CPU-Kerne nutzen
    )
    model.fit(training_features, training_target)

    accuracy = model.score(test_features, test_target)
    print(f"Accuracy auf den Testdaten: {accuracy:.1%}")
    print("(50 % waere Muenzwurf, da die Klassen fast genau 50/50 verteilt sind)\n")

    predicted_target = model.predict(test_features)
    print(classification_report(test_target, predicted_target, target_names=["verliert", "gewinnt"]))

    print("Confusion Matrix (Zeilen = Wahrheit, Spalten = Vorhersage):")
    print(confusion_matrix(test_target, predicted_target))

    print("\n=== Feature Importance: worauf das Modell tatsaechlich achtet ===")
    feature_importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMN_NAMES
    ).sort_values(ascending=False)
    for feature_name, importance in feature_importances.items():
        balken = "#" * int(importance * 100)
        print(f"  {feature_name:28s} {importance:6.1%} {balken}")

    # Feature-Namen mitspeichern, damit die API spaeter garantiert dieselbe
    # Spalten-Reihenfolge verwendet wie das Training.
    joblib.dump(
        {"model": model, "feature_column_names": FEATURE_COLUMN_NAMES}, MODEL_FILE_PATH
    )
    print(f"\nModell gespeichert -> {MODEL_FILE_PATH}")


if __name__ == "__main__":
    main()
