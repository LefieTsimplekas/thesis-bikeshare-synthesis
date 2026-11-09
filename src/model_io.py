"""
model_io.py

Κοινές συναρτήσεις για όλα τα γενετικά μοντέλα (TabDDPM, CTGAN, TVAE, GaussianCopula):
φόρτωση των πινάκων του 06, κωδικοποίηση των κατηγορικών στηλών σε ακέραιους
0..K-1 με βάση το schema.json (όχι με βάση τα δεδομένα, άρα χωρίς διαρροή),
αποκωδικοποίηση και υπολογισμός των στηλών που δεν παράγονται.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_INPUT_DIR = Path("data/model_input")
SCHEMA_PATH = Path("artifacts/model_input/schema.json")

# Ίδια όρια με το 01_build_featured_data.py
PART_OF_DAY = [(5, 10, "morning_peak"), (11, 15, "midday"), (16, 19, "evening_peak")]


def load_schema() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema


def load_table(name: str) -> pd.DataFrame:
    """name: π.χ. 'A_train', 'D_val', 'london_test'."""
    return pd.read_parquet(MODEL_INPUT_DIR / f"{name}.parquet")


def encode(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Κατηγορικές -> ακέραιοι 0..K-1 με τη σειρά του schema· συνεχείς ως έχουν."""
    out = df.copy()
    for col, cats in schema["categorical"].items():
        mapping = {c: i for i, c in enumerate(cats)}
        codes = out[col].map(mapping)
        if codes.isna().any():
            bad = sorted(out.loc[codes.isna(), col].unique().tolist())[:5]
            raise ValueError(f"{col}: values not in schema {bad}")
        out[col] = codes.astype("int64")
    for col in schema["continuous"]:
        out[col] = out[col].astype("float64")
    return out[list(schema["categorical"]) + schema["continuous"]]


def decode(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Αντίστροφο του encode."""
    out = df.copy()
    for col, cats in schema["categorical"].items():
        codes = np.clip(np.rint(out[col].astype(float)), 0, len(cats) - 1).astype(int)
        out[col] = pd.Series(np.asarray(cats, dtype=object)[codes], index=out.index)
        if not isinstance(cats[0], str):
            out[col] = out[col].astype("int64")
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Στήλες που υπολογίζονται μετά τη γένεση (schema['derived_after_generation'])."""
    out = df.copy()
    out.loc[out["round_trip"] == 1, "distance_km"] = 0.0
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype("int8")
    pod = np.full(len(out), "night", dtype=object)
    for lo, hi, name in PART_OF_DAY:
        pod[(out["hour"] >= lo).to_numpy() & (out["hour"] <= hi).to_numpy()] = name
    out["part_of_day"] = pod
    return out


def category_counts(schema: dict) -> dict[str, int]:
    return {c: len(v) for c, v in schema["categorical"].items()}
