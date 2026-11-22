"""
08c_train_level2.py

Επίπεδο 2: εκπαίδευση CTGAN, TVAE και GaussianCopula (SDV 2.2) σε όλα τα σενάρια
(επιλογή Β, 22/11) και γένεση 200.000 γραμμών από το καθένα. Το TabDDPM του
Επιπέδου 2 είναι τα μοντέλα του Επιπέδου 1 (07), ήδη για A–D × 3 seeds.

Εποχές από τον έλεγχο σύγκλισης (08b, μέσοι όροι 3 seeds, κανόνας 10% του transfer gap):
  CTGAN 50 εποχές · TVAE 100 εποχές · GaussianCopula χωρίς εποχές.
Ίδιες υπόλοιπες ρυθμίσεις με το 08b: batch 500, enforce_min_max_values=False,
enforce_rounding=False, καμία ρύθμιση υπερπαραμέτρων, ίδια μετεπεξεργασία με το 07.

Αυτόματη συνέχεια: μια εκτέλεση θεωρείται έτοιμη όταν υπάρχουν ΚΑΙ το parquet ΚΑΙ
το μοντέλο (.pkl). Τα σύνολα του σεναρίου C για CTGAN 50 και TVAE 100 υπάρχουν ήδη από
το 08b (ίδια ονόματα) και παραλείπονται. Αν υπάρχει το parquet αλλά όχι το μοντέλο
(π.χ. το seed 42 του 08b, πριν αποθηκεύονται τα μοντέλα), η εκτέλεση γίνεται ξανά και
αντικαθιστά και τα δύο, ώστε το σύνολο που αξιολογείται και το μοντέλο που δίνει το
pool του Επιπέδου 1b να είναι το ίδιο μοντέλο· το 09 ξανατρέχει για αυτά τα αρχεία.

Σειρά: GaussianCopula (λεπτά) → D (για το transfer gap) → C → A → B, TVAE πριν από
CTGAN, ώστε μια διακοπή να αφήνει τα πιο χρήσιμα αποτελέσματα.

Χρήση:
  python src/08c_train_level2.py                       # όλα όσα λείπουν
  python src/08c_train_level2.py --scenarios D --models tvae
  python src/08c_train_level2.py --dry-run             # μόνο τι θα τρέξει

Έξοδοι:
  data/synthetic/{ctgan_X_e50 | tvae_X_e100 | gaussiancopula_X}_s{seed}_g1.parquet
  models/{ίδιο όνομα χωρίς _g1}.pkl     (φόρτωση: sdv.utils.load_synthesizer)
  artifacts/training/{ίδιο όνομα}.json
Αξιολόγηση:
  python src/09_evaluate.py --synthetic "data/synthetic/ctgan_*_e50_s*_g1.parquet" \
      "data/synthetic/tvae_*_e100_s*_g1.parquet" "data/synthetic/gaussiancopula_*_g1.parquet"
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import SAMPLING_SEED
from model_io import add_derived, load_schema, load_table

SYNTHETIC_DIR = Path("data/synthetic")
MODELS_DIR = Path("models")
TRAIN_LOG_DIR = Path("artifacts/training")
TABLE = "trips"
N_SYNTH = 200_000
BATCH = 500                                   # προεπιλογή SDV 2.2
EPOCHS = {"ctgan": 50, "tvae": 100, "gaussiancopula": None}
SCENARIO_ORDER = ["D", "C", "A", "B"]
MODEL_ORDER = ["gaussiancopula", "tvae", "ctgan"]


def build_metadata(schema: dict):
    from sdv.metadata import Metadata

    columns = {c: {"sdtype": "categorical"} for c in schema["categorical"]}
    columns.update({c: {"sdtype": "numerical"} for c in schema["continuous"]})
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": columns}}, "relationships": []})


def make(model: str, metadata):
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

    common = dict(enforce_min_max_values=False, enforce_rounding=False)
    if model == "gaussiancopula":
        return GaussianCopulaSynthesizer(metadata, **common)
    kw = dict(epochs=EPOCHS[model], batch_size=BATCH, enable_gpu=True, **common)
    return (CTGANSynthesizer if model == "ctgan" else TVAESynthesizer)(metadata, **kw)


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_schema_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Οι κατηγορικές με τον τύπο του schema· έλεγχος ότι όλες οι τιμές επιτρέπονται."""
    out = df.copy()
    for col, cats in schema["categorical"].items():
        if not isinstance(cats[0], str):
            out[col] = out[col].astype("int64")
        bad = set(out[col].unique()) - set(cats)
        if bad:
            raise ValueError(f"{col}: τιμές εκτός schema {sorted(bad)[:5]}")
    for col in schema["continuous"]:
        out[col] = out[col].astype(float)
    return out


def loss_table(synth) -> list | None:
    try:
        loss = synth.get_loss_values()
    except Exception:  # δεν υπάρχει στο GaussianCopula
        return None
    return None if loss is None else loss.to_dict(orient="records")


def tag_of(model: str, scenario: str, seed: int) -> str:
    e = EPOCHS[model]
    return f"{model}_{scenario}{'' if e is None else f'_e{e}'}_s{seed}"


def status(tag: str) -> str:
    has_out = (SYNTHETIC_DIR / f"{tag}_g1.parquet").exists()
    has_model = (MODELS_DIR / f"{tag}.pkl").exists()
    if has_out and has_model:
        return "done"
    return "retrain" if has_out else "todo"


def run_one(model: str, scenario: str, seed: int, train: pd.DataFrame, schema: dict, metadata) -> None:
    tag = tag_of(model, scenario, seed)
    out = SYNTHETIC_DIR / f"{tag}_g1.parquet"
    st = status(tag)
    if st == "done":
        print(f"{out.name}: υπάρχει ήδη, παραλείπεται")
        return
    if st == "retrain":
        print(f"{out.name}: υπάρχει χωρίς μοντέλο -> εκπαιδεύεται ξανά και αντικαθίσταται "
              f"(ξανατρέξε το 09 γι' αυτό)")

    print(f"\n===== {model} | scenario {scenario} | "
          f"{'' if EPOCHS[model] is None else f'{EPOCHS[model]} epochs | '}seed {seed} =====", flush=True)
    synth = make(model, metadata)
    set_seed(seed)
    t0 = time.time()
    synth.fit({TABLE: train})
    t_fit = time.time() - t0
    synth.save(str(MODELS_DIR / f"{tag}.pkl"))
    steps = None if EPOCHS[model] is None else EPOCHS[model] * math.ceil(len(train) / BATCH)
    print(f"Training: {t_fit / 60:.1f} min" + ("" if steps is None else f", {steps:,} steps"), flush=True)

    set_seed(1)
    t0 = time.time()
    syn = synth.sample(num_rows=N_SYNTH)
    syn = syn[TABLE] if isinstance(syn, dict) else syn
    t_gen = time.time() - t0
    cols = list(schema["categorical"]) + schema["continuous"]
    syn = add_derived(to_schema_types(syn[cols], schema))
    syn.to_parquet(out, index=False)
    print(f"Generation: {len(syn):,} rows in {t_gen / 60:.1f} min -> {out}", flush=True)

    for name, d in [("train", train), ("synthetic", syn)]:
        print(f"  {name:<9} e-bike {100 * (d.rideable_type == 'electric_bike').mean():5.1f}% | "
              f"round trip {100 * d.round_trip.mean():4.1f}% | "
              f"median duration {d.duration_min.median():5.2f} min | "
              f"median distance {d.distance_km.median():5.2f} km")

    import sdv
    log = {"model": model, "scenario": scenario, "epochs": EPOCHS[model], "steps": steps,
           "batch_size": None if EPOCHS[model] is None else BATCH, "seed": seed, "gen_seed": 1,
           "train_rows": len(train), "sdv": sdv.__version__,
           "enforce_min_max_values": False, "enforce_rounding": False,
           "retrained_missing_model": st == "retrain",
           "fit_minutes": round(t_fit / 60, 2), "gen_minutes": round(t_gen / 60, 2),
           "file": out.name, "loss": loss_table(synth)}
    (TRAIN_LOG_DIR / f"{tag}.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIO_ORDER, choices=SCENARIO_ORDER)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[SAMPLING_SEED, SAMPLING_SEED + 1, SAMPLING_SEED + 2])
    ap.add_argument("--dry-run", action="store_true", help="μόνο η λίστα των εκτελέσεων")
    args = ap.parse_args()

    for d in (SYNTHETIC_DIR, MODELS_DIR, TRAIN_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Σειρά: πρώτα όλα τα GaussianCopula (γρήγορα), μετά ανά σενάριο TVAE → CTGAN
    models = [m for m in MODEL_ORDER if m in args.models]
    scenarios = [s for s in SCENARIO_ORDER if s in args.scenarios]
    plan = [(m, s, seed) for m in models if m == "gaussiancopula" for s in scenarios for seed in args.seeds]
    plan += [(m, s, seed) for s in scenarios for m in models if m != "gaussiancopula" for seed in args.seeds]

    todo = [(m, s, seed, status(tag_of(m, s, seed))) for m, s, seed in plan]
    print("Εκτελέσεις:", ", ".join(f"{tag_of(m, s, seed)} [{st}]" for m, s, seed, st in todo))
    print(f"Μένουν {sum(st != 'done' for *_, st in todo)} από {len(todo)}", flush=True)
    if args.dry_run:
        return

    schema = load_schema()
    cols = list(schema["categorical"]) + schema["continuous"]
    metadata = build_metadata(schema)
    t0 = time.time()
    loaded = {}
    for m, s, seed, st in todo:
        if st == "done":
            print(f"{tag_of(m, s, seed)}_g1.parquet: υπάρχει ήδη, παραλείπεται")
            continue
        if s not in loaded:
            loaded = {s: to_schema_types(load_table(f"{s}_train")[cols], schema)}
        run_one(m, s, seed, loaded[s], schema, metadata)
    print(f"\nΤέλος σε {(time.time() - t0) / 60:.0f} λεπτά")


if __name__ == "__main__":
    main()
