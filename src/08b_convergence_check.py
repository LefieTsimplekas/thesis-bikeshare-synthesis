"""
08b_convergence_check.py

Έλεγχος σύγκλισης των CTGAN και TVAE (SDV) πριν από το Επίπεδο 2 (§4).

Το 08 έδειξε ότι με τις 300 εποχές των Xu et al. (2019) ένα CTGAN στις 720.809
γραμμές θέλει ~3 ώρες (~430.000 βήματα), ενώ με ίδιο προϋπολογισμό βημάτων με το
TabDDPM (20.000 βήματα ≈ 14 εποχές) ~12 λεπτά. Εδώ κάθε μοντέλο εκπαιδεύεται στο
σενάριο C με 14, 50 και 100 εποχές και τρία training seeds (42, 43, 44, μία γένεση
το καθένα), ώστε το 09 να δείξει αν περισσότερες εποχές βελτιώνουν ουσιαστικά τα
δεδομένα. Η πρώτη εκτέλεση (21/11) είχε μόνο το seed 42· στο CTGAN το μέσο TVD άλλαξε
μη μονότονα με τις εποχές (0,154 / 0,183 / 0,167), δηλαδή ο θόρυβος ενός seed ήταν
μεγαλύτερος από το όριο του κανόνα, γι' αυτό ο κανόνας εφαρμόζεται σε μέσους όρους
τριών seeds, όπως στο Επίπεδο 1.

Κανόνας απόφασης (ορίστηκε πριν από την εκτέλεση, 21/11): περισσότερες εποχές
θεωρούνται καλύτερες αν βελτιώνουν κατά περισσότερο από 10% του transfer gap του C
(TabDDPM, Επίπεδο 1) τουλάχιστον μία από τις R² του TSTR διάρκειας, detection
χωρίς ημερολόγιο και μέσο TVD.

Ρυθμίσεις ίδιες με αυτές που θα έχει το Επίπεδο 2:
  - προεπιλογές SDV 2.2 (αρχιτεκτονική, batch 500, ρυθμοί μάθησης) εκτός από τις εποχές
  - enforce_min_max_values=False και enforce_rounding=False: καμία μετεπεξεργασία
    πέρα από αυτή του 07, όπως στο TabDDPM, που γεννά χωρίς το φίλτρο του synthcity
  - μετεπεξεργασία ίδια με το 07 (model_io.add_derived): απόσταση 0 στις κυκλικές,
    is_weekend, part_of_day
  - 200.000 γραμμές ανά σύνολο, όσες το London test

Περιβάλλον: SDV 2.2 με το system Python του Colab (notebook, ενότητα 7β), όχι με το
/content/py311 του synthcity (βλ. 08). Το London test δεν διαβάζεται εδώ.

Χρήση (από τη ρίζα του repo):
  python src/08b_convergence_check.py
  python src/08b_convergence_check.py --models tvae --epochs 14 --seeds 42 --rows 20000   # γρήγορη δοκιμή

Έξοδοι (ένα σύνολο ανά μοντέλο × εποχές × seed· όσα υπάρχουν ήδη παραλείπονται):
  data/synthetic/{μοντέλο}_C_e{εποχές}_s{seed}_g1.parquet
  models/{μοντέλο}_C_e{εποχές}_s{seed}.pkl             (για επαναχρησιμοποίηση στο Επίπεδο 2)
  artifacts/training/{μοντέλο}_C_e{εποχές}_s{seed}.json (χρόνοι, βήματα, απώλεια ανά εποχή)
Το seed 42 της πρώτης εκτέλεσης δεν έχει αποθηκευμένο μοντέλο.
Φόρτωση μοντέλου (SDV 2.2): from sdv.utils import load_synthesizer; load_synthesizer(path)
Αξιολόγηση μετά, τοπικά:
  python src/09_evaluate.py --synthetic "data/synthetic/*_C_e*_g1.parquet"
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
BATCH = 500  # προεπιλογή SDV 2.2


def build_metadata(schema: dict):
    from sdv.metadata import Metadata

    columns = {c: {"sdtype": "categorical"} for c in schema["categorical"]}
    columns.update({c: {"sdtype": "numerical"} for c in schema["continuous"]})
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": columns}}, "relationships": []})


def make(model: str, metadata, epochs: int):
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

    kw = dict(epochs=epochs, batch_size=BATCH, enable_gpu=True,
              enforce_min_max_values=False, enforce_rounding=False)
    if model == "ctgan":
        return CTGANSynthesizer(metadata, **kw)
    if model == "tvae":
        return TVAESynthesizer(metadata, **kw)
    raise ValueError(model)


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
    except Exception:  # η συνάρτηση δεν υπάρχει σε κάθε μοντέλο/έκδοση
        return None
    return None if loss is None else loss.to_dict(orient="records")


def run_one(model: str, epochs: int, seed: int, train: pd.DataFrame, schema: dict, metadata, args) -> None:
    tag = f"{model}_{args.scenario}_e{epochs}_s{seed}"
    out = SYNTHETIC_DIR / f"{tag}_g{args.gen_seed}.parquet"
    if out.exists() and not args.rows:
        print(f"{out.name}: υπάρχει ήδη, παραλείπεται")
        return
    if args.rows:
        out = SYNTHETIC_DIR / f"{tag}_pilot_g{args.gen_seed}.parquet"

    print(f"\n===== {model} | scenario {args.scenario} | {epochs} epochs | seed {seed} =====", flush=True)
    synth = make(model, metadata, epochs)
    set_seed(seed)
    t0 = time.time()
    synth.fit({TABLE: train})
    t_fit = time.time() - t0
    if not args.rows:
        synth.save(str(MODELS_DIR / f"{tag}.pkl"))
    steps = epochs * math.ceil(len(train) / BATCH)
    print(f"Training: {t_fit / 60:.1f} min, {steps:,} steps", flush=True)

    set_seed(args.gen_seed)
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
    log = {"model": model, "scenario": args.scenario, "epochs": epochs, "steps": steps,
           "batch_size": BATCH, "seed": seed, "gen_seed": args.gen_seed,
           "train_rows": len(train), "sdv": sdv.__version__,
           "enforce_min_max_values": False, "enforce_rounding": False,
           "fit_minutes": round(t_fit / 60, 2), "gen_minutes": round(t_gen / 60, 2),
           "file": out.name, "loss": loss_table(synth)}
    (TRAIN_LOG_DIR / f"{tag}{'_pilot' if args.rows else ''}.json").write_text(
        json.dumps(log, indent=1, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="C", choices=["A", "B", "C", "D"])
    ap.add_argument("--models", nargs="+", default=["tvae", "ctgan"], choices=["tvae", "ctgan"])
    ap.add_argument("--epochs", nargs="+", type=int, default=[14, 50, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SAMPLING_SEED, SAMPLING_SEED + 1, SAMPLING_SEED + 2])
    ap.add_argument("--gen-seed", type=int, default=1)
    ap.add_argument("--rows", type=int, default=None, help="μόνο οι πρώτες γραμμές (δοκιμή)")
    args = ap.parse_args()

    for d in (SYNTHETIC_DIR, MODELS_DIR, TRAIN_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    schema = load_schema()
    cols = list(schema["categorical"]) + schema["continuous"]
    train = to_schema_types(load_table(f"{args.scenario}_train")[cols], schema)
    if args.rows:
        train = train.sample(n=min(args.rows, len(train)), random_state=SAMPLING_SEED).reset_index(drop=True)
    metadata = build_metadata(schema)

    # TVAE πρώτα (γρηγορότερο), και από τις λίγες προς τις πολλές εποχές, ώστε μια
    # διακοπή να αφήνει τα πιο χρήσιμα αποτελέσματα.
    t0 = time.time()
    for model in args.models:
        for epochs in sorted(args.epochs):
            for seed in args.seeds:
                run_one(model, epochs, seed, train, schema, metadata, args)
    print(f"\nΤέλος σε {(time.time() - t0) / 60:.0f} λεπτά")


if __name__ == "__main__":
    main()
