"""
08_pilot_timing.py

Δοκιμαστική χρονομέτρηση των CTGAN, TVAE και GaussianCopula (SDV) πριν από το
Επίπεδο 2. Απαντά σε δύο ερωτήματα:

  1. Πόσο κοστίζει η επιλογή Α (καλύτερο καθεστώς + D) και πόσο η Β (A, B, C, D);
  2. Πόσο κοστίζει κάθε προϋπολογισμός εκπαίδευσης: οι προεπιλογές του SDV
     (300 εποχές) ή ίδιος αριθμός βημάτων με το TabDDPM (20.000);

Τι κάνει: για τα CTGAN και TVAE εκπαιδεύει δύο φορές, με 1 και με k εποχές, και
από τη διαφορά βρίσκει τον χρόνο ανά εποχή και το σταθερό κόστος (προεπεξεργασία).
Μετά παράγει 200.000 γραμμές. Το GaussianCopula εκπαιδεύεται κανονικά, γιατί δεν
έχει εποχές. Από αυτά υπολογίζει τον χρόνο ολόκληρων εκπαιδεύσεων και του σχεδίου.

Δεν παράγει δεδομένα για ανάλυση και δεν διαβάζει το London test. Οι
υπερπαράμετροι του Επιπέδου 2 (εποχές, enforce_min_max_values) δεν έχουν
αποφασιστεί· εδώ μετριέται μόνο ο χρόνος.

Περιβάλλον: το SDV 2.2 σε Python 3.13 θέλει numpy >= 2.1 και torch >= 2.6, άρα ΔΕΝ
μπαίνει στο περιβάλλον του synthcity (torch 2.2.2, numpy 1.26.4). Στο Colab τρέχει
με το system Python (notebook, ενότητα 7), όχι με το /content/py311.

Χρήση (από τη ρίζα του repo):
    pip install sdv==2.2.0
    python src/08_pilot_timing.py --scenario C
    python src/08_pilot_timing.py --scenario C --rows 50000 --probe 2   # γρήγορη δοκιμή

Ορίσματα:
    --scenario   σενάριο του 06 (A, B, C, D)· διαβάζεται το data/model_input/{σενάριο}_train
    --train      εναλλακτικά, ένα αρχείο parquet ή csv
    --probe      εποχές της δεύτερης εκπαίδευσης (προεπιλογή 3)
    --rows       χρήση μόνο των πρώτων γραμμών, για γρήγορη δοκιμή (προεπιλογή: όλες)
    --models     ποια μοντέλα (προεπιλογή: ctgan tvae gaussiancopula)
    --out        αρχείο JSON (προεπιλογή artifacts/level2/08_pilot_timing.json)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_io import load_schema, load_table

# Στήλες γένεσης από το schema.json του 06 (ίδιες με το 07).
_SCHEMA = load_schema()
CATEGORICAL = list(_SCHEMA["categorical"])
CONTINUOUS = list(_SCHEMA["continuous"])

N_SYNTH = 200_000          # γραμμές ανά συνθετικό σύνολο, όπως στο 07
TABDDPM_STEPS = 20_000     # προϋπολογισμός βημάτων του TabDDPM
SDV_DEFAULT_EPOCHS = 300   # προεπιλογή CTGAN/TVAE στο SDV 2.2
BATCH = 500                # προεπιλογή batch_size στο SDV 2.2
SEEDS_TRAIN, SEEDS_GEN = 3, 3
TABLE = "trips"

# Νέες διαμορφώσεις ανά μοντέλο (μοντέλο × σενάριο). Το TabDDPM έχει ήδη τρέξει.
PLANS = {"A (καλύτερο + D)": 2, "B (A, B, C, D)": 4}


def load_train(scenario: str, path: Path | None, rows: int | None) -> pd.DataFrame:
    if path is None:
        df, name = load_table(f"{scenario}_train"), f"{scenario}_train"
    else:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        name = path.name
    missing = [c for c in CATEGORICAL + CONTINUOUS if c not in df.columns]
    if missing:
        raise SystemExit(f"Λείπουν στήλες από το {name}: {missing}")
    df = df[CATEGORICAL + CONTINUOUS].copy()
    if rows:
        df = df.iloc[:rows].copy()
    for c in CONTINUOUS:
        df[c] = df[c].astype(float)
    return df


def build_metadata():
    from sdv.metadata import Metadata

    columns = {c: {"sdtype": "categorical"} for c in CATEGORICAL}
    columns.update({c: {"sdtype": "numerical"} for c in CONTINUOUS})
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": columns}}, "relationships": []})


def make(model: str, metadata, epochs: int | None = None):
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

    if model == "ctgan":
        return CTGANSynthesizer(metadata, epochs=epochs, batch_size=BATCH, enable_gpu=True)
    if model == "tvae":
        return TVAESynthesizer(metadata, epochs=epochs, batch_size=BATCH, enable_gpu=True)
    if model == "gaussiancopula":
        return GaussianCopulaSynthesizer(metadata)
    raise ValueError(model)


def timed_fit(model: str, metadata, df: pd.DataFrame, epochs: int | None):
    synth = make(model, metadata, epochs)
    t0 = time.perf_counter()
    synth.fit({TABLE: df})  # στο SDV 2.x το fit δέχεται {πίνακας: DataFrame}
    return synth, time.perf_counter() - t0


def timed_sample(synth) -> float:
    t0 = time.perf_counter()
    out = synth.sample(num_rows=N_SYNTH)
    out = out[TABLE] if isinstance(out, dict) else out  # το SDV 2.x επιστρέφει {πίνακας: DataFrame}
    assert len(out) == N_SYNTH
    return time.perf_counter() - t0


def plan_hours(train_s: float, sample_s: float, configs: int) -> float:
    trainings = configs * SEEDS_TRAIN
    generations = trainings * SEEDS_GEN
    return (trainings * train_s + generations * sample_s) / 3600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="C", choices=["A", "B", "C", "D"])
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--probe", type=int, default=3)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["ctgan", "tvae", "gaussiancopula"])
    ap.add_argument("--out", type=Path, default=Path("artifacts/level2/08_pilot_timing.json"))
    args = ap.parse_args()
    if args.probe < 2:
        raise SystemExit("--probe πρέπει να είναι τουλάχιστον 2")

    import sdv
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    df = load_train(args.scenario, args.train, args.rows)
    n = len(df)
    steps_per_epoch = math.ceil(n / BATCH)
    epochs_equal_steps = math.ceil(TABDDPM_STEPS / steps_per_epoch)
    metadata = build_metadata()

    env = {
        "python": platform.python_version(),
        "sdv": sdv.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "καμία (CPU)",
        "scenario": args.scenario if args.train is None else str(args.train),
        "rows": n,
        "steps_per_epoch": steps_per_epoch,
        "epochs_equal_to_tabddpm_steps": epochs_equal_steps,
    }
    print(json.dumps(env, ensure_ascii=False, indent=2))

    results: dict = {"env": env, "models": {}}
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        if model == "gaussiancopula":
            synth, fit_s = timed_fit(model, metadata, df, None)
            sample_s = timed_sample(synth)
            budgets = {"μοναδική εκπαίδευση": fit_s}
            res = {"fit_s": fit_s, "sample_s": sample_s}
        else:
            # Προθέρμανση (δεν χρονομετρείται): η πρώτη κλήση έχει κόστος εκκίνησης
            # (CUDA, imports), που αλλοιώνει τη διαφορά των δύο μετρήσεων.
            timed_fit(model, metadata, df.iloc[:5_000], 1)
            _, t1 = timed_fit(model, metadata, df, 1)
            synth, tk = timed_fit(model, metadata, df, args.probe)
            per_epoch = (tk - t1) / (args.probe - 1)
            if per_epoch <= 0:  # θόρυβος μέτρησης: συντηρητική εκτίμηση χωρίς σταθερό κόστος
                print("  Προσοχή: μη αξιόπιστη διαφορά, χρήση tk / probe ως χρόνου ανά εποχή.")
                per_epoch = tk / args.probe
            overhead = max(t1 - per_epoch, 0.0)
            sample_s = timed_sample(synth)
            budgets = {
                f"προεπιλογή SDV ({SDV_DEFAULT_EPOCHS} εποχές)": overhead + SDV_DEFAULT_EPOCHS * per_epoch,
                f"ίσα βήματα με TabDDPM ({epochs_equal_steps} εποχές)": overhead + epochs_equal_steps * per_epoch,
            }
            res = {"fit_1_epoch_s": t1, f"fit_{args.probe}_epochs_s": tk, "per_epoch_s": per_epoch,
                   "overhead_s": overhead, "sample_s": sample_s}
        res["one_training_min"] = {b: s / 60 for b, s in budgets.items()}
        res["plan_hours"] = {b: {p: plan_hours(s, sample_s, c) for p, c in PLANS.items()} for b, s in budgets.items()}
        results["models"][model] = res
        print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)

    # Σύνοψη: ώρες για όλο το Επίπεδο 2 (τρία μοντέλα), ανά σχέδιο και προϋπολογισμό.
    print("\n=== Σύνοψη: ώρες για τα νέα μοντέλα του Επιπέδου 2 ===")
    for label, key in [("προεπιλογή SDV", "προεπιλογή"), ("ίσα βήματα με TabDDPM", "ίσα βήματα")]:
        for plan in PLANS:
            total = 0.0
            for m, res in results["models"].items():
                budget = next((b for b in res["plan_hours"] if b.startswith(key)), None) or next(iter(res["plan_hours"]))
                total += res["plan_hours"][budget][plan]
            print(f"  {label:<22} · σχέδιο {plan:<18}: {total:6.1f} ώρες")
    print("Οι ώρες δεν περιλαμβάνουν την αξιολόγηση (09), που τρέχει τοπικά.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nΑποτελέσματα: {args.out}")


if __name__ == "__main__":
    main()
