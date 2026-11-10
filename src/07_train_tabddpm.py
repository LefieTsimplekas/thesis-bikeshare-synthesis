"""
07_train_tabddpm.py

Εκπαίδευση TabDDPM (synthcity, plugin "ddpm") για τα σενάρια του Επιπέδου 1 και
γένεση συνθετικών διαδρομών (§4, §5).

Κάθε σενάριο εκπαιδεύεται ΜΟΝΟ με το {σενάριο}_train του 06. Οι υπερπαράμετροι
είναι ίδιες σε όλα τα σενάρια (config.TABDDPM_PARAMS), ώστε να διαφέρει μόνο η
πηγή των δεδομένων. Το London test δεν διαβάζεται εδώ.

Τρεις προσαρμογές στο synthcity 0.2.12 (καταγράφονται στη μεθοδολογία):
  1. Κατηγορικές στήλες: το synthcity θεωρεί κατηγορική μόνο στήλη με <= 10 τιμές,
     άρα η ώρα (24) και ο μήνας (12) θα γίνονταν συνεχείς. Εδώ οι 6 κατηγορικές
     του schema.json δηλώνονται ρητά, με τον αριθμό κατηγοριών του schema.
  2. Ανακάτεμα: ο DataLoader του synthcity δεν ανακατεύει τις γραμμές, ενώ τα
     αρχεία του 06 είναι ταξινομημένα. Εδώ οι παρτίδες ανακατεύονται σε κάθε εποχή
     (με seed).
  3. Βήματα: το synthcity μετρά εποχές (n_iter). Εδώ ορίζονται βήματα
     βελτιστοποίησης και μετατρέπονται σε εποχές για το μέγεθος του συνόλου.
  4. Γένεση: χωρίς το φίλτρο απόρριψης του plugin.generate() (βλ. run()).

Seeds (§7, 3 training × 3 generation seeds):
  training seeds 42, 43, 44 · generation seeds 1, 2, 3 ανά εκπαιδευμένο μοντέλο

Χρήση:
  python src/07_train_tabddpm.py --scenario D --pilot-rows 50000 --steps 2000 --gen-rows 20000
  python src/07_train_tabddpm.py --scenario A --seed 43 --gen-seeds 1 2 3
  python src/07_train_tabddpm.py --scenario A --seed 42 --generate-only --gen-seeds 1 2 3

Έξοδοι ανά σενάριο:
  data/synthetic/tabddpm_{σενάριο}_s{seed}_g{gen}.parquet  (γένεση + is_weekend, part_of_day)
  models/tabddpm_{σενάριο}_s{seed}.pkl                      (εκπαιδευμένο μοντέλο)
  artifacts/training/tabddpm_{σενάριο}_s{seed}.json         (παράμετροι, χρόνοι, απώλεια, γενέσεις)
  Η πρώτη εκτέλεση (seed 42) έχει ονόματα χωρίς seed (tabddpm_{σενάριο}.*).
"""

import os
# Το Colab ορίζει MPLBACKEND για το notebook· το matplotlib του ξεχωριστού
# περιβάλλοντος δεν το γνωρίζει και το synthcity σταματά στην εισαγωγή.
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import synthcity.plugins.core.models.tabular_ddpm as tddpm
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load_from_file, save_to_file

from config import SAMPLING_SEED, TABDDPM_PARAMS
from model_io import load_schema, load_table, encode, decode, add_derived, category_counts

SYNTHETIC_DIR = Path("data/synthetic")
MODELS_DIR = Path("models")
TRAIN_LOG_DIR = Path("artifacts/training")


# ---------------------------------------------------------------------------
# Προσαρμογές του synthcity
# ---------------------------------------------------------------------------
def patch_synthcity(schema: dict, seed: int) -> None:
    counts = category_counts(schema)

    def discrete_columns(X, max_classes=10, return_counts=False):
        cols = [c for c in X.columns if c in counts]
        return [(c, counts[c]) for c in cols] if return_counts else cols

    def shuffled_loader(dataset, batch_size):
        g = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           generator=g)

    tddpm.discrete_columns = discrete_columns
    tddpm.DataLoader = shuffled_loader


def build_plugin(n_rows: int, params: dict, device: str, seed: int):
    steps_per_epoch = math.ceil(n_rows / params["batch_size"])
    n_epochs = max(1, math.ceil(params["steps"] / steps_per_epoch))
    plugin = Plugins().get(
        "ddpm",
        n_iter=n_epochs,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        num_timesteps=params["num_timesteps"],
        model_type="mlp",
        model_params=dict(n_layers_hidden=params["n_layers_hidden"],
                          n_units_hidden=params["n_units_hidden"], dropout=0.0),
        is_classification=False,
        device=torch.device(device),
        random_state=seed,
        log_interval=100,
    )
    # οι κατηγορικές (ακέραιοι 0..K-1, έως 24 τιμές) να μη γίνουν συνεχείς στον encoder
    plugin.encoder.categorical_limit = 24
    return plugin, n_epochs, steps_per_epoch


# ---------------------------------------------------------------------------
def train_model(scenario: str, args, schema: dict, device: str, seed: int):
    """Εκπαίδευση ενός TabDDPM με το δοσμένο training seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    patch_synthcity(schema, seed)

    train = load_table(f"{scenario}_train")
    if args.pilot_rows:
        train = train.sample(n=min(args.pilot_rows, len(train)), random_state=seed)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    X = encode(train, schema)

    for col, k in category_counts(schema).items():
        missing = set(range(k)) - set(X[col].unique())
        if missing:
            print(f"  note: {col} has no rows for categories {sorted(missing)}")

    params = dict(TABDDPM_PARAMS)
    if args.steps:
        params["steps"] = args.steps
    plugin, n_epochs, spe = build_plugin(len(X), params, device, seed)
    print(f"Rows {len(X):,} | batch {params['batch_size']} | {spe} steps/epoch | "
          f"{n_epochs} epochs ≈ {n_epochs * spe:,} steps | device {device} | seed {seed}")

    t0 = time.time()
    plugin.fit(GenericDataLoader(X))
    t_fit = time.time() - t0
    real_steps = n_epochs * spe
    print(f"Training time {t_fit / 60:.1f} min ({real_steps / t_fit:.1f} steps/s)")
    plugin.model.dataloader = None           # δεν χρειάζεται πια· δεν αποθηκεύεται

    info = {"train_rows": len(X), "params": params, "epochs": n_epochs, "steps": real_steps,
            "fit_minutes": round(t_fit / 60, 2), "steps_per_second": round(real_steps / t_fit, 2)}
    loss = getattr(plugin, "loss_history", None)
    info["loss_history_[step,mloss,gloss,loss]"] = (
        None if loss is None else pd.DataFrame(loss).values.tolist())
    return plugin, train, info


def generate(plugin, schema: dict, n: int, gen_seed: int) -> tuple[pd.DataFrame, float]:
    """Γένεση απευθείας από το μοντέλο με δικό της seed. Το plugin.generate() του
    synthcity απορρίπτει και ξαναπαράγει γραμμές εκτός των ορίων του train, κάτι που
    αλλάζει την κατανομή και κρύβει αδυναμίες του μοντέλου· εδώ κρατάμε ό,τι παράγει."""
    torch.manual_seed(gen_seed)
    torch.cuda.manual_seed_all(gen_seed)
    np.random.seed(gen_seed)
    t0 = time.time()
    with torch.no_grad():
        syn = plugin.model.generate(n)
    syn = plugin.encoder.inverse_transform(syn)
    cols = list(schema["categorical"]) + schema["continuous"]
    return add_derived(decode(syn[cols], schema)), time.time() - t0


def quick_check(train: pd.DataFrame | None, syn: pd.DataFrame) -> None:
    rows = ([("train", train)] if train is not None else []) + [("synthetic", syn)]
    for name, d in rows:
        print(f"  {name:<9} e-bike {100 * (d.rideable_type == 'electric_bike').mean():5.1f}% | "
              f"round trip {100 * d.round_trip.mean():4.1f}% | "
              f"median duration {d.duration_min.median():5.2f} min | "
              f"median distance {d.distance_km.median():5.2f} km")


def run(scenario: str, args, schema: dict, device: str) -> None:
    seed = args.seed
    print(f"\n===== TabDDPM | scenario {scenario} | training seed {seed} =====")
    for d in (SYNTHETIC_DIR, MODELS_DIR, TRAIN_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pilot = "_pilot" if args.pilot_rows else ""
    model_tag = f"tabddpm_{scenario}_s{seed}{pilot}"
    log_path = TRAIN_LOG_DIR / f"{model_tag}.json"

    if args.generate_only:
        # Μοντέλο που έχει ήδη εκπαιδευτεί. Η πρώτη εκτέλεση (seed 42) είχε όνομα χωρίς seed.
        candidates = [MODELS_DIR / f"{model_tag}.pkl", MODELS_DIR / f"tabddpm_{scenario}{pilot}.pkl"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No saved model among {[str(c) for c in candidates]}")
        print(f"Loading {path}")
        plugin = load_from_file(path)
        train = None
        old = next((p for p in [log_path, TRAIN_LOG_DIR / f"tabddpm_{scenario}{pilot}.json"]
                    if p.exists()), None)
        log = json.loads(old.read_text(encoding="utf-8")) if old else {}
    else:
        plugin, train, info = train_model(scenario, args, schema, device, seed)
        save_to_file(MODELS_DIR / f"{model_tag}.pkl", plugin)
        log = {"scenario": scenario, "pilot_rows": args.pilot_rows, "seed": seed,
               "device": device, "torch_threads": torch.get_num_threads(), **info}

    gens = log.get("generations", {})
    for g in args.gen_seeds:
        syn, t_gen = generate(plugin, schema, args.gen_rows, g)
        out = SYNTHETIC_DIR / f"{model_tag}_g{g}.parquet"
        syn.to_parquet(out, index=False)
        gens[str(g)] = {"rows": len(syn), "minutes": round(t_gen / 60, 2), "file": out.name}
        print(f"Generation seed {g}: {len(syn):,} rows in {t_gen / 60:.1f} min -> {out}")
        quick_check(train, syn)
        train = None                         # το train τυπώνεται μόνο την πρώτη φορά

    log.update({"scenario": scenario, "seed": seed, "generations": gens})
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", default=["A", "B", "C", "D"],
                    choices=["A", "B", "C", "D"])
    ap.add_argument("--seed", type=int, default=SAMPLING_SEED, help="training seed")
    ap.add_argument("--gen-seeds", type=int, nargs="+", default=[1],
                    help="seeds γένεσης· ένα αρχείο συνθετικών ανά seed")
    ap.add_argument("--generate-only", action="store_true",
                    help="χωρίς εκπαίδευση: φορτώνει το αποθηκευμένο μοντέλο του --seed")
    ap.add_argument("--pilot-rows", type=int, default=0,
                    help="πιλοτική εκτέλεση: τυχαίο υποσύνολο τόσων γραμμών του train")
    ap.add_argument("--steps", type=int, default=0, help="αντικαθιστά το TABDDPM_PARAMS['steps']")
    ap.add_argument("--gen-rows", type=int, default=200_000,
                    help="συνθετικές γραμμές ανά seed γένεσης (προεπιλογή: όσες το London test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schema = load_schema()
    for s in args.scenario:
        run(s, args, schema, device)


if __name__ == "__main__":
    main()
