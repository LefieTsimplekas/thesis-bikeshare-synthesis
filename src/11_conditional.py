"""
11_conditional.py

Επίπεδο 1b: γένεση με συνθήκη τον τύπο σταθμού αφετηρίας (origin_type).

Η συνθήκη ορίζει μόνο ΠΟΣΕΣ διαδρομές ξεκινούν από κάθε τύπο σταθμού. Το πώς είναι
μια διαδρομή μέσα σε κάθε τύπο (ώρα, διάρκεια, απόσταση, ποδήλατο) έρχεται από το
μοντέλο, άρα από τις πόλεις-πηγές. Δεν χρειάζεται νέα εκπαίδευση: ένα μεγάλο σύνολο
(pool) από το ήδη εκπαιδευμένο μοντέλο υποδειγματίζεται ανά τύπο στο μείγμα-στόχο.
Είναι η ίδια ιδέα με τη δειγματοληψία με απόρριψη του sample_from_conditions του SDV,
αλλά ίδια για όλα τα μοντέλα (και για το TabDDPM του synthcity).

Τρία μείγματα-στόχοι (δύο zero-shot εκτιμήσεις, ορισμένες πριν από την αξιολόγηση,
και αναφέρονται ΚΑΙ οι δύο· καμία δεν επιλέγεται με βάση το London train):
  stations  zero-shot. Μερίδιο των σταθμών του Λονδίνου σε κάθε τύπο (BikePoint + OSM,
            05b). Παραδοχή: ίδια ζήτηση ανά σταθμό σε όλους τους τύπους.
  rate      zero-shot. Σταθμοί του Λονδίνου ανά τύπο × διαδρομές ανά σταθμό του τύπου
            στις πόλεις-πηγές του σεναρίου. Παραδοχή: η ζήτηση ανά σταθμό κάθε τύπου
            μεταφέρεται από τις πόλεις-πηγές.
  oracle    ΟΧΙ zero-shot. Το πραγματικό μερίδιο αφετηριών στο D_train (London train).
            Άνω όριο της συνθήκης· δηλώνεται ρητά ως έλεγχος ευαισθησίας.
Καμία διαδρομή του Λονδίνου στα stations και rate.
Το London test δεν διαβάζεται εδώ.

Βήματα:
  1. mix       υπολογίζει τα μείγματα ανά σενάριο (A, B, C) και τα συγκρίνει (TVD)
               με το μείγμα του D_train, ώστε να ελέγχονται οι δύο παραδοχές.
  2. pool      (μόνο SDV) φορτώνει ένα αποθηκευμένο μοντέλο και παράγει μεγάλο pool.
               Για το TabDDPM το pool φτιάχνεται με το 07:
               python src/07_train_tabddpm.py --scenario C --seed 42 --generate-only
                      --gen-seeds 101 --gen-rows 1000000
               και μετακίνηση του αρχείου στο data/synthetic/pool/ (αλλιώς το πιάνουν
               τα μοτίβα του 09/10 ως σύνολο του Επιπέδου 1).
  3. resample  υποδειγματίζει pools στο μείγμα-στόχο, χωρίς επανάθεση όπου φτάνουν οι
               γραμμές· αλλιώς με επανάθεση και καταγραφή του ποσοστού διπλοτύπων.

Χρήση:
  python src/11_conditional.py mix
  python src/11_conditional.py pool --model models/tvae_C_e14_s42.pkl --rows 1000000
  python src/11_conditional.py resample --pool "data/synthetic/pool/*.parquet" --mix stations rate oracle
  python src/09_evaluate.py --synthetic "data/synthetic/*_cond-*.parquet"

Έξοδοι:
  artifacts/conditional/target_mix.csv        μείγματα ανά σενάριο και τύπο
  artifacts/conditional/target_mix.json       ίδια + TVD κάθε μείγματος έναντι του oracle
  data/synthetic/pool/{tag}_pool.parquet      pools (pool)
  data/synthetic/{tag}_cond-{mix}.parquet     σύνολα με συνθήκη (resample), mix = stations | rate | oracle
  artifacts/conditional/resample_{tag}_cond-{mix}.json   πλήθη ανά τύπο, διπλότυπα
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import OSM_DIR, SAMPLING_SEED
from model_io import load_schema, load_table, add_derived

OUT_DIR = Path("artifacts/conditional")
SYNTHETIC_DIR = Path("data/synthetic")
POOL_DIR = SYNTHETIC_DIR / "pool"
STATION_TYPES = OSM_DIR / "station_types.parquet"
SCENARIO_CITIES = {"A": ["chicago"], "B": ["nyc"], "C": ["chicago", "nyc"]}
CITY_TRAIN = {"chicago": "A_train", "nyc": "B_train"}
N_OUT = 200_000
TAG_RE = re.compile(r"_(?P<sc>[ABCD])(_|$)")


# ---------------------------------------------------------------------------
# 1. Μείγματα-στόχοι
# ---------------------------------------------------------------------------
def origin_shares(table: str, types: list[int]) -> pd.Series:
    s = load_table(table)["origin_type"].value_counts()
    return s.reindex(types, fill_value=0)


def compute_mixes() -> pd.DataFrame:
    types = sorted(int(t) for t in load_schema()["categorical"]["origin_type"])
    st = pd.read_parquet(STATION_TYPES, columns=["city", "origin_type"])
    stations = st.groupby(["city", "origin_type"]).size().unstack(fill_value=0)
    stations = stations.reindex(columns=types, fill_value=0)

    # Διαδρομές ανά σταθμό κάθε τύπου σε κάθε πόλη-πηγή
    rate = {}
    for city, table in CITY_TRAIN.items():
        trips = origin_shares(table, types)
        rate[city] = (trips / stations.loc[city].replace(0, np.nan)).fillna(0.0)

    rows = []
    n_lon = stations.loc["london"]
    for sc, cities in SCENARIO_CITIES.items():
        # Κάθε πόλη δίνει ένα εκτιμώμενο μείγμα· στο C ο μέσος όρος (50/50, όπως το C_train)
        est = np.mean([(n_lon * rate[c]) / (n_lon * rate[c]).sum() for c in cities], axis=0)
        src = load_table(f"{sc}_train")["origin_type"].value_counts(normalize=True)
        for k, t in enumerate(types):
            rows.append({"scenario": sc, "origin_type": t, "london_stations": int(n_lon[t]),
                         "source_share": float(src.get(t, 0.0)),
                         "stations": float(n_lon[t] / n_lon.sum()), "rate": float(est[k])})
    mix = pd.DataFrame(rows)
    oracle = load_table("D_train")["origin_type"].value_counts(normalize=True)
    mix["oracle"] = mix.origin_type.map(oracle).fillna(0.0)
    return mix


def tvd(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def cmd_mix(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mix = compute_mixes()
    mix.to_csv(OUT_DIR / "target_mix.csv", index=False)
    summary = {}
    for sc, g in mix.groupby("scenario"):
        summary[sc] = {
            "tvd_source_vs_oracle": tvd(g.source_share, g.oracle),
            "tvd_stations_vs_oracle": tvd(g.stations, g.oracle),
            "tvd_rate_vs_oracle": tvd(g.rate, g.oracle),
            "stations": dict(zip(g.origin_type.astype(str), g.stations.round(4))),
            "rate": dict(zip(g.origin_type.astype(str), g.rate.round(4))),
        }
    summary["oracle"] = dict(zip(mix[mix.scenario == "C"].origin_type.astype(str),
                                 mix[mix.scenario == "C"].oracle.round(4)))
    (OUT_DIR / "target_mix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    show = mix.copy()
    for c in ["source_share", "stations", "rate", "oracle"]:
        show[c] = (100 * show[c]).round(1)
    print("Μερίδιο αφετηριών ανά τύπο σταθμού (%):")
    print(show.to_string(index=False))
    print("\nTVD έναντι του πραγματικού μείγματος του London train (oracle):")
    for sc in SCENARIO_CITIES:
        s = summary[sc]
        print(f"  {sc}: χωρίς συνθήκη {s['tvd_source_vs_oracle']:.3f} | "
              f"stations {s['tvd_stations_vs_oracle']:.3f} | rate {s['tvd_rate_vs_oracle']:.3f}")
    print(f"\nSaved {OUT_DIR / 'target_mix.csv'} and target_mix.json")


# ---------------------------------------------------------------------------
# 2. Pool από αποθηκευμένο μοντέλο SDV
# ---------------------------------------------------------------------------
def to_schema_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
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


def cmd_pool(args) -> None:
    from sdv.utils import load_synthesizer
    schema = load_schema()
    cols = list(schema["categorical"]) + schema["continuous"]
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for path in args.model:
        tag = Path(path).stem
        out = POOL_DIR / f"{tag}_pool.parquet"
        if out.exists():
            print(f"{out.name}: υπάρχει ήδη, παραλείπεται")
            continue
        synth = load_synthesizer(path)
        syn = synth.sample(num_rows=args.rows)
        syn = syn["trips"] if isinstance(syn, dict) else syn
        syn = add_derived(to_schema_types(syn[cols], schema))
        syn.to_parquet(out, index=False)
        print(f"{tag}: {len(syn):,} rows -> {out}")


# ---------------------------------------------------------------------------
# 3. Υποδειγματισμός στο μείγμα-στόχο
# ---------------------------------------------------------------------------
def counts_for(shares: np.ndarray, n: int) -> np.ndarray:
    """Ακέραια πλήθη που αθροίζουν σε n (μέθοδος μεγαλύτερου υπολοίπου)."""
    raw = shares / shares.sum() * n
    base = np.floor(raw).astype(int)
    base[np.argsort(raw - base)[::-1][: n - base.sum()]] += 1
    return base


def resample(pool: pd.DataFrame, target: dict[int, int], seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    parts, report = [], {}
    for t, m in target.items():
        idx = np.flatnonzero(pool["origin_type"].to_numpy() == t)
        if m == 0:
            continue
        if len(idx) >= m:
            pick = rng.choice(idx, m, replace=False)
            dup = 0
        elif len(idx) == 0:
            raise ValueError(f"Το pool δεν έχει καμία διαδρομή τύπου {t}: μεγαλύτερο pool")
        else:
            pick = np.concatenate([idx, rng.choice(idx, m - len(idx), replace=True)])
            dup = m - len(idx)
        parts.append(pick)
        report[str(t)] = {"target": int(m), "available": int(len(idx)), "duplicates": int(dup)}
    out = pool.iloc[rng.permutation(np.concatenate(parts))].reset_index(drop=True)
    return out, report


def scenario_of(tag: str) -> str:
    m = TAG_RE.search(tag)
    if not m or m.group("sc") == "D":
        raise ValueError(f"{tag}: δεν βρέθηκε σενάριο A, B ή C στο όνομα")
    return m.group("sc")


def cmd_resample(args) -> None:
    mix_path = OUT_DIR / "target_mix.csv"
    if not mix_path.exists():
        raise SystemExit("Τρέξε πρώτα: python src/11_conditional.py mix")
    mix = pd.read_csv(mix_path)
    paths = sorted({p for pat in args.pool for p in (glob.glob(pat) or [pat])})
    for path in paths:
        tag = Path(path).stem.removesuffix("_pool")
        sc = scenario_of(tag)
        pool = pd.read_parquet(path)
        g = mix[mix.scenario == sc].sort_values("origin_type")
        for name in args.mix:
            out = SYNTHETIC_DIR / f"{tag}_cond-{name}.parquet"
            if out.exists() and not args.overwrite:
                print(f"{out.name}: υπάρχει ήδη, παραλείπεται")
                continue
            target = dict(zip(g.origin_type.astype(int), counts_for(g[name].to_numpy(), args.rows)))
            syn, rep = resample(pool, target, args.seed)
            syn.to_parquet(out, index=False)
            dup_share = sum(r["duplicates"] for r in rep.values()) / len(syn)
            (OUT_DIR / f"resample_{out.stem}.json").write_text(json.dumps(
                {"pool": Path(path).name, "pool_rows": len(pool), "mix": name, "scenario": sc,
                 "rows": len(syn), "duplicate_share": dup_share, "by_type": rep}, indent=2),
                encoding="utf-8")
            flag = "  <- λίγες γραμμές, μεγαλύτερο pool" if dup_share > 0 else ""
            print(f"{out.name}: {len(syn):,} rows, διπλότυπα {100 * dup_share:.2f}%{flag}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mix")
    p = sub.add_parser("pool")
    p.add_argument("--model", nargs="+", required=True, help="models/*.pkl του SDV")
    p.add_argument("--rows", type=int, default=1_000_000)
    r = sub.add_parser("resample")
    r.add_argument("--pool", nargs="+", required=True, help="αρχεία ή μοτίβα pool")
    r.add_argument("--mix", nargs="+", choices=["stations", "rate", "oracle"],
                   default=["stations", "rate", "oracle"])
    r.add_argument("--rows", type=int, default=N_OUT)
    r.add_argument("--seed", type=int, default=SAMPLING_SEED)
    r.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    {"mix": cmd_mix, "pool": cmd_pool, "resample": cmd_resample}[args.cmd](args)


if __name__ == "__main__":
    main()
