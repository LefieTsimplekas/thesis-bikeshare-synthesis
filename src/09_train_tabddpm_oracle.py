"""
Experiment D (Oracle): Train TabDDPM on raw London 80%, generate synthetic,
evaluate fidelity + utility vs. locked london_test (20%).

Usage:
  python src/09_train_tabddpm_oracle.py                          # full run
  python src/09_train_tabddpm_oracle.py --train-rows 100000 \
      --n-iter 800 --n-synth 50000                              # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_model_schema import (
    MODEL_CATEGORICAL_COLS,
    MODEL_CONTINUOUS_COLS,
    build_model_table,
)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "oracle_london"

TRAIN_PATH  = PROCESSED_DIR / "london_train.parquet"
TEST_PATH   = PROCESSED_DIR / "london_test.parquet"
SYNTH_PATH  = SYNTHETIC_DIR / "london_oracle_synthetic.parquet"
REPORT_PATH = ARTIFACTS_DIR / "oracle_report.json"

TARGET = "trip_duration"
SEED   = 42


# ── synthcity guard ──────────────────────────────────────────────────────────
def _load_synthcity():
    try:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
        return Plugins, GenericDataLoader
    except ModuleNotFoundError:
        raise SystemExit("synthcity not installed. Run: pip install synthcity")


def _get_model(Plugins, n_iter: int, batch_size: int):
    available = Plugins().list()
    if "ddpm" not in available:
        print("Available generic plugins:", available)
        raise SystemExit("Plugin 'ddpm' not found in this synthcity installation.")
    try:
        return Plugins().get("ddpm", n_iter=n_iter, batch_size=batch_size)
    except Exception:
        print("Note: this synthcity version ignores n_iter/batch_size — using defaults.")
        return Plugins().get("ddpm")


# ── training + generation ────────────────────────────────────────────────────
def train_and_sample(
    train_df: pd.DataFrame,
    n_synth: int,
    n_iter: int,
    batch_size: int,
) -> pd.DataFrame:
    Plugins, GenericDataLoader = _load_synthcity()

    try:
        from synthcity.utils.reproducibility import enable_reproducible_results
        enable_reproducible_results(SEED)
    except Exception:
        pass

    # Unconditional joint generation — trip_duration is one of the 16 columns,
    # not a special label. Using target_column would contaminate the TSTR evaluation.
    loader = GenericDataLoader(train_df)
    model  = _get_model(Plugins, n_iter, batch_size)

    print(f"Fitting TabDDPM on {len(train_df):,} rows, n_iter={n_iter}...")
    model.fit(loader)

    print(f"Generating {n_synth:,} synthetic rows...")
    synth_df = model.generate(count=n_synth).dataframe()
    return synth_df


# ── fidelity ─────────────────────────────────────────────────────────────────
def fidelity_report(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MODEL_CONTINUOUS_COLS:
        stat, _ = ks_2samp(
            real[col].dropna().astype(float),
            synth[col].dropna().astype(float),
        )
        rows.append({"column": col, "type": "continuous", "metric": "ks_statistic", "value": stat})

    for col in MODEL_CATEGORICAL_COLS:
        real_p  = real[col].astype(str).value_counts(normalize=True)
        synth_p = synth[col].astype(str).value_counts(normalize=True)
        all_cats = real_p.index.union(synth_p.index)
        tvd = 0.5 * (
            real_p.reindex(all_cats, fill_value=0)
            - synth_p.reindex(all_cats, fill_value=0)
        ).abs().sum()
        rows.append({"column": col, "type": "categorical", "metric": "tvd", "value": tvd})

    return pd.DataFrame(rows)


# ── utility: TSTR / TRTR ─────────────────────────────────────────────────────
def utility_tstr_trtr(
    real_train: pd.DataFrame,
    real_test:  pd.DataFrame,
    synth:      pd.DataFrame,
) -> dict:
    def prep_X(df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df.drop(columns=[TARGET]))

    X_rt = prep_X(real_train)
    X_s  = prep_X(synth)
    X_te = prep_X(real_test)

    all_cols = X_rt.columns.union(X_s.columns).union(X_te.columns)
    X_rt = X_rt.reindex(columns=all_cols, fill_value=0)
    X_s  = X_s.reindex(columns=all_cols,  fill_value=0)
    X_te = X_te.reindex(columns=all_cols, fill_value=0)

    y_rt = real_train[TARGET].astype(float)
    y_s  = synth[TARGET].astype(float)
    y_te = real_test[TARGET].astype(float)

    rf_trtr = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_trtr.fit(X_rt, y_rt)
    pred_trtr = rf_trtr.predict(X_te)

    rf_tstr = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_tstr.fit(X_s, y_s)
    pred_tstr = rf_tstr.predict(X_te)

    return {
        "trtr": {
            "r2":  round(r2_score(y_te, pred_trtr), 4),
            "mae": round(mean_absolute_error(y_te, pred_trtr), 4),
        },
        "tstr": {
            "r2":  round(r2_score(y_te, pred_tstr), 4),
            "mae": round(mean_absolute_error(y_te, pred_tstr), 4),
        },
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle TabDDPM experiment (London->London)")
    parser.add_argument("--train-rows", type=int, default=None,
                        help="Subsample train for smoke test")
    parser.add_argument("--n-synth",    type=int, default=50_000)
    parser.add_argument("--n-iter",     type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    args = parser.parse_args()

    print("Loading london_train / london_test...")
    train_df = build_model_table(pd.read_parquet(TRAIN_PATH))
    test_df  = build_model_table(pd.read_parquet(TEST_PATH))

    if args.train_rows:
        train_df = train_df.sample(n=args.train_rows, random_state=SEED).reset_index(drop=True)
        print(f"Subsampled train to {len(train_df):,} rows.")

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    synth_raw = train_and_sample(train_df, args.n_synth, args.n_iter, args.batch_size)
    synth_df  = build_model_table(synth_raw)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    synth_df.to_parquet(SYNTH_PATH, index=False)
    print(f"\nSaved synthetic: {SYNTH_PATH}  ({len(synth_df):,} rows)")

    print(f"\n{'=' * 60}")
    print("Fidelity report")
    print(f"{'=' * 60}")
    fid = fidelity_report(test_df, synth_df)
    print(fid.to_string(index=False))
    mean_ks  = fid.loc[fid["type"] == "continuous",  "value"].mean()
    mean_tvd = fid.loc[fid["type"] == "categorical", "value"].mean()
    print(f"\nMean KS:  {mean_ks:.4f}")
    print(f"Mean TVD: {mean_tvd:.4f}")

    print(f"\n{'=' * 60}")
    print("Utility: TSTR vs TRTR  (target = trip_duration)")
    print(f"{'=' * 60}")
    utility = utility_tstr_trtr(train_df, test_df, synth_df)
    print(f"TRTR  R²={utility['trtr']['r2']:.4f}  MAE={utility['trtr']['mae']:.4f}")
    print(f"TSTR  R²={utility['tstr']['r2']:.4f}  MAE={utility['tstr']['mae']:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "oracle_london",
        "model": "ddpm",
        "train_rows": len(train_df),
        "n_synth": len(synth_df),
        "n_iter": args.n_iter,
        "batch_size": args.batch_size,
        "seed": SEED,
        "fidelity": fid.to_dict(orient="records"),
        "mean_ks":  round(mean_ks,  4),
        "mean_tvd": round(mean_tvd, 4),
        "utility": utility,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
