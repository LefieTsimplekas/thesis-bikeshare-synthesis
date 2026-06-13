"""
Experiments A/B/C (Cross-city): Train TabDDPM on US data (chicago / nyc /
chicago_nyc), evaluate fidelity + TSTR against the locked london_test.
TRTR is read from the oracle report — no retraining.

Usage:
  python src/10_train_tabddpm_crosscity.py --regime chicago
  python src/10_train_tabddpm_crosscity.py --regime nyc --n-iter 100 --n-synth 5000
  python src/10_train_tabddpm_crosscity.py --regime chicago_nyc \
      --drive-save "/content/drive/MyDrive/ColabNotebooks/thesis-bikeshare-synthesis"
"""
from __future__ import annotations

import argparse
import json
import shutil
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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LONDON_TEST_PATH = PROCESSED_DIR / "london_test.parquet"

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
    regime: str,
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

    loader = GenericDataLoader(train_df)
    model  = _get_model(Plugins, n_iter, batch_size)

    print(f"Training TabDDPM on {regime} ({len(train_df):,} rows)...")
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


# ── utility: TSTR only ───────────────────────────────────────────────────────
def _encode_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df.drop(columns=[TARGET]))


def utility_tstr(real_test: pd.DataFrame, synth: pd.DataFrame) -> dict:
    X_s  = _encode_for_ml(synth)
    X_te = _encode_for_ml(real_test)

    all_cols = X_s.columns.union(X_te.columns)
    X_s  = X_s.reindex(columns=all_cols,  fill_value=0)
    X_te = X_te.reindex(columns=all_cols, fill_value=0)

    y_s  = synth[TARGET].astype(float)
    y_te = real_test[TARGET].astype(float)

    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_s, y_s)
    pred = rf.predict(X_te)

    return {
        "r2":  round(r2_score(y_te, pred), 4),
        "mae": round(mean_absolute_error(y_te, pred), 4),
    }


# ── drive auto-save ───────────────────────────────────────────────────────────
def save_to_drive(regime: str, drive_root: str) -> None:
    drive = Path(drive_root)

    synth_src = SYNTHETIC_DIR / f"{regime}_synthetic.parquet"
    synth_dst = drive / "data" / "synthetic" / f"{regime}_synthetic.parquet"
    synth_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(synth_src, synth_dst)
    print(f"  Saved synthetic → {synth_dst}")

    artifacts_src = ARTIFACTS_DIR / f"crosscity_{regime}"
    artifacts_dst = drive / "artifacts" / f"crosscity_{regime}"
    if artifacts_dst.exists():
        shutil.rmtree(artifacts_dst)
    shutil.copytree(artifacts_src, artifacts_dst)
    print(f"  Saved artifacts → {artifacts_dst}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-city TabDDPM experiment (US → London)"
    )
    parser.add_argument(
        "--regime", required=True,
        choices=["chicago", "nyc", "chicago_nyc"],
        help="US training regime",
    )
    parser.add_argument("--n-iter",     type=int, default=200)
    parser.add_argument("--n-synth",    type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument(
        "--oracle-report",
        default="artifacts/oracle_london/oracle_report.json",
        help="Path to oracle_report.json (source of TRTR)",
    )
    parser.add_argument(
        "--drive-save",
        default=None,
        metavar="DRIVE_PATH",
        help="Optional Drive root for auto-save after run",
    )
    args = parser.parse_args()
    regime = args.regime

    oracle_path = Path(args.oracle_report)
    if not oracle_path.is_absolute():
        oracle_path = PROJECT_ROOT / oracle_path
    synth_path  = SYNTHETIC_DIR / f"{regime}_synthetic.parquet"
    report_dir  = ARTIFACTS_DIR / f"crosscity_{regime}"
    report_path = report_dir / "crosscity_report.json"

    # ── TRTR from oracle ──────────────────────────────────────────────────────
    print(f"Loading oracle report: {oracle_path}")
    with oracle_path.open(encoding="utf-8") as f:
        oracle = json.load(f)
    trtr_r2  = oracle["utility"]["trtr"]["r2"]
    trtr_mae = oracle["utility"]["trtr"]["mae"]
    print(f"TRTR loaded from oracle: R²={trtr_r2}  MAE={trtr_mae}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / f"{regime}_sampled_with_weather.parquet"
    print(f"\nLoading {regime} training data: {train_path}")
    train_df = build_model_table(pd.read_parquet(train_path))
    print(f"Loading london_test: {LONDON_TEST_PATH}")
    test_df  = build_model_table(pd.read_parquet(LONDON_TEST_PATH))
    print(f"US train ({regime}): {len(train_df):,}  |  London test: {len(test_df):,}")

    # ── Train + generate ──────────────────────────────────────────────────────
    synth_raw = train_and_sample(train_df, regime, args.n_synth, args.n_iter, args.batch_size)
    synth_df  = build_model_table(synth_raw)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    synth_df.to_parquet(synth_path, index=False)
    print(f"\nSaved synthetic: {synth_path}  ({len(synth_df):,} rows)")

    # ── Fidelity ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Fidelity report  (london_test vs synthetic)")
    print(f"{'=' * 60}")
    fid = fidelity_report(test_df, synth_df)
    print(fid.to_string(index=False))
    mean_ks  = fid.loc[fid["type"] == "continuous",  "value"].mean()
    mean_tvd = fid.loc[fid["type"] == "categorical", "value"].mean()
    print(f"\nMean KS:  {mean_ks:.4f}")
    print(f"Mean TVD: {mean_tvd:.4f}")

    # ── TSTR ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Utility: TRTR (oracle) vs TSTR  (target = trip_duration)")
    print(f"{'=' * 60}")
    print("Computing TSTR (RF on synthetic → predict london_test)...")
    tstr = utility_tstr(test_df, synth_df)

    transfer_gap = round(trtr_r2 - tstr["r2"], 4)
    if transfer_gap < 0.05:
        gap_label = "Excellent transfer"
    elif transfer_gap < 0.10:
        gap_label = "Good transfer"
    elif transfer_gap < 0.20:
        gap_label = "Moderate transfer"
    else:
        gap_label = "Large transfer gap"

    print(f"TRTR  R²={trtr_r2:.4f}  MAE={trtr_mae:.4f}  (from oracle report)")
    print(f"TSTR  R²={tstr['r2']:.4f}  MAE={tstr['mae']:.4f}")
    print(f"Transfer gap (TRTR R² − TSTR R²): {transfer_gap:.4f}  → {gap_label}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": f"crosscity_{regime}",
        "regime": regime,
        "model": "ddpm",
        "train_rows": len(train_df),
        "n_synth": len(synth_df),
        "n_iter": args.n_iter,
        "batch_size": args.batch_size,
        "seed": SEED,
        "fidelity": fid.to_dict(orient="records"),
        "mean_ks":  round(mean_ks,  4),
        "mean_tvd": round(mean_tvd, 4),
        "utility": {
            "trtr": {"r2": trtr_r2, "mae": trtr_mae},
            "tstr": tstr,
            "transfer_gap_r2": transfer_gap,
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # ── Drive auto-save ───────────────────────────────────────────────────────
    if args.drive_save:
        print(f"\nAuto-saving to Drive: {args.drive_save}")
        save_to_drive(regime, args.drive_save)
        print("Drive save complete.")


if __name__ == "__main__":
    main()

# Colab batch cell:
# DRIVE = "/content/drive/MyDrive/ColabNotebooks/thesis-bikeshare-synthesis"
# for regime in ["chicago", "nyc", "chicago_nyc"]:
#     !python src/10_train_tabddpm_crosscity.py \
#         --regime {regime} \
#         --drive-save "{DRIVE}"
