# Cross-city synthetic bike-sharing data with diffusion models

> **Work in progress.** MSc thesis (Business Analytics & Data Science, University of Macedonia).
> This README is updated as the pipeline evolves; the thesis text itself is written in Greek.

**Research question.** Which combination of training regime and generative model produces the most
faithful and useful synthetic bike-sharing trips for an unseen European city, when the model is
trained only on American cities?

- **Source cities:** Chicago (Divvy) and New York (Citi Bike), all trips of 2025.
- **Evaluation city:** London (Santander Cycles), 2025. A day-level test set (every 5th day, 200,000 trips)
  is locked with a SHA-256 manifest and used for every experiment.
- **Level 1 – training regime (TabDDPM):** A = Chicago, B = NYC, C = Chicago + NYC (50/50),
  D = London train (oracle / upper bound). All four use the same size, N = 720,809 trips.
- **Level 2 – model comparison** on the best regime: TabDDPM, CTGAN, TVAE, GaussianCopula.
- **Baselines:** the real training data of each scenario without a model (R-A … R-D; R-D = TRTR).

## Generated columns

Every model generates the same 8 columns; `is_weekend` and `part_of_day` are derived afterwards.

| Column | Type | Values |
|---|---|---|
| `hour` | categorical | 0–23 |
| `day_of_week` | categorical | 0–6 (0 = Monday) |
| `month` | categorical | 1–12 |
| `rideable_type` | categorical | `classic_bike`, `electric_bike` |
| `origin_type` | categorical | 0–5, station context type from OpenStreetMap (k-means fitted on US stations only) |
| `round_trip` | categorical | 0 / 1 (same physical station) |
| `duration_min` | continuous | minutes |
| `distance_km` | continuous | shortest path on the OSM bike network; 0 for round trips |

## Repository layout

```
src/                    pipeline scripts (run from the repository root)
  config.py             paths and all constants (cleaning rules, sampling, OSM, TabDDPM)
  model_io.py           shared loading / encoding for all generative models
Notebooks/              EDA notebooks and colab_train_tabddpm.ipynb (GPU training)
artifacts/              small outputs kept in git: logs/, evaluation/, model_input/schema.json,
                        station_types/, network_distance/, data_quality/, training/
data/                   not in git (raw, processed, synthetic data); only the split manifests are tracked
models/                 trained models, not in git
```

`artifacts/logs/` holds the console output of every run (index in `artifacts/logs/INDEX.md`);
the numbers reported in the thesis come from these files.

## Pipeline

Run in this order from the repository root (`python src/<script>.py`).

| Step | Script | What it does | Runs on |
|---|---|---|---|
| 01 | `01_build_featured_data.py` | Read monthly files with DuckDB, keep station-to-station trips, time features | local |
| 01b | `01b_cleaning_checks.py` | Checks on the raw data that justify the cleaning rules | local |
| 02b, 02c | `02b_station_table.py`, `02c_attach_station_keys.py` | Physical stations (IDs within 25 m merged), station keys on trips | local |
| 02 | `02_eda_featured_data.py` | EDA of the featured data, impact of each rule | local |
| 03 | `03_filtering.py` | Duration 1–100 min, distance 50 m–12 km, speed ≤ 40 km/h | local |
| 04 | `04_stratified_sampling.py` | Proportional stratified sample of 1M trips (192 strata, hash with seed) | local |
| 04b | `04b_source_val_split.py` | Day-level train / validation split for Chicago and NYC | local |
| L1, L1b | `l1_lnd_dataset_build.py`, `l1b_london_raw_checks.py` | London: same rules, BikePoint matching, raw-data checks | local |
| L2 | `l2_lnd_split.py` | London day-level split, locked test set (manifest) | local |
| 05a | `05a_osm_station_profiles.py` | Five OSM features per station (500 m), bike network | local |
| 05b | `05b_station_types.py` | Station types (k = 6), city profile distances (MMD, Wasserstein) | local |
| 05c | `05c_network_distance.py` | Network distance per station pair (Dijkstra) and circuity | local |
| 06 | `06_prepare_model_input.py` | Model input tables for A–D and the London test, `schema.json` | local |
| 06b | `06b_data_quality_report.py` | Missing values in all pipeline files | local |
| 07 | `07_train_tabddpm.py` | Train TabDDPM per scenario and generate 200,000 trips | **Colab GPU** |
| 09 | `09_evaluate.py` | Fidelity, detection, TSTR (3 tasks), diversity, plausibility vs the London test | local |
| 10 | `10_level1_analysis.py` | Level 1 over seeds (unit = trained model): scenario means, Δ vs R-*, A/B/C contrasts and transfer gap vs D with hierarchical bootstrap and Welch CIs, Cliff's δ, variance split training/generation | local |

`96_build_weather_context.py` and `97_add_weather_to_final_dataset.py` (weather) are outside the main flow.

## Setup (local)

Python 3.11 in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

`requirements.txt` is the single list of dependencies. Pinned on purpose:
`numpy==1.26.4` and `torch==2.2.2` (required by synthcity 0.2.12), `xgboost==2.1.4` (synthcity needs < 3),
`geopandas==1.1.1` (1.2 needs numpy 2). The raw trip files go in `data/raw/{chicago,nyc,london}/`.

## Training on Google Colab (GPU)

TabDDPM runs at ~1.2 steps/s on a laptop CPU and ~12.5 steps/s on a Colab T4
(about 36 minutes per scenario: 27 min training, 9 min generation).

1. Put the 9 files of `data/model_input/*.parquet` and `artifacts/model_input/schema.json`
   in Google Drive under `My Drive/thesis/model_input/`.
2. Open [`Notebooks/colab_train_tabddpm.ipynb`](https://colab.research.google.com/github/LefieTsimplekas/thesis-bikeshare-synthesis/blob/main/Notebooks/colab_train_tabddpm.ipynb)
   and choose *Runtime → Change runtime type → T4 GPU*.
3. Run the cells in order. The notebook builds a separate Python 3.11 environment with `uv`
   (Colab ships Python 3.13, for which torch 2.2.2 is not available), mounts Drive and clones this repository.
4. Outputs are written to `My Drive/thesis/outputs/` (synthetic data, models, training logs).
5. Seeds (section 6): cell 6a shows what is already on Drive, cell 6b runs only what is missing
   (a saved model is reused with `--generate-only`; half-written files are redone) and then
   disconnects the runtime. After an interrupted session, rerun cells 4, 6, 7, 6a, 6b.

## Reproducibility

- Fixed seed (42) for sampling, k-means and training; the sampling hash makes samples independent of file order.
- The London test set is checked against its SHA-256 manifest before it is used.
- OpenStreetMap snapshot of 2026-11-07 (Geofabrik extracts and Overpass).
- Adjustments to synthcity's TabDDPM are documented in `src/07_train_tabddpm.py`
  (explicit categorical columns, shuffled batches, steps instead of epochs, generation without rejection filtering).

## Status

| Part | Status |
|---|---|
| Data preparation (01–06b, L1–L2) | done, logged |
| Real-data baselines R-A … R-D (09) | done |
| Level 1, TabDDPM, one seed (A, B, C, D) | trained and evaluated |
| Level 1 seeds (3 training × 3 generation) | running on Colab |
| Seeds and statistical analysis (10) | script ready, waits for the seeds |
| Level 2 (CTGAN, TVAE, GaussianCopula) | to do |
