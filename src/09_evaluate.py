"""
09_evaluate.py

Αξιολόγηση συνόλων δεδομένων απέναντι στο κλειδωμένο London test (§6).
Δουλεύει για κάθε σύνολο με τις 8 στήλες γένεσης: συνθετικά (07, Επίπεδο 2) και
πραγματικά σύνολα εκπαίδευσης ως baselines (R-A, R-B, R-C, R-D, χωρίς μοντέλο).

Μετρικές (για κάθε σύνολο S, απέναντι στο London test T):
  Fidelity ανά στήλη
    κατηγορικές : TVD, JSD
    συνεχείς    : KS (στατιστικό, όχι p-value), Wasserstein-1 κανονικοποιημένο με
                  την τυπική απόκλιση του T, JSD σε 50 bins ποσοστημορίων του T
  Fidelity ζευγών
    CorrelationSimilarity (συνεχείς: 1 - |ρS - ρT| / 2, Spearman)
    ContingencySimilarity (κατηγορικές: 1 - TVD του πίνακα συνάφειας)
    Διαφορά πίνακα συσχετίσεων μικτών τύπων (Spearman / Cramér's V / correlation
    ratio): νόρμα Frobenius και μέση απόλυτη διαφορά
  Detection    ταξινομητής πραγματικά (T) έναντι S, ίσα πλήθη, 5-fold CV, AUC
               (0,5 = δεν ξεχωρίζουν). Δύο εκδοχές: με όλες τις στήλες, και χωρίς
               month και day_of_week (_nocal). Το test έχει 73 ημέρες, οπότε ο
               συνδυασμός μήνα×ημέρας του είναι αραιός και ξεχωρίζει ακόμη και από
               πραγματικά δεδομένα του Λονδίνου (R-D). Τυπώνεται και το TVD μήνα×ημέρας.
  Utility      TSTR με XGBoost, εκπαίδευση σε S, έλεγχος σε T, τρεις εργασίες:
               1. διάρκεια (παλινδρόμηση): MAE, R², με και χωρίς distance_km
               2. τύπος ποδηλάτου (ταξινόμηση e-bike/classic): AUC
               3. απόσταση (παλινδρόμηση, μόνο μη κυκλικές, χωρίς duration_min): MAE, R²
               Τα πραγματικά baselines υποδειγματοληπτούνται στο ίδιο πλήθος με το test.
  Diversity    κάλυψη κελιών ώρα×ημέρα, μήνα, τύπου σταθμού×τύπου ποδηλάτου
  Plausibility ταχύτητα = distance_km / (duration_min / 60), μη κυκλικές:
               % πάνω από 30 km/h (classic) / 35 km/h (e-bike), Wasserstein ταχύτητας

Χρήση:
  python src/09_evaluate.py --synthetic data/synthetic/tabddpm_D.parquet ...
  python src/09_evaluate.py --real A B C D
  (τα δύο μαζί επιτρέπονται)

Έξοδοι:
  artifacts/evaluation/metrics.csv       μία γραμμή ανά σύνολο (ενημερώνεται, δεν σβήνεται)
  artifacts/evaluation/columns_{name}.csv  μετρικές ανά στήλη
  artifacts/evaluation/by_origin_{name}.csv  διάρκεια/απόσταση ανά τύπο σταθμού
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from config import SAMPLING_SEED
from model_io import load_schema, load_table, add_derived

EVAL_DIR = Path("artifacts/evaluation")
SPEED_LIMIT = {"classic_bike": 30.0, "electric_bike": 35.0}
N_BINS = 50
TSTR_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                   colsample_bytree=0.9, tree_method="hist", enable_categorical=True)
DET_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=6,
                  tree_method="hist", enable_categorical=True)


# ---------------------------------------------------------------------------
# Βοηθητικά
# ---------------------------------------------------------------------------
def shares(s: pd.Series, cats) -> np.ndarray:
    return s.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()


def binned(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    h = np.histogram(np.clip(s, edges[0], edges[-1]), bins=edges)[0].astype(float)
    return h / h.sum()


def as_features(df: pd.DataFrame, schema: dict, cols) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if c in schema["categorical"]:
            X[c] = pd.Categorical(X[c], categories=schema["categorical"][c])
    return X


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b).to_numpy().astype(float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    n = t.sum()
    e = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = ((t - e) ** 2 / np.where(e > 0, e, 1)).sum()
    return float(np.sqrt(chi2 / n / (min(t.shape) - 1)))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    g = num.groupby(cat)
    ss_between = (g.count() * (g.mean() - num.mean()) ** 2).sum()
    ss_total = ((num - num.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def association_matrix(df: pd.DataFrame, cat_cols, num_cols) -> pd.DataFrame:
    cols = cat_cols + num_cols
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        if a in num_cols and b in num_cols:
            v = abs(spearmanr(df[a], df[b]).statistic)
        elif a in cat_cols and b in cat_cols:
            v = cramers_v(df[a], df[b])
        else:
            c, n = (a, b) if a in cat_cols else (b, a)
            v = correlation_ratio(df[c], df[n])
        M.loc[a, b] = M.loc[b, a] = v
    return M


def tvd_joint(S: pd.DataFrame, T: pd.DataFrame, cols: list) -> float:
    p = S.groupby(cols).size() / len(S)
    q = T.groupby(cols).size() / len(T)
    p, q = p.align(q, fill_value=0)
    return float(0.5 * (p - q).abs().sum())


def speed(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.round_trip == 0) & (df.duration_min > 0)]
    return d.assign(speed_kmh=d.distance_km / (d.duration_min / 60))


# ---------------------------------------------------------------------------
# Μετρικές
# ---------------------------------------------------------------------------
def fidelity(S: pd.DataFrame, T: pd.DataFrame, schema: dict) -> tuple[dict, pd.DataFrame]:
    cat_cols, num_cols = list(schema["categorical"]), schema["continuous"]
    rows = []
    for c in cat_cols:
        p, q = shares(S[c], schema["categorical"][c]), shares(T[c], schema["categorical"][c])
        rows.append({"column": c, "type": "categorical",
                     "TVD": 0.5 * np.abs(p - q).sum(), "JSD": jensenshannon(p, q, base=2) ** 2})
    for c in num_cols:
        edges = np.unique(T[c].quantile(np.linspace(0, 1, N_BINS + 1)).to_numpy())
        p, q = binned(S[c], edges), binned(T[c], edges)
        rows.append({"column": c, "type": "continuous",
                     "KS": ks_2samp(S[c], T[c]).statistic,
                     "W1_norm": wasserstein_distance(S[c], T[c]) / T[c].std(),
                     "JSD": jensenshannon(p, q, base=2) ** 2})
    cols = pd.DataFrame(rows)

    # Ζεύγη (ορισμοί όπως στο SDMetrics)
    corr_sim = [1 - abs(spearmanr(S[a], S[b]).statistic - spearmanr(T[a], T[b]).statistic) / 2
                for a, b in itertools.combinations(num_cols, 2)]
    cont_sim = []
    for a, b in itertools.combinations(cat_cols, 2):
        ps = pd.crosstab(S[a], S[b], normalize=True)
        pt = pd.crosstab(T[a], T[b], normalize=True)
        ps, pt = ps.align(pt, fill_value=0)
        ps, pt = ps.fillna(0), pt.fillna(0)
        cont_sim.append(1 - 0.5 * np.abs(ps.to_numpy() - pt.to_numpy()).sum())

    MS, MT = association_matrix(S, cat_cols, num_cols), association_matrix(T, cat_cols, num_cols)
    D = (MS - MT).to_numpy()
    off = ~np.eye(len(D), dtype=bool)

    summary = {
        "mean_TVD_cat": cols.loc[cols.type == "categorical", "TVD"].mean(),
        "mean_JSD": cols["JSD"].mean(),
        "mean_KS_num": cols.loc[cols.type == "continuous", "KS"].mean(),
        "mean_W1norm_num": cols.loc[cols.type == "continuous", "W1_norm"].mean(),
        "CorrelationSimilarity": float(np.mean(corr_sim)),
        "ContingencySimilarity": float(np.mean(cont_sim)),
        "assoc_frobenius": float(np.linalg.norm(D[off])),
        "assoc_mean_abs_diff": float(np.abs(D[off]).mean()),
    }
    for _, r in cols.iterrows():
        key = "TVD" if r["type"] == "categorical" else "KS"
        summary[f"{key}_{r['column']}"] = r[key]
    return summary, cols


def detection(S: pd.DataFrame, T: pd.DataFrame, schema: dict, seed: int,
              drop: tuple = (), tag: str = "") -> dict:
    cols = [c for c in list(schema["categorical"]) + schema["continuous"] if c not in drop]
    n = min(len(S), len(T))
    X = pd.concat([S.sample(n, random_state=seed)[cols], T.sample(n, random_state=seed)[cols]],
                  ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    X = as_features(X, schema, cols)
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = XGBClassifier(**DET_PARAMS, random_state=seed, n_jobs=-1)
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return {f"detection_AUC{tag}": float(np.mean(aucs)),
            f"detection_AUC{tag}_std": float(np.std(aucs))}


def tstr(S: pd.DataFrame, T: pd.DataFrame, schema: dict, seed: int) -> dict:
    out = {}
    allc = list(schema["categorical"]) + schema["continuous"]

    # 1. Διάρκεια
    base = [c for c in allc if c != "duration_min"]
    for tag, feats in [("", base), ("_nodist", [c for c in base if c != "distance_km"])]:
        m = XGBRegressor(**TSTR_PARAMS, random_state=seed, n_jobs=-1)
        m.fit(as_features(S, schema, feats), S["duration_min"])
        pred = m.predict(as_features(T, schema, feats))
        out[f"TSTR_MAE{tag}"] = float(mean_absolute_error(T["duration_min"], pred))
        out[f"TSTR_R2{tag}"] = float(r2_score(T["duration_min"], pred))

    # 2. Τύπος ποδηλάτου (AUC: ανεξάρτητο από το ποσοστό e-bikes κάθε συνόλου)
    feats = [c for c in allc if c != "rideable_type"]
    yS = (S["rideable_type"] == "electric_bike").astype(int)
    yT = (T["rideable_type"] == "electric_bike").astype(int)
    if yS.nunique() == 2:
        m = XGBClassifier(**TSTR_PARAMS, random_state=seed, n_jobs=-1)
        m.fit(as_features(S, schema, feats), yS)
        out["TSTR_bike_AUC"] = float(roc_auc_score(yT, m.predict_proba(as_features(T, schema, feats))[:, 1]))
    else:
        out["TSTR_bike_AUC"] = np.nan

    # 3. Απόσταση (μόνο μη κυκλικές, χωρίς διάρκεια ως είσοδο)
    feats = [c for c in allc if c not in ("distance_km", "duration_min", "round_trip")]
    s, t = S[S.round_trip == 0], T[T.round_trip == 0]
    m = XGBRegressor(**TSTR_PARAMS, random_state=seed, n_jobs=-1)
    m.fit(as_features(s, schema, feats), s["distance_km"])
    pred = m.predict(as_features(t, schema, feats))
    out["TSTR_dist_MAE"] = float(mean_absolute_error(t["distance_km"], pred))
    out["TSTR_dist_R2"] = float(r2_score(t["distance_km"], pred))
    return out


def diversity(S: pd.DataFrame, T: pd.DataFrame) -> dict:
    def cover(cols):
        t = set(map(tuple, T[cols].drop_duplicates().to_numpy()))
        s = set(map(tuple, S[cols].drop_duplicates().to_numpy()))
        return len(t & s) / len(t)
    return {"coverage_hour_x_dow": cover(["hour", "day_of_week"]),
            "coverage_month": cover(["month"]),
            "coverage_origin_x_bike": cover(["origin_type", "rideable_type"])}


def plausibility(S: pd.DataFrame, T: pd.DataFrame) -> dict:
    vs, vt = speed(S), speed(T)
    limit = vs.rideable_type.map(SPEED_LIMIT)
    return {"pct_impossible_speed": float(100 * (vs.speed_kmh > limit).mean()),
            "median_speed_kmh": float(vs.speed_kmh.median()),
            "W1_speed": float(wasserstein_distance(vs.speed_kmh, vt.speed_kmh)),
            "pct_round_trip": float(100 * S.round_trip.mean()),
            "pct_ebike": float(100 * (S.rideable_type == "electric_bike").mean()),
            "median_duration_min": float(S.duration_min.median()),
            "median_distance_km": float(S.distance_km.median())}


def by_origin(S: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(T.origin_type.unique()):
        s, t = S[S.origin_type == k], T[T.origin_type == k]
        r = {"origin_type": k, "share_S": len(s) / len(S), "share_T": len(t) / len(T)}
        for c in ["duration_min", "distance_km"]:
            r[f"W1_{c}"] = wasserstein_distance(s[c], t[c]) if len(s) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def evaluate(name: str, S: pd.DataFrame, T: pd.DataFrame, schema: dict, seed: int) -> dict:
    print(f"\n===== {name} ({len(S):,} rows) =====")
    S = add_derived(S)
    fid, cols = fidelity(S, T, schema)
    res = {"dataset": name, "rows": len(S), **fid}
    res["TVD_month_x_dow"] = tvd_joint(S, T, ["month", "day_of_week"])
    print(f"  fidelity: mean TVD {fid['mean_TVD_cat']:.4f} | mean KS {fid['mean_KS_num']:.4f} | "
          f"Corr.Sim {fid['CorrelationSimilarity']:.4f} | Cont.Sim {fid['ContingencySimilarity']:.4f} | "
          f"TVD month x day {res['TVD_month_x_dow']:.4f}")
    res.update(detection(S, T, schema, seed))
    res.update(detection(S, T, schema, seed, drop=("month", "day_of_week"), tag="_nocal"))
    print(f"  detection AUC {res['detection_AUC']:.4f} | without month/day {res['detection_AUC_nocal']:.4f}")
    res.update(tstr(S, T, schema, seed))
    print(f"  TSTR duration: MAE {res['TSTR_MAE']:.3f} (R2 {res['TSTR_R2']:.3f}) | "
          f"without distance MAE {res['TSTR_MAE_nodist']:.3f} (R2 {res['TSTR_R2_nodist']:.3f})")
    print(f"  TSTR bike type: AUC {res['TSTR_bike_AUC']:.4f} | "
          f"TSTR distance: MAE {res['TSTR_dist_MAE']:.3f} km (R2 {res['TSTR_dist_R2']:.3f})")
    res.update(diversity(S, T))
    res.update(plausibility(S, T))
    print(f"  impossible speed {res['pct_impossible_speed']:.2f}% | e-bike {res['pct_ebike']:.1f}% | "
          f"round trip {res['pct_round_trip']:.1f}% | coverage hour x day {res['coverage_hour_x_dow']:.3f}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cols.to_csv(EVAL_DIR / f"columns_{name}.csv", index=False)
    by_origin(S, T).to_csv(EVAL_DIR / f"by_origin_{name}.csv", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", nargs="*", default=[], help="αρχεία parquet συνθετικών")
    ap.add_argument("--real", nargs="*", default=[], choices=["A", "B", "C", "D"],
                    help="πραγματικά baselines R-* από το {σενάριο}_train")
    ap.add_argument("--seed", type=int, default=SAMPLING_SEED)
    args = ap.parse_args()

    schema = load_schema()
    T = add_derived(load_table("london_test"))
    print(f"London test: {len(T):,} rows")

    results = []
    for path in args.synthetic:
        S = pd.read_parquet(path)
        results.append(evaluate(Path(path).stem, S, T, schema, args.seed))
    for sc in args.real:
        R = load_table(f"{sc}_train").sample(n=len(T), random_state=args.seed)
        results.append(evaluate(f"R-{sc}", R, T, schema, args.seed))

    new = pd.DataFrame(results)
    out = EVAL_DIR / "metrics.csv"
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"\nSaved {out} ({len(new)} datasets)")


if __name__ == "__main__":
    main()
