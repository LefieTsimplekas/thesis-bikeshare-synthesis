"""
10_level1_analysis.py

Ανάλυση του Επιπέδου 1 πάνω στα seeds (§7). Διαβάζει το artifacts/evaluation/metrics.csv
του 09 και δεν ξανατρέχει καμία αξιολόγηση.

Σχεδιασμός seeds: ανά σενάριο (A, B, C, D) 3 training seeds × 3 generation seeds.
  Μονάδα ανάλυσης = το εκπαιδευμένο μοντέλο (training seed). Οι 3 γενέσεις του ίδιου
  μοντέλου δεν είναι ανεξάρτητες, οπότε δεν μετρούν ως 9 παρατηρήσεις (ψευδο-επανάληψη).
  Με n = 3 μοντέλα ανά σενάριο ένα Mann–Whitney δεν μπορεί να δώσει p < 0,10, γι' αυτό
  δεν γίνονται τυπικοί έλεγχοι υποθέσεων· αναφέρονται εκτιμήσεις με διαστήματα.

Τι υπολογίζεται, για κάθε μετρική:
  1. Μέση τιμή ανά σενάριο (μέσος όρος των μοντέλων, καθένα = μέσος των γενέσεών του)
     με 95% CI από ιεραρχικό bootstrap: επαναδειγματοληψία training seeds και, μέσα σε
     κάθε επιλεγμένο μοντέλο, generation seeds (B = 10.000).
  2. Αποσύνθεση διακύμανσης (μέθοδος ροπών, φωλιασμένος σχεδιασμός):
     σ²_train (μεταξύ μοντέλων) και σ²_gen (μεταξύ γενέσεων του ίδιου μοντέλου).
  3. Δ = σενάριο − R-σενάριο (πραγματικά δεδομένα εκπαίδευσης χωρίς μοντέλο): πόσο
     σφάλμα προσθέτει το TabDDPM. Το R-* είναι μία τιμή, άρα το CI προέρχεται μόνο
     από τα συνθετικά.
  4. Αντιθέσεις: A−B, A−C, B−C (καθεστώτα εκπαίδευσης) και A−D, B−D, C−D (transfer gap
     ως προς το oracle), με ιεραρχικό bootstrap CI της διαφοράς, Cliff's δ στα μοντέλα
     (περιγραφικό μέγεθος επίδρασης) και, ως έλεγχο ευαισθησίας, Welch t-CI στους μέσους
     των μοντέλων. Δίπλα δίνεται και το αντίστοιχο χάσμα των πραγματικών (R-X − R-Y).
     «Νικητής» δηλώνεται μόνο όταν και τα δύο διαστήματα αποκλείουν το 0.

Σημείωση: με 3 μοντέλα το bootstrap τείνει να στενεύει το διάστημα (η διακύμανση
μεταξύ μοντέλων υποεκτιμάται κατά (n−1)/n). Γι' αυτό δίνεται και το Welch t-CI, που
είναι πιο συντηρητικό· ένα αποτέλεσμα θεωρείται σταθερό όταν συμφωνούν και τα δύο.

Χρήση:
  python src/10_level1_analysis.py
  python src/10_level1_analysis.py --all-metrics      (όλες οι αριθμητικές στήλες)
  python src/10_level1_analysis.py --boot 20000

Έξοδοι (artifacts/evaluation/):
  level1_summary.csv     σενάριο × μετρική: μέσος, CI, sd μεταξύ μοντέλων / γενέσεων, Δ
  level1_contrasts.csv   αντιθέσεις A/B/C και transfer gap, CI, Cliff's δ, Welch
  level1_variance.csv    αποσύνθεση διακύμανσης ανά σενάριο και μετρική
Τα αρχεία της πρώτης εκτέλεσης χωρίς seed στο όνομα (tabddpm_A κ.λπ.) δεν μπαίνουν.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import SAMPLING_SEED

EVAL_DIR = Path("artifacts/evaluation")
SCENARIOS = ["A", "B", "C", "D"]
NAME_RE = re.compile(r"^tabddpm_(?P<sc>[ABCD])_s(?P<seed>\d+)_g(?P<gen>\d+)$")

# Κύριες μετρικές: (στήλη, ετικέτα, κατεύθυνση) — "low"/"high" = ποια τιμή είναι καλύτερη,
# None = καμία (περιγραφική μετρική, χωρίς σωστή κατεύθυνση).
METRICS = [
    ("detection_AUC_nocal", "Detection AUC (no calendar)", "low"),
    ("mean_TVD_cat", "Mean TVD (categorical)", "low"),
    ("mean_KS_num", "Mean KS (continuous)", "low"),
    ("mean_W1norm_num", "Mean W1 norm. (continuous)", "low"),
    ("ContingencySimilarity", "Contingency similarity", "high"),
    ("CorrelationSimilarity", "Correlation similarity", "high"),
    ("assoc_frobenius", "Association matrix diff. (Frobenius)", "low"),
    ("TSTR_R2", "TSTR duration R2", "high"),
    ("TSTR_MAE", "TSTR duration MAE (min)", "low"),
    ("TSTR_bike_AUC", "TSTR bike type AUC", "high"),
    ("TSTR_dist_MAE", "TSTR distance MAE (km)", "low"),
    ("W1_speed", "W1 speed (km/h)", "low"),
    ("pct_impossible_speed", "% impossible speed", None),
]
CONTRASTS = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")]


# ---------------------------------------------------------------------------
# Δεδομένα
# ---------------------------------------------------------------------------
def load_runs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(path)
    parsed = m["dataset"].str.extract(NAME_RE)
    runs = pd.concat([parsed, m], axis=1).dropna(subset=["sc"])
    runs = runs.astype({"seed": int, "gen": int})
    real = m[m["dataset"].str.fullmatch(r"R-[ABCD]")].copy()
    real["sc"] = real["dataset"].str[-1]
    skipped = sorted(set(m["dataset"]) - set(runs["dataset"]) - set(real["dataset"]))
    if skipped:
        print(f"Not used (no seed in name): {', '.join(skipped)}")
    return runs, real.set_index("sc")


def grid(runs: pd.DataFrame, sc: str, metric: str) -> np.ndarray:
    """Πίνακας μοντέλα × γενέσεις (NaN όπου λείπει γένεση)."""
    t = runs[runs.sc == sc].pivot_table(index="seed", columns="gen", values=metric)
    return t.to_numpy(dtype=float)


def model_means(V: np.ndarray) -> np.ndarray:
    return np.nanmean(V, axis=1)


# ---------------------------------------------------------------------------
# Ιεραρχικό bootstrap
# ---------------------------------------------------------------------------
def boot_means(V: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B bootstrap μέσοι του σεναρίου: training seeds με επανάθεση, και μέσα σε κάθε
    επιλεγμένο μοντέλο οι γενέσεις του με επανάθεση (όσες έχει)."""
    n_t, max_g = V.shape
    counts = (~np.isnan(V)).sum(axis=1)
    vals = np.where(np.isnan(V), 0.0, V)
    # οι διαθέσιμες γενέσεις κάθε μοντέλου πρώτα (για μη ισόρροπους πίνακες)
    order = np.argsort(np.isnan(V), axis=1, kind="stable")
    vals = np.take_along_axis(vals, order, axis=1)

    t = rng.integers(0, n_t, size=(B, n_t))                      # επιλεγμένα μοντέλα
    c = counts[t]                                                # γενέσεις ανά επιλογή
    g = np.floor(rng.random((B, n_t, max_g)) * c[..., None]).astype(int)
    picked = vals[t[..., None], g]                               # (B, n_t, max_g)
    mask = np.arange(max_g)[None, None, :] < c[..., None]
    per_model = (picked * mask).sum(axis=2) / c
    return per_model.mean(axis=1)


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


# ---------------------------------------------------------------------------
# Μεγέθη επίδρασης και διακύμανση
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) − P(X < Y). Θετικό: το πρώτο σενάριο έχει μεγαλύτερες τιμές."""
    d = np.sign(x[:, None] - y[None, :])
    return float(d.mean())


def cliffs_label(d: float) -> str:
    a = abs(d)          # κατώφλια Romano et al. (2006)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def welch_ci(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    if se == 0:
        return float(x.mean() - y.mean()), float(x.mean() - y.mean())
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    q = stats.t.ppf(0.5 + level / 2, df)
    d = x.mean() - y.mean()
    return float(d - q * se), float(d + q * se)


def variance_components(V: np.ndarray) -> tuple[float, float]:
    """Μέθοδος ροπών για φωλιασμένο σχεδιασμό (γενέσεις μέσα σε μοντέλα).
    Επιστρέφει (σ²_train, σ²_gen)."""
    counts = (~np.isnan(V)).sum(axis=1)
    if len(V) < 2 or counts.min() < 2:
        return np.nan, np.nan
    n_g = len(counts) / np.sum(1 / counts)                       # αρμονικός μέσος
    ms_within = np.nanmean(np.nanvar(V, axis=1, ddof=1))
    ms_between = n_g * np.var(model_means(V), ddof=1)
    return float(max(0.0, (ms_between - ms_within) / n_g)), float(ms_within)


def better(diff: float, cis: list, first: str, second: str, direction) -> str:
    """Ποιο σενάριο είναι καλύτερο, μόνο όταν ΚΑΝΕΝΑ από τα διαθέσιμα CI (bootstrap,
    Welch) δεν περιέχει το 0."""
    cis = [(lo, hi) for lo, hi in cis if not np.isnan(lo)]
    if direction is None or not cis or any(lo <= 0 <= hi for lo, hi in cis):
        return "–"
    first_higher = diff > 0
    return first if first_higher == (direction == "high") else second


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-file", default=str(EVAL_DIR / "metrics.csv"))
    ap.add_argument("--boot", type=int, default=10_000, help="επαναλήψεις bootstrap")
    ap.add_argument("--all-metrics", action="store_true",
                    help="όλες οι αριθμητικές στήλες του metrics.csv, όχι μόνο οι κύριες")
    args = ap.parse_args()

    runs, real = load_runs(Path(args.metrics_file))
    if runs.empty:
        raise SystemExit("No seeded runs (tabddpm_{scenario}_s{seed}_g{gen}) in metrics.csv")

    present = [s for s in SCENARIOS if s in set(runs.sc)]
    print(f"Seeded runs: {len(runs)}")
    for s in present:
        r = runs[runs.sc == s]
        per = r.groupby("seed").gen.apply(lambda g: sorted(g)).to_dict()
        print(f"  {s}: {r.seed.nunique()} models, {len(r)} runs  {per}")

    metrics = METRICS
    if args.all_metrics:
        known = {m for m, _, _ in METRICS}
        skip = {"rows", "seed", "gen"} | {c for c in runs.columns if c.endswith("_std")}
        extra = [(c, c, None) for c in runs.select_dtypes("number").columns
                 if c not in known and c not in skip]
        metrics = METRICS + extra

    summary, contrasts, variance = [], [], []
    for k, (metric, label, direction) in enumerate(metrics):
        rng = np.random.default_rng(SAMPLING_SEED + k)
        V = {s: grid(runs, s, metric) for s in present}
        M = {s: model_means(V[s]) for s in present}
        boots = {s: boot_means(V[s], args.boot, rng) for s in present}

        for s in present:
            lo, hi = ci(boots[s])
            var_t, var_g = variance_components(V[s])
            r_val = real.loc[s, metric] if s in real.index else np.nan
            d_lo, d_hi = ci(boots[s] - r_val) if not np.isnan(r_val) else (np.nan, np.nan)
            summary.append({
                "metric": metric, "label": label, "better": direction or "", "scenario": s,
                "n_models": len(M[s]), "n_runs": int((~np.isnan(V[s])).sum()),
                "mean": M[s].mean(), "ci_low": lo, "ci_high": hi,
                "sd_models": M[s].std(ddof=1) if len(M[s]) > 1 else np.nan,
                "sd_generations": np.sqrt(np.nanmean(np.nanvar(V[s], axis=1, ddof=1)))
                if V[s].shape[1] > 1 else np.nan,
                "real": r_val, "delta": M[s].mean() - r_val, "delta_ci_low": d_lo,
                "delta_ci_high": d_hi,
            })
            total = var_t + var_g
            variance.append({"metric": metric, "scenario": s, "var_train": var_t,
                             "var_gen": var_g,
                             "share_train": var_t / total if total and total > 0 else np.nan})

        for x, y in CONTRASTS:
            if x not in present or y not in present:
                continue
            diff = M[x].mean() - M[y].mean()
            lo, hi = ci(boots[x] - boots[y])
            w_lo, w_hi = welch_ci(M[x], M[y])
            delta = cliffs_delta(M[x], M[y])
            real_gap = (real.loc[x, metric] - real.loc[y, metric]
                        if x in real.index and y in real.index else np.nan)
            contrasts.append({
                "metric": metric, "label": label, "better": direction or "",
                "contrast": f"{x}-{y}", "kind": "transfer gap" if y == "D" else "regime",
                "diff": diff, "ci_low": lo, "ci_high": hi, "welch_low": w_lo, "welch_high": w_hi,
                "ci_excludes_0": not (lo <= 0 <= hi),
                "welch_excludes_0": not np.isnan(w_lo) and not (w_lo <= 0 <= w_hi),
                "cliffs_delta": delta, "cliffs_size": cliffs_label(delta),
                "winner": better(diff, [(lo, hi), (w_lo, w_hi)], x, y, direction), "real_gap": real_gap,
            })

    summary, contrasts, variance = map(pd.DataFrame, (summary, contrasts, variance))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EVAL_DIR / "level1_summary.csv", index=False)
    contrasts.to_csv(EVAL_DIR / "level1_contrasts.csv", index=False)
    variance.to_csv(EVAL_DIR / "level1_variance.csv", index=False)

    # ---------------- εκτύπωση ----------------
    pd.set_option("display.width", 200)
    print("\n===== Scenario means [95% hierarchical bootstrap CI]; delta = synthetic - real =====")
    for metric, label, direction in METRICS:
        s = summary[summary.metric == metric]
        if s.empty:
            continue
        arrow = {"low": "lower is better", "high": "higher is better"}.get(direction, "descriptive")
        print(f"\n{label} ({arrow})")
        for _, r in s.iterrows():
            sd = (f"sd models {r.sd_models:.4f} | sd gens {r.sd_generations:.4f}"
                  if not np.isnan(r.sd_models) else "")
            print(f"  {r.scenario}: {r['mean']:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]  "
                  f"real {r.real:.4f}  delta {r.delta:+.4f} [{r.delta_ci_low:+.4f}, {r.delta_ci_high:+.4f}]  {sd}")

    if not contrasts.empty:
        print("\n===== Contrasts: diff [bootstrap CI] | Welch CI | Cliff's delta | winner | real gap =====")
        for metric, label, _ in METRICS:
            c = contrasts[contrasts.metric == metric]
            if c.empty:
                continue
            print(f"\n{label}")
            for _, r in c.iterrows():
                print(f"  {r.contrast:4s} {r['diff']:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
                      f" | Welch [{r.welch_low:+.4f}, {r.welch_high:+.4f}]"
                      f" | d {r.cliffs_delta:+.2f} ({r.cliffs_size}) | {r.winner:2s}"
                      f" | real {r.real_gap:+.4f}")

    v = variance[variance.metric.isin([m for m, _, _ in METRICS])].dropna(subset=["share_train"])
    if not v.empty:
        print("\n===== Share of variance from training seeds (rest: generation seeds) =====")
        print(v.pivot(index="metric", columns="scenario", values="share_train").round(2).to_string())

    print(f"\nSaved level1_summary.csv, level1_contrasts.csv, level1_variance.csv in {EVAL_DIR}")


if __name__ == "__main__":
    main()
