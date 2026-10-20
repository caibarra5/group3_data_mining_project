import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency

from src.config import (
    FIGURES_DIR,
    PREDICTION_COLUMN,
    PREDICTIONS_DIR,
    ROW_ID,
    SEED,
    TABLES_DIR,
    TARGET,
    TRAIN_FILE,
)

MODELS = ["logistic", "knn", "decision_tree", "ensemble"]
N_BOOTSTRAP = 2000


def load_pair(model):
    def read(feature_set):
        path = os.path.join(PREDICTIONS_DIR, f"{model}_{feature_set}_test.csv")
        df = pd.read_csv(path, dtype={ROW_ID: str})
        return df.set_index(ROW_ID)[[TARGET, PREDICTION_COLUMN]]

    baseline, expanded = read("baseline"), read("expanded")
    expanded = expanded.loc[baseline.index]
    assert (baseline[TARGET] == expanded[TARGET]).all()
    return baseline[TARGET].to_numpy(), baseline[PREDICTION_COLUMN].to_numpy(), \
        expanded[PREDICTION_COLUMN].to_numpy()


def prf(y, p):
    tp = np.sum((p == 1) & (y == 1))
    fp = np.sum((p == 1) & (y == 0))
    fn = np.sum((p == 0) & (y == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mcnemar(y, base, exp):
    base_right, exp_right = base == y, exp == y
    only_base = int(np.sum(base_right & ~exp_right))
    only_exp = int(np.sum(~base_right & exp_right))
    n = only_base + only_exp
    p = binomtest(only_exp, n, 0.5).pvalue if n else 1.0
    return only_base, only_exp, p


def bootstrap(y, base, exp, rng):
    n = len(y)
    diffs = {m: [] for m in ["precision", "recall", "f1"]}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        b, e = prf(y[idx], base[idx]), prf(y[idx], exp[idx])
        for m in diffs:
            diffs[m].append(e[m] - b[m])
    return {m: np.percentile(v, [2.5, 97.5]) for m, v in diffs.items()}


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    for model in MODELS:
        y, base, exp = load_pair(model)
        b, e = prf(y, base), prf(y, exp)
        only_base, only_exp, p_mcnemar = mcnemar(y, base, exp)
        ci = bootstrap(y, base, exp, rng)
        for metric in ["precision", "recall", "f1"]:
            low, high = ci[metric]
            rows.append({
                "model": model,
                "metric": metric,
                "baseline": b[metric],
                "expanded": e[metric],
                "expanded_minus_baseline": e[metric] - b[metric],
                "ci_95_low": low,
                "ci_95_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "mcnemar_only_baseline_correct": only_base,
                "mcnemar_only_expanded_correct": only_exp,
                "mcnemar_p_value": p_mcnemar,
            })

    tests = pd.DataFrame(rows)
    tests.to_csv(os.path.join(TABLES_DIR, "significance_expanded_vs_baseline_test.csv"),
                 index=False)

    # Majority-class reference on the test set.
    y_test = pd.read_csv(os.path.join(PREDICTIONS_DIR, "ensemble_baseline_test.csv"))[TARGET]
    reference = {
        "model": "majority_class_reference",
        "accuracy": (y_test == 0).mean(),
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }
    pd.DataFrame([reference]).to_csv(
        os.path.join(TABLES_DIR, "majority_class_reference_test.csv"), index=False)

    # Chi-square tests on the training set.
    train = pd.read_stata(TRAIN_FILE, convert_categoricals=False)
    chi_rows = []
    labels = {
        "PHYEXE53": {1: "Exercises", 2: "Does not exercise"},
        "OFTSMK53": {1: "Smokes every day", 2: "Smokes some days", 3: "Does not smoke"},
    }
    for var, mapping in labels.items():
        table = pd.crosstab(train[var], train[TARGET])
        chi2, p, dof, _ = chi2_contingency(table)
        rates = train.groupby(var)[TARGET].mean()
        for code, label in mapping.items():
            chi_rows.append({
                "variable": var, "group": label, "n": int(table.loc[code].sum()),
                "pct_high_spender": rates[code], "chi2": chi2, "dof": dof, "p_value": p,
            })
    chi = pd.DataFrame(chi_rows)
    chi.to_csv(os.path.join(TABLES_DIR, "chi_square_behaviors_train.csv"), index=False)

    # Figure: baseline vs expanded test F1 / precision / recall for every model.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.38
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        sub = tests[tests["metric"] == metric].set_index("model").loc[MODELS]
        ax.bar(x - width / 2, sub["baseline"], width, label="Baseline", color="#4C72B0")
        ax.bar(x + width / 2, sub["expanded"], width, label="Expanded", color="#DD8452")
        for i, (bv, ev) in enumerate(zip(sub["baseline"], sub["expanded"])):
            ax.text(i - width / 2, bv + 0.015, f"{bv:.2f}", ha="center", fontsize=8)
            ax.text(i + width / 2, ev + 0.015, f"{ev:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Logistic", "KNN", "Decision\ntree", "Ensemble"])
        ax.set_title(metric.upper() if metric == "f1" else metric.capitalize())
        ax.set_ylim(0, 0.85)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Test-set score")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Baseline vs. expanded features on the test set (n = 2,900)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "baseline_vs_expanded_test.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Figure: high-spender rate by behavior (training set).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    overall = train[TARGET].mean()
    for ax, var in zip(axes, labels):
        sub = chi[chi["variable"] == var]
        ax.barh(sub["group"], sub["pct_high_spender"], color="#4C72B0")
        for i, v in enumerate(sub["pct_high_spender"]):
            ax.text(v / 2, i, f"{v:.1%}", va="center", ha="center", fontsize=10,
                    color="white", fontweight="bold")
        ax.axvline(overall, color="gray", linestyle="--", linewidth=1, label="All adults")
        p = sub["p_value"].iloc[0]
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        ax.set_title(f"{'Exercise' if var == 'PHYEXE53' else 'Smoking'}  (chi-square {p_text})")
        ax.set_xlim(0, 0.35)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Share who are high spenders")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "high_spender_rate_by_behavior.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(tests.round(4).to_string(index=False))
    print("\nMajority-class reference:", {k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in reference.items()})
    print("\n", chi.round(4).to_string(index=False))