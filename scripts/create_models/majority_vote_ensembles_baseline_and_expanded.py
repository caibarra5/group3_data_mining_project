import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from src.config import (
    TEST_FILE,
    MODELS_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    BASELINE_FEATURES,
    EXPANDED_FEATURES,
    TARGET,
    ROW_ID,
    PREDICTION_COLUMN,
)

# Load shared test observations.
test = pd.read_stata(TEST_FILE, convert_categoricals=False)

assert test[ROW_ID].notna().all()
assert test[ROW_ID].is_unique
assert test[TARGET].isin([0, 1]).all()

for directory in [PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

feature_sets = {
    "baseline": BASELINE_FEATURES,
    "expanded": EXPANDED_FEATURES,
}

model_names = ["logistic", "knn", "decision_tree"]


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


all_metrics = []

for feature_set, features in feature_sets.items():
    X_test = test[features]

    # Every model predicts on the same rows in the same order.
    output = pd.DataFrame({
        ROW_ID: test[ROW_ID].to_numpy(),
        TARGET: test[TARGET].to_numpy(),
    })

    vote_columns = []

    for model_name in model_names:
        model_path = os.path.join(
            MODELS_DIR,
            f"{model_name}_{feature_set}.joblib",
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        if os.path.getsize(model_path) == 0:
            raise ValueError(f"Empty model placeholder: {model_path}")

        pipeline = joblib.load(model_path)
        if model_name == "knn":
            pipeline.set_params(classifier__n_jobs=1)
        predictions = pipeline.predict(X_test)

        assert len(predictions) == len(test)
        assert np.isin(predictions, [0, 1]).all()

        vote_column = f"{model_name}_vote"
        output[vote_column] = predictions
        vote_columns.append(vote_column)

        # Recalculate individual metrics for direct comparison.
        all_metrics.append({
            "model": model_name,
            "feature_set": feature_set,
            **calculate_metrics(test[TARGET], predictions),
        })

    # Two or three positive votes produce class 1.
    output["positive_votes"] = output[vote_columns].sum(axis=1)
    output[PREDICTION_COLUMN] = (
        output["positive_votes"] >= 2
    ).astype("int8")

    ensemble_predictions = output[PREDICTION_COLUMN].to_numpy()

    output.to_csv(
        os.path.join(
            PREDICTIONS_DIR,
            f"ensemble_{feature_set}_test.csv",
        ),
        index=False,
    )

    all_metrics.append({
        "model": "ensemble",
        "feature_set": feature_set,
        **calculate_metrics(test[TARGET], ensemble_predictions),
    })

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        test[TARGET],
        ensemble_predictions,
        labels=[0, 1],
        display_labels=["Lower expenditure", "High expenditure"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(f"Majority-vote ensemble: {feature_set}")
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            FIGURES_DIR,
            f"ensemble_{feature_set}_confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# Save individual and ensemble test metrics together.
metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    os.path.join(TABLES_DIR, "models_and_ensembles_test_metrics.csv"),
    index=False,
)

# Compare expanded minus baseline performance.
metric_names = ["accuracy", "precision", "recall", "f1"]

baseline_metrics = metrics_df[
    metrics_df["feature_set"] == "baseline"
].set_index("model")[metric_names]

expanded_metrics = metrics_df[
    metrics_df["feature_set"] == "expanded"
].set_index("model")[metric_names]

differences = expanded_metrics - baseline_metrics
differences.columns = [
    f"{metric}_expanded_minus_baseline"
    for metric in metric_names
]

differences.to_csv(
    os.path.join(TABLES_DIR, "expanded_minus_baseline_test_metrics.csv")
)

print("\nIndividual and ensemble test metrics:")
print(metrics_df.round(4).to_string(index=False))

print("\nExpanded minus baseline:")
print(differences.round(4).to_string())