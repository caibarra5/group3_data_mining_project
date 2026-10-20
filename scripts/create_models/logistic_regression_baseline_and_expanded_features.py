import os
import pandas as pd

from src.config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    MODELS_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
)

# Load datasets.
train = pd.read_stata(TRAIN_FILE, convert_categoricals=False)
validation = pd.read_stata(VALIDATION_FILE, convert_categoricals=False)
test = pd.read_stata(TEST_FILE, convert_categoricals=False)

# Change to "knn" or "decision_tree" for your algorithm.
MODEL_NAME = "logistic"

# Model paths.
baseline_model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_baseline.joblib")
expanded_model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_expanded.joblib")

# Prediction paths.
baseline_test_predictions_path = os.path.join(
    PREDICTIONS_DIR, f"{MODEL_NAME}_baseline_test.csv"
)
expanded_test_predictions_path = os.path.join(
    PREDICTIONS_DIR, f"{MODEL_NAME}_expanded_test.csv"
)

# Results table paths.
validation_results_path = os.path.join(
    TABLES_DIR, f"{MODEL_NAME}_validation_results.csv"
)
test_metrics_path = os.path.join(
    TABLES_DIR, f"{MODEL_NAME}_test_metrics.csv"
)

# Figure paths.
baseline_confusion_matrix_path = os.path.join(
    FIGURES_DIR, f"{MODEL_NAME}_baseline_confusion_matrix.png"
)
expanded_confusion_matrix_path = os.path.join(
    FIGURES_DIR, f"{MODEL_NAME}_expanded_confusion_matrix.png"
)

# ######################################## Model training and evaluation ########################################
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from src.config import (
    BASELINE_FEATURES,
    EXPANDED_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ROW_ID,
    PREDICTION_COLUMN,
)

for directory in [MODELS_DIR, PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

feature_sets = {
    "baseline": BASELINE_FEATURES,
    "expanded": EXPANDED_FEATURES,
}

model_paths = {
    "baseline": baseline_model_path,
    "expanded": expanded_model_path,
}

prediction_paths = {
    "baseline": baseline_test_predictions_path,
    "expanded": expanded_test_predictions_path,
}

figure_paths = {
    "baseline": baseline_confusion_matrix_path,
    "expanded": expanded_confusion_matrix_path,
}


def create_pipeline(features, C):
    categorical = [
        column for column in features
        if column in CATEGORICAL_FEATURES
    ]
    numeric = [
        column for column in features
        if column not in CATEGORICAL_FEATURES
    ]

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])


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


def save_predictions(data, predictions, path):
    pd.DataFrame({
        ROW_ID: data[ROW_ID].to_numpy(),
        TARGET: data[TARGET].to_numpy(),
        PREDICTION_COLUMN: predictions,
    }).to_csv(path, index=False)


# Check shared data.
for data in [train, validation, test]:
    assert data[ROW_ID].notna().all()
    assert data[ROW_ID].is_unique
    assert data[TARGET].isin([0, 1]).all()
    assert not data[EXPANDED_FEATURES].isna().any().any()

train_ids = set(train[ROW_ID])
validation_ids = set(validation[ROW_ID])
test_ids = set(test[ROW_ID])

assert train_ids.isdisjoint(validation_ids)
assert train_ids.isdisjoint(test_ids)
assert validation_ids.isdisjoint(test_ids)


# Train and validate.
validation_results = []
selected_C = {}

for name, features in feature_sets.items():
    best_f1 = -1
    best_pipeline = None
    best_row = None

    for C in np.linspace(0.02, 0.5, 50):
        pipeline = create_pipeline(features, C)
        pipeline.fit(train[features], train[TARGET])

        predictions = pipeline.predict(validation[features])
        metrics = calculate_metrics(validation[TARGET], predictions)

        result = {
            "model": MODEL_NAME,
            "feature_set": name,
            "C": C,
            **metrics,
            "selected": False,
        }
        validation_results.append(result)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            selected_C[name] = C
            best_pipeline = pipeline
            best_row = result

    best_row["selected"] = True

    # These predictions come from a model fitted only on training data.
    validation_predictions = best_pipeline.predict(validation[features])
    print(pd.Series(validation_predictions).value_counts())

    save_predictions(
        validation,
        validation_predictions,
        os.path.join(
            PREDICTIONS_DIR,
            f"{MODEL_NAME}_{name}_validation.csv",
        ),
    )
    print(validation[TARGET].value_counts(normalize=True))

    print(
        f"{name}: selected C={selected_C[name]}, "
        f"validation F1={best_f1:.4f}"
    )

pd.DataFrame(validation_results).to_csv(
    validation_results_path, index=False
)

validation_df = pd.DataFrame(validation_results)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ax, metric in zip(
    axes.flat, ["accuracy", "precision", "recall", "f1"]
):
    for name in feature_sets:
        subset = validation_df[
            validation_df["feature_set"] == name
        ].sort_values("C")

        ax.plot(
            subset["C"],
            subset[metric],
            marker="o",
            label=name.capitalize(),
        )

    ax.set_xscale("log")
    ax.set_xlabel("C (larger = weaker regularization)")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.set_title(f"Validation {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Logistic Regression: Validation Metrics vs. C")
fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES_DIR, f"{MODEL_NAME}_validation_metrics_vs_C.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# Set True only when all model and ensemble choices are finalized.
RUN_FINAL_TEST = True

if RUN_FINAL_TEST:
    train_validation = pd.concat(
        [train, validation], ignore_index=True
    )
    test_results = []

    for name, features in feature_sets.items():
        final_pipeline = create_pipeline(features, selected_C[name])
        final_pipeline.fit(
            train_validation[features],
            train_validation[TARGET],
        )

        joblib.dump(final_pipeline, model_paths[name])

        predictions = final_pipeline.predict(test[features])
        metrics = calculate_metrics(test[TARGET], predictions)

        save_predictions(test, predictions, prediction_paths[name])

        test_results.append({
            "model": MODEL_NAME,
            "feature_set": name,
            "C": selected_C[name],
            **metrics,
        })

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            test[TARGET],
            predictions,
            labels=[0, 1],
            display_labels=["Lower expenditure", "High expenditure"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"{MODEL_NAME}: {name}")
        fig.tight_layout()
        fig.savefig(figure_paths[name], dpi=300)
        plt.close(fig)

        print(f"\n{name} test metrics:")
        print(metrics)

    pd.DataFrame(test_results).to_csv(
        test_metrics_path, index=False
    )