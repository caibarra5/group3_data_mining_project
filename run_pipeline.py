"""Run the project pipeline. Place this file in the repository root.

The modeling scripts must have RUN_FINAL_TEST = True for a complete run.
Outputs are overwritten. Existing raw data and dependencies are required.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

PREPARATION_MODULES = [
    "scripts.preprocessing_data.create_column_subset_data",
    "scripts.preprocessing_data.clean_column_subset_data",
    "scripts.preprocessing_data.split_column_subset_and_cleaned_data",
]
MODEL_MODULES = [
    "scripts.create_models.logistic_regression_baseline_and_expanded_features",
    "scripts.create_models.knn_baseline_and_expanded_features",
    "scripts.create_models.decision_tree_baseline_and_expanded_features",
]
ENSEMBLE_MODULE = (
    "scripts.create_models.majority_vote_ensembles_baseline_and_expanded"
)


def module_path(module):
    return PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")


def preflight():
    # Check all scripts before overwriting any outputs.
    for module in PREPARATION_MODULES + MODEL_MODULES + [ENSEMBLE_MODULE]:
        if not module_path(module).is_file():
            raise FileNotFoundError(module_path(module))

    # Avoid evaluating stale saved models when final fitting is disabled.
    for module in MODEL_MODULES:
        tree = ast.parse(module_path(module).read_text(encoding="utf-8-sig"))
        settings = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "RUN_FINAL_TEST"
                for target in node.targets
            )
        ]
        if not settings or not isinstance(settings[-1], ast.Constant) or settings[-1].value is not True:
            raise ValueError(
                f"Set RUN_FINAL_TEST = True in {module_path(module)} "
                "before running the complete pipeline."
            )


def main():
    preflight()
    environment = os.environ.copy()
    # Save figures without opening blocking windows.
    environment["MPLBACKEND"] = "Agg"

    modules = PREPARATION_MODULES + MODEL_MODULES + [ENSEMBLE_MODULE]
    for number, module in enumerate(modules, start=1):
        print(f"\n[{number}/{len(modules)}] Running {module}", flush=True)
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    print("\nPipeline completed. Outputs are in results/.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"\nPipeline stopped: {error}", file=sys.stderr)
        sys.exit(1)
