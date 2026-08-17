"""
shap_explainer.py

General SHAP explainability module for DataMedic-AI.

This module:
- receives the output from train_models()
- identifies the selected best model
- chooses an appropriate SHAP explainer
- calculates SHAP values
- calculates global feature importance
- generates a SHAP summary plot

Nothing here is tied to a specific dataset.
"""

from pathlib import Path

import matplotlib

# Use a non-GUI backend because SHAP plots are generated
# inside the Flask web application and saved to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


MAX_SHAP_SAMPLES = 200
MAX_BACKGROUND_SAMPLES = 100


def _prepare_explanation_data(
    X_test,
    feature_names,
):
    """
    Convert processed test data into a DataFrame
    with readable feature names.
    """

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_names,
    )

    if len(X_test_df) > MAX_SHAP_SAMPLES:
        return X_test_df.sample(
            n=MAX_SHAP_SAMPLES,
            random_state=42,
        )

    return X_test_df.copy()


def _prepare_background_data(
    X_explain,
):
    """
    Create a smaller background dataset for SHAP.
    """

    if len(X_explain) > MAX_BACKGROUND_SAMPLES:
        return X_explain.sample(
            n=MAX_BACKGROUND_SAMPLES,
            random_state=42,
        )

    return X_explain.copy()


def _create_explainer(
    model,
    X_explain,
):
    """
    Select an appropriate SHAP explainer based on model type.
    """

    tree_models = (
        RandomForestClassifier,
        RandomForestRegressor,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )

    linear_models = (
        LogisticRegression,
        LinearRegression,
    )

    if isinstance(model, tree_models):
        print(
            "Using SHAP TreeExplainer..."
        )

        explainer = shap.TreeExplainer(
            model
        )

        return (
            explainer,
            "tree",
        )

    if isinstance(model, linear_models):
        print(
            "Using SHAP LinearExplainer..."
        )

        background_data = _prepare_background_data(
            X_explain
        )

        explainer = shap.LinearExplainer(
            model,
            background_data,
        )

        return (
            explainer,
            "linear",
        )

    print(
        "Using general SHAP Explainer..."
    )

    background_data = _prepare_background_data(
        X_explain
    )

    explainer = shap.Explainer(
        model,
        background_data,
    )

    return (
        explainer,
        "general",
    )


def _calculate_shap_values(
    explainer,
    X_explain,
    explainer_type,
):
    """
    Calculate SHAP values safely.

    TreeExplainer may occasionally fail its strict
    additivity check because of small numerical differences.
    """

    if explainer_type == "tree":
        try:
            return explainer(
                X_explain
            )

        except Exception as error:
            error_message = str(
                error
            ).lower()

            if "additivity" in error_message:
                print(
                    "SHAP additivity warning detected."
                )

                print(
                    "Retrying with additivity check disabled..."
                )

                return explainer(
                    X_explain,
                    check_additivity=False,
                )

            raise

    return explainer(
        X_explain
    )


def _calculate_feature_importance(
    shap_values,
    feature_names,
):
    """
    Calculate overall feature importance using
    the average absolute SHAP value.
    """

    values = np.asarray(
        shap_values.values
    )

    if values.ndim == 3:
        importance = np.abs(
            values
        ).mean(
            axis=(0, 2)
        )

    elif values.ndim == 2:
        importance = np.abs(
            values
        ).mean(
            axis=0
        )

    else:
        raise ValueError(
            "Unexpected SHAP value dimensions: "
            f"{values.shape}"
        )

    if len(importance) != len(feature_names):
        raise ValueError(
            "The number of SHAP importance values "
            "does not match the number of feature names."
        )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return feature_importance


def _save_summary_plot(
    shap_values,
    X_explain,
    output_dir,
):
    """
    Generate and save the SHAP summary visualization.

    All Matplotlib figures are closed after saving so
    Flask does not leave GUI resources running.
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        output_path
        / "shap_summary.png"
    )

    # Clear any previous plots.
    plt.close("all")

    try:
        shap.summary_plot(
            shap_values,
            X_explain,
            show=False,
        )

    except Exception:
        shap.summary_plot(
            shap_values.values,
            X_explain,
            show=False,
        )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close("all")

    return str(
        plot_path
    )


def explain_best_model(
    training_output,
    output_dir="assets/generated",
):
    """
    Explain the best model returned by train_models().

    Parameters
    ----------
    training_output : dict
        Output returned by train_models().

    output_dir : str
        Directory where SHAP plots will be saved.

    Returns
    -------
    dict
        Contains:
        - best_model_name
        - problem_type
        - explainer_type
        - feature_importance
        - shap_values
        - feature_names
        - summary_plot_path
    """

    best_model = training_output[
        "best_model"
    ]

    best_model_name = training_output[
        "best_model_name"
    ]

    problem_type = training_output[
        "problem_type"
    ]

    data = training_output[
        "data"
    ]

    X_test = data[
        "X_test"
    ]

    feature_names = data[
        "feature_names"
    ]

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    X_explain = _prepare_explanation_data(
        X_test,
        feature_names,
    )

    print(
        f"Explaining {len(X_explain):,} "
        f"test rows with SHAP..."
    )

    # ---------------------------------------------------------
    # Choose explainer
    # ---------------------------------------------------------

    (
        explainer,
        explainer_type,
    ) = _create_explainer(
        best_model,
        X_explain,
    )

    # ---------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------

    shap_values = _calculate_shap_values(
        explainer,
        X_explain,
        explainer_type,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    feature_importance = _calculate_feature_importance(
        shap_values,
        feature_names,
    )

    # ---------------------------------------------------------
    # Save SHAP plot
    # ---------------------------------------------------------

    summary_plot_path = _save_summary_plot(
        shap_values,
        X_explain,
        output_dir,
    )

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------

    return {
        "best_model_name": best_model_name,
        "problem_type": problem_type,
        "explainer_type": explainer_type,
        "feature_importance": feature_importance,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "summary_plot_path": summary_plot_path,
    }