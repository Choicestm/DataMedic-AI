"""
train_models.py

General model training for the Dataset Doctor project.

Given a dataframe and a target column, this:
 1. calls prepare_data() to get a clean train/test split
 2. trains a small set of classification OR regression models,
    depending on what prepare_data() detected
 3. scores every model with sensible metrics
 4. picks the best model
 5. returns everything the SHAP step / interface will need

Nothing here is tied to a specific dataset, column name, or file.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

try:
    # Works when imported as part of the machine_learning package
    # (e.g. from app.py: `from machine_learning.train_models import train_models`)
    from .preprocessing import prepare_data
except ImportError:
    # Works when running this file directly: `python train_models.py`
    from preprocessing import prepare_data


def _get_classification_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }


def _get_regression_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }


def _score_classification(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def _score_regression(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, y_pred),
    }


def train_models(df, target_column):
    """
    Train and compare models on any dataframe + target column.

    Returns
    -------
    dict with:
        problem_type    : "classification" or "regression"
        primary_metric  : metric used to pick the best model
        results         : {model_name: {"model": fitted_model, "metrics": {...}}}
        best_model_name : name of the best-performing model
        best_model      : the fitted best model object
        data            : the dict returned by prepare_data (needed later
                           for SHAP - contains preprocessor, feature_names, etc.)
    """
    data = prepare_data(df, target_column)

    problem_type = data["problem_type"]
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    if problem_type == "classification":
        models = _get_classification_models()
        score_fn = _score_classification
        primary_metric = "accuracy"
    else:
        models = _get_regression_models()
        score_fn = _score_regression
        primary_metric = "r2"

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = score_fn(y_test, y_pred)
        results[name] = {"model": model, "metrics": metrics}

    # Higher is better for every metric we picked (accuracy and R^2).
    best_model_name = max(results, key=lambda name: results[name]["metrics"][primary_metric])
    best_model = results[best_model_name]["model"]

    return {
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "data": data,
    }


def print_results(training_output):
    """Pretty-print results - a preview of what the interface will show."""
    problem_type = training_output["problem_type"]
    print(f"Problem type: {problem_type.capitalize()}\n")
    print("Model Results:")
    for name, info in training_output["results"].items():
        metrics = info["metrics"]
        metrics_str = "  ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items())
        print(f"  {name:<20} {metrics_str}")

    print(f"\nBest model: {training_output['best_model_name']}")


if __name__ == "__main__":
    # Manual test block - asks for any CSV, same style as analyzer.py.
    import pandas as pd

    csv_path = input("Enter the path to your CSV file: ")
    df = pd.read_csv(csv_path)

    print(f"\nAvailable columns: {', '.join(df.columns)}")
    target_column = input("Enter target column: ")

    output = train_models(df, target_column)
    print_results(output)