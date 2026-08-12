"""
preprocessing.py

General-purpose preprocessing for the Dataset Doctor project.

Given ANY dataframe and a target column, this module:
 - figures out whether the problem is classification or regression
 - drops columns that would hurt a general pipeline (constant columns,
   ID-like columns with one unique value per row)
 - imputes missing values
 - scales numeric features and one-hot encodes categorical features
 - splits the data into train/test sets

It does not know or care what dataset it is looking at. There is
nothing in this file specific to any CSV.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def detect_problem_type(y, classification_max_unique=20, unique_ratio_threshold=0.05):
    """
    Decide whether a target column represents classification or regression.

    Rules, in order:
      1. Non-numeric target (text, category, bool) -> classification.
         (e.g. "NORMAL"/"AGGRESSIVE"/"SLOW")
      2. Numeric target with few distinct values -> classification.
         Triggered if the number of unique values is small in absolute
         terms (<= classification_max_unique) OR small relative to the
         number of rows (< unique_ratio_threshold). This catches things
         like a 0/1 flag or a 1-5 star rating stored as numbers, without
         misclassifying a continuous target (e.g. house price) that just
         happens to repeat a few values.
      3. Otherwise -> regression.
    """
    y = pd.Series(y).dropna()

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n_unique = y.nunique()
    n_rows = len(y)

    if n_unique <= classification_max_unique:
        return "classification"

    if n_rows > 0 and (n_unique / n_rows) < unique_ratio_threshold:
        return "classification"

    return "regression"


def _coerce_numeric_like_columns(X, min_success_ratio=0.95):
    """
    Some real-world CSVs store a genuinely numeric column as text,
    usually because a few rows contain a stray value like a blank
    space, "N/A", or "-" instead of a number (a very common export
    quirk, e.g. Excel/CSV exports of billing data).

    If a text column can be parsed as numbers for at least
    `min_success_ratio` of its non-null values, we convert it to a
    real numeric column (the few unparseable entries become NaN and
    get imputed normally downstream). Otherwise we leave it alone.

    This is a general fix, not specific to any one dataset - it
    protects against silently losing a useful numeric feature just
    because a handful of rows had bad values.
    """
    X = X.copy()
    for col in X.select_dtypes(exclude=["number", "bool"]).columns:
        non_null = X[col].dropna()
        if len(non_null) == 0:
            continue
        converted = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
        success_ratio = converted.notna().mean()
        if success_ratio >= min_success_ratio:
            X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors="coerce")
    return X


def _clean_columns(X):
    """
    Drop columns that are useless or actively harmful for a general model:
      - constant columns (only one distinct value -> zero information),
        for ANY dtype
      - ID-like TEXT columns (as many unique values as rows -> something
        like a name, a UUID, or a free-text identifier; one-hot encoding
        it would create one column per row, which is useless)

    Note: this deliberately does NOT drop numeric columns just because
    every value is unique. A continuous feature such as "income" or a
    sensor reading is *expected* to have a unique value per row - that's
    normal, not an ID column, and must stay.
    """
    cols_to_drop = []
    n_rows = len(X)

    for col in X.columns:
        series = X[col]
        n_unique = series.nunique(dropna=True)

        if n_unique <= 1:
            cols_to_drop.append(col)
            continue

        is_text_like = not pd.api.types.is_numeric_dtype(series)
        if is_text_like and n_rows > 1 and n_unique == n_rows:
            cols_to_drop.append(col)

    return X.drop(columns=cols_to_drop), cols_to_drop


def build_preprocessor(X):
    """
    Build a ColumnTransformer that:
      - imputes (median) + scales numeric columns
      - imputes (most frequent) + one-hot encodes categorical columns
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False needs scikit-learn >= 1.2.
        # On older versions replace with: OneHotEncoder(handle_unknown="ignore", sparse=False)
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Main entry point used by train_models.py.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw dataset (any CSV loaded into a DataFrame).
    target_column : str
        Name of the column the user picked as the prediction target.

    Returns
    -------
    dict with:
        X_train, X_test  : preprocessed feature matrices (numpy arrays)
        y_train, y_test   : target arrays
        problem_type      : "classification" or "regression"
        preprocessor      : fitted ColumnTransformer (needed later for
                             SHAP and for transforming new rows)
        feature_names     : list of feature names after encoding
        target_encoder    : fitted LabelEncoder, or None for regression
        dropped_columns   : columns removed during cleaning
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    df = df.copy()

    # Rows with a missing target can't be used for training/testing.
    df = df.dropna(subset=[target_column])

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    problem_type = detect_problem_type(y_raw)

    X = _coerce_numeric_like_columns(X)
    X, dropped_columns = _clean_columns(X)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    # Build readable feature names (needed later for SHAP plots).
    feature_names = list(numeric_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names += list(cat_encoder.get_feature_names_out(categorical_cols))

    target_encoder = None
    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)
    else:
        y = y_raw.to_numpy()

    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "problem_type": problem_type,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "target_encoder": target_encoder,
        "dropped_columns": dropped_columns,
    }