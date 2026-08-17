"""
preprocessing.py

General-purpose preprocessing for the Dataset Doctor project.

Given any dataframe and a target column, this module:
- detects whether the problem is classification or regression
- removes unsuitable columns
- converts numeric-like text columns
- handles infinite values
- imputes missing values
- scales numeric features
- one-hot encodes categorical features
- splits the data into train/test sets

Nothing in this file is specific to a particular dataset.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


def detect_problem_type(
    y,
    classification_max_unique=20,
):
    """
    Detect whether the target represents classification or regression.

    Rules:
    1. Non-numeric targets -> classification.
    2. Numeric targets with only a small number of unique values
       -> classification.
    3. Numeric targets with many distinct values -> regression.

    This prevents continuous numeric targets such as age, price,
    income, or charges from being incorrectly treated as classes.
    """

    y = pd.Series(y).dropna()

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n_unique = y.nunique()

    if n_unique <= classification_max_unique:
        return "classification"

    return "regression"


def _coerce_numeric_like_columns(
    X,
    min_success_ratio=0.95,
):
    """
    Convert text columns to numeric when most values are numeric-like.

    Invalid values become NaN and are handled later by the imputer.
    """

    X = X.copy()

    text_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns

    for col in text_columns:
        non_null = X[col].dropna()

        if len(non_null) == 0:
            continue

        converted = pd.to_numeric(
            non_null.astype(str).str.strip(),
            errors="coerce",
        )

        success_ratio = converted.notna().mean()

        if success_ratio >= min_success_ratio:
            X[col] = pd.to_numeric(
                X[col].astype(str).str.strip(),
                errors="coerce",
            )

    return X


def _clean_infinite_values(X):
    """
    Replace positive and negative infinity with NaN.

    NaN values are later handled by the imputation pipeline.
    """

    X = X.copy()

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns

    for col in numeric_columns:
        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return X


def _clean_columns(X):
    """
    Remove columns that are unsuitable for general machine learning.

    Removes:
    - constant columns
    - text ID-like columns with one unique value per row
    """

    cols_to_drop = []
    n_rows = len(X)

    for col in X.columns:
        series = X[col]
        n_unique = series.nunique(
            dropna=True
        )

        if n_unique <= 1:
            cols_to_drop.append(
                col
            )
            continue

        is_text_like = (
            not pd.api.types.is_numeric_dtype(
                series
            )
        )

        if (
            is_text_like
            and n_rows > 1
            and n_unique == n_rows
        ):
            cols_to_drop.append(
                col
            )

    cleaned_X = X.drop(
        columns=cols_to_drop
    )

    return (
        cleaned_X,
        cols_to_drop,
    )


def build_preprocessor(X):
    """
    Build the preprocessing pipeline.

    Numeric columns:
    - median imputation
    - standard scaling

    Categorical columns:
    - most-frequent imputation
    - one-hot encoding
    - infrequent-category grouping
    """

    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=50,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_cols,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    return (
        preprocessor,
        numeric_cols,
        categorical_cols,
    )


def prepare_data(
    df,
    target_column,
    test_size=0.2,
    random_state=42,
):
    """
    Prepare a dataframe for machine learning.

    Returns
    -------
    dict
        Contains:
        - X_train
        - X_test
        - y_train
        - y_test
        - problem_type
        - preprocessor
        - feature_names
        - target_encoder
        - dropped_columns
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column "
            f"'{target_column}' "
            f"not found in dataset."
        )

    df = df.copy()

    # Remove rows where target is missing.
    df = df.dropna(
        subset=[target_column]
    )

    y_raw = df[
        target_column
    ]

    X = df.drop(
        columns=[target_column]
    )

    # ---------------------------------------------------------
    # Detect problem type
    # ---------------------------------------------------------

    problem_type = detect_problem_type(
        y_raw
    )

    # ---------------------------------------------------------
    # Clean feature data
    # ---------------------------------------------------------

    X = _coerce_numeric_like_columns(
        X
    )

    X = _clean_infinite_values(
        X
    )

    (
        X,
        dropped_columns,
    ) = _clean_columns(
        X
    )

    # Ensure categorical columns use one consistent type.
    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns

    for col in categorical_columns:
        X[col] = X[col].astype(
            "string"
        )

    # ---------------------------------------------------------
    # Build and apply preprocessing
    # ---------------------------------------------------------

    (
        preprocessor,
        numeric_cols,
        categorical_cols,
    ) = build_preprocessor(
        X
    )

    X_processed = (
        preprocessor.fit_transform(
            X
        )
    )

    # ---------------------------------------------------------
    # Build readable feature names
    # ---------------------------------------------------------

    feature_names = list(
        numeric_cols
    )

    if categorical_cols:
        cat_encoder = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        encoded_names = (
            cat_encoder
            .get_feature_names_out(
                categorical_cols
            )
        )

        feature_names.extend(
            encoded_names.tolist()
        )

    # ---------------------------------------------------------
    # Prepare target
    # ---------------------------------------------------------

    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y_raw
        )

    else:
        y = pd.to_numeric(
            y_raw,
            errors="coerce",
        ).to_numpy()

        valid_target_mask = np.isfinite(
            y
        )

        X_processed = X_processed[
            valid_target_mask
        ]

        y = y[
            valid_target_mask
        ]

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    stratify = None

    if problem_type == "classification":
        class_counts = pd.Series(
            y
        ).value_counts()

        # Only stratify when every class has at least 2 examples.
        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        ):
            stratify = y

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_processed,
        y,
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