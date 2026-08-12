"""
analysis.py

General-purpose dataset analysis for the Dataset Doctor project.

Turns Person 1's original exploratory script into a reusable function
that returns a structured dictionary instead of only printing to the
console. This is what both the interface and the "AI assistant" module
(Person 3) will call to get facts about the uploaded CSV.

Nothing here is tied to a specific dataset.
"""

import numpy as np
import pandas as pd


def _classify_column(series):
    """Give a human-readable description of a column's type."""
    n_unique = series.nunique(dropna=True)
    dtype = series.dtype

    if not pd.api.types.is_numeric_dtype(series):
        return "Text / Categorical"
    if n_unique == 2:
        return "Numeric (Boolean-like)"
    if n_unique <= 10:
        return "Numeric (Ordinal-like)"
    return "Numeric"


def _quality_scores(df, missing_total, duplicate_pct):
    """
    Compute each quality sub-score FROM THE DATA, instead of hard-coding
    a label. Each one also returns a numeric penalty so the overall
    score is consistent with what's displayed.
    """
    n_rows, n_cols = df.shape
    penalties = {}

    # --- Missing values ---
    missing_pct = (missing_total / (n_rows * n_cols) * 100) if n_rows and n_cols else 0
    if missing_pct == 0:
        missing_status, missing_penalty = "Excellent", 0
    elif missing_pct < 2:
        missing_status, missing_penalty = "Good", 5
    elif missing_pct < 10:
        missing_status, missing_penalty = "Fair", 12
    else:
        missing_status, missing_penalty = "Poor", 20
    penalties["missing"] = missing_penalty

    # --- Duplicates ---
    if duplicate_pct > 5:
        duplicate_status, duplicate_penalty = "Poor", 15
    elif duplicate_pct > 1:
        duplicate_status, duplicate_penalty = "Fair", 8
    else:
        duplicate_status, duplicate_penalty = "Excellent", 0
    penalties["duplicates"] = duplicate_penalty

    # --- Data types (flag columns that are almost entirely one dtype
    #     mixed with a few stray values, a common messy-CSV symptom) ---
    mixed_type_cols = 0
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        types_seen = sample.map(type).nunique()
        if types_seen > 1:
            mixed_type_cols += 1
    if mixed_type_cols == 0:
        data_type_status, data_type_penalty = "Excellent", 0
    elif mixed_type_cols <= 2:
        data_type_status, data_type_penalty = "Fair", 5
    else:
        data_type_status, data_type_penalty = "Poor", 10
    penalties["data_types"] = data_type_penalty

    # --- Correlation (redundant numeric columns) ---
    numeric_df = df.select_dtypes(include="number")
    high_corr_pairs = []
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        for i in range(len(corr.columns)):
            for j in range(i):
                value = corr.iloc[i, j]
                if pd.notna(value) and abs(value) > 0.90:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], round(value, 3)))
    if not high_corr_pairs:
        correlation_status, correlation_penalty = "Good", 0
    elif len(high_corr_pairs) <= 2:
        correlation_status, correlation_penalty = "Fair", 5
    else:
        correlation_status, correlation_penalty = "Poor", 10
    penalties["correlation"] = correlation_penalty

    # --- Outliers (IQR rule on numeric columns) ---
    outlier_cols = 0
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if len(col_data) < 4:
            continue
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_ratio = ((col_data < lower) | (col_data > upper)).mean()
        if outlier_ratio > 0.05:
            outlier_cols += 1
    if outlier_cols == 0:
        outlier_status, outlier_penalty = "Excellent", 0
    elif outlier_cols <= 2:
        outlier_status, outlier_penalty = "Fair", 5
    else:
        outlier_status, outlier_penalty = "Poor", 10
    penalties["outliers"] = outlier_penalty

    score = max(0, 100 - sum(penalties.values()))

    return {
        "score": score,
        "missing_status": missing_status,
        "duplicate_status": duplicate_status,
        "data_type_status": data_type_status,
        "correlation_status": correlation_status,
        "outlier_status": outlier_status,
        "high_correlation_pairs": high_corr_pairs,
    }


def analyze_dataset(df, target_column=None):
    """
    Analyze any dataframe and return a structured summary.

    Parameters
    ----------
    df : pandas.DataFrame
    target_column : str, optional
        If provided, also computes target balance. If not provided,
        that part is simply skipped (the caller can run this before
        the user has picked a target).

    Returns
    -------
    dict with:
        n_rows, n_columns, n_duplicates, n_missing
        quality        : dict from _quality_scores()
        columns        : {column_name: description}
        target_balance : dict or None (only if target_column given)
    """
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())
    n_missing = int(df.isnull().sum().sum())
    duplicate_pct = (n_duplicates / n_rows * 100) if n_rows else 0

    quality = _quality_scores(df, n_missing, duplicate_pct)

    columns = {col: _classify_column(df[col]) for col in df.columns}

    target_balance = None
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        counts = df[target_column].value_counts(normalize=True)
        target_balance = {
            "is_imbalanced": bool(counts.max() > 0.80) if len(counts) else False,
            "class_proportions": counts.round(4).to_dict(),
        }

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "n_duplicates": n_duplicates,
        "n_missing": n_missing,
        "quality": quality,
        "columns": columns,
        "target_balance": target_balance,
    }


def print_report(analysis):
    """Console-friendly version of the old script's output, for quick local testing."""
    print("\n## Basic Dataset Information")
    print(f"Number of rows      : {analysis['n_rows']:,}")
    print(f"Number of columns   : {analysis['n_columns']}")
    print(f"Duplicate rows      : {analysis['n_duplicates']:,}")
    print(f"Missing values      : {analysis['n_missing']}")

    q = analysis["quality"]
    print(f"\n## Data Quality Score\n{q['score']} / 100\n")
    print(f"Missing values : {q['missing_status']}")
    print(f"Duplicate rows : {q['duplicate_status']}")
    print(f"Data types     : {q['data_type_status']}")
    print(f"Correlation    : {q['correlation_status']}")
    print(f"Outliers       : {q['outlier_status']}")

    print("\n## Columns and Data Types")
    for col, desc in analysis["columns"].items():
        print(f"{col:<20} {desc}")

    print("\n## Similar Columns")
    if q["high_correlation_pairs"]:
        for a, b, v in q["high_correlation_pairs"]:
            print(f"{a} <-> {b}  (corr={v})")
    else:
        print("None")

    if analysis["target_balance"] is not None:
        print("\n## Target Balance")
        tb = analysis["target_balance"]
        print("The dataset is imbalanced." if tb["is_imbalanced"] else "Target classes are balanced.")


if __name__ == "__main__":
    # Manual test block only.
    csv_path = input("Enter the path to your CSV file: ")
    df = pd.read_csv(csv_path)

    print(f"\nAnalyzing: {csv_path.split('/')[-1]}")
    analysis = analyze_dataset(df)
    print_report(analysis)

    print("\nAvailable Columns:")
    for column in df.columns:
        print("-", column)
    target_column = input("\nEnter the target column: ")

    analysis = analyze_dataset(df, target_column=target_column)
    print("\n## Target Balance")
    tb = analysis["target_balance"]
    print("The dataset is imbalanced." if tb["is_imbalanced"] else "Target classes are balanced.")

    print("\nAnalysis completed successfully!")
    print(f"Target Column: {target_column}")