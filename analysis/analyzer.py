"""
analyzer.py

General-purpose dataset analysis for the Dataset Doctor project.

This module analyses any uploaded CSV and returns structured
information that can be used by:
- the user interface
- machine learning
- the PDF report

Nothing here is tied to a specific dataset.
"""

import pandas as pd


def _classify_column(series):
    """Return a human-readable description of a column's data type."""

    n_unique = series.nunique(dropna=True)

    if not pd.api.types.is_numeric_dtype(series):
        return "Text / Categorical"

    if n_unique == 2:
        return "Numeric (Boolean-like)"

    if n_unique <= 10:
        return "Numeric (Ordinal-like)"

    return "Numeric"


def _quality_scores(df, missing_total, duplicate_pct):
    """
    Calculate data-quality scores.

    Checks:
    - missing values
    - duplicate rows
    - mixed data types
    - highly correlated numeric columns
    - outliers
    """

    n_rows, n_cols = df.shape
    penalties = {}

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    if n_rows > 0 and n_cols > 0:
        missing_pct = (
            missing_total / (n_rows * n_cols)
        ) * 100
    else:
        missing_pct = 0

    if missing_pct == 0:
        missing_status = "Excellent"
        missing_penalty = 0

    elif missing_pct < 2:
        missing_status = "Good"
        missing_penalty = 5

    elif missing_pct < 10:
        missing_status = "Fair"
        missing_penalty = 12

    else:
        missing_status = "Poor"
        missing_penalty = 20

    penalties["missing"] = missing_penalty

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    if duplicate_pct > 5:
        duplicate_status = "Poor"
        duplicate_penalty = 15

    elif duplicate_pct > 1:
        duplicate_status = "Fair"
        duplicate_penalty = 8

    else:
        duplicate_status = "Excellent"
        duplicate_penalty = 0

    penalties["duplicates"] = duplicate_penalty

    # ---------------------------------------------------------
    # Data types
    # ---------------------------------------------------------

    mixed_type_cols = 0

    for col in df.columns:

        sample = df[col].dropna()

        if len(sample) == 0:
            continue

        types_seen = sample.map(type).nunique()

        if types_seen > 1:
            mixed_type_cols += 1

    if mixed_type_cols == 0:
        data_type_status = "Excellent"
        data_type_penalty = 0

    elif mixed_type_cols <= 2:
        data_type_status = "Fair"
        data_type_penalty = 5

    else:
        data_type_status = "Poor"
        data_type_penalty = 10

    penalties["data_types"] = data_type_penalty

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    high_corr_pairs = []

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr(
            numeric_only=True
        )

        for i in range(len(corr.columns)):

            for j in range(i):

                value = corr.iloc[i, j]

                if (
                    pd.notna(value)
                    and abs(value) > 0.90
                ):
                    high_corr_pairs.append(
                        (
                            corr.columns[i],
                            corr.columns[j],
                            round(value, 3),
                        )
                    )

    if not high_corr_pairs:
        correlation_status = "Good"
        correlation_penalty = 0

    elif len(high_corr_pairs) <= 2:
        correlation_status = "Fair"
        correlation_penalty = 5

    else:
        correlation_status = "Poor"
        correlation_penalty = 10

    penalties["correlation"] = correlation_penalty

    # ---------------------------------------------------------
    # Outliers
    # ---------------------------------------------------------

    outlier_cols = 0

    for col in numeric_df.columns:

        col_data = numeric_df[col].dropna()

        if len(col_data) < 4:
            continue

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outlier_ratio = (
            (col_data < lower)
            | (col_data > upper)
        ).mean()

        if outlier_ratio > 0.05:
            outlier_cols += 1

    if outlier_cols == 0:
        outlier_status = "Excellent"
        outlier_penalty = 0

    elif outlier_cols <= 2:
        outlier_status = "Fair"
        outlier_penalty = 5

    else:
        outlier_status = "Poor"
        outlier_penalty = 10

    penalties["outliers"] = outlier_penalty

    # ---------------------------------------------------------
    # Overall quality score
    # ---------------------------------------------------------

    score = max(
        0,
        100 - sum(penalties.values()),
    )

    return {
        "score": score,
        "missing_status": missing_status,
        "duplicate_status": duplicate_status,
        "data_type_status": data_type_status,
        "correlation_status": correlation_status,
        "outlier_status": outlier_status,
        "high_correlation_pairs": high_corr_pairs,
    }


def analyze_dataset(
    df,
    target_column=None,
):
    """
    Analyse any dataframe and return a structured summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to analyse.

    target_column : str, optional
        Target selected by the user.

    Returns
    -------
    dict
        Dataset analysis results.
    """

    n_rows, n_cols = df.shape

    n_duplicates = int(
        df.duplicated().sum()
    )

    n_missing = int(
        df.isnull().sum().sum()
    )

    if n_rows > 0:
        duplicate_pct = (
            n_duplicates / n_rows
        ) * 100
    else:
        duplicate_pct = 0

    # Calculate quality information.
    quality = _quality_scores(
        df,
        n_missing,
        duplicate_pct,
    )

    # Describe each column.
    columns = {
        col: _classify_column(df[col])
        for col in df.columns
    }

    # ---------------------------------------------------------
    # Target balance
    # ---------------------------------------------------------

    target_balance = None

    if target_column is not None:

        if target_column not in df.columns:
            raise ValueError(
                f"Target column "
                f"'{target_column}' "
                f"not found."
            )

        counts = df[
            target_column
        ].value_counts(
            normalize=True
        )

        if len(counts) > 0:
            is_imbalanced = bool(
                counts.max() > 0.80
            )
        else:
            is_imbalanced = False

        target_balance = {
            "is_imbalanced": is_imbalanced,
            "class_proportions": (
                counts
                .round(4)
                .to_dict()
            ),
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
    """
    Print dataset analysis results to the terminal.

    This is mainly useful for testing.
    """

    print("\n## Basic Dataset Information")

    print(
        f"Number of rows      : "
        f"{analysis['n_rows']:,}"
    )

    print(
        f"Number of columns   : "
        f"{analysis['n_columns']}"
    )

    print(
        f"Duplicate rows      : "
        f"{analysis['n_duplicates']:,}"
    )

    print(
        f"Missing values      : "
        f"{analysis['n_missing']:,}"
    )

    # ---------------------------------------------------------
    # Data quality
    # ---------------------------------------------------------

    quality = analysis["quality"]

    print("\n## Data Quality Score")

    print(
        f"{quality['score']} / 100"
    )

    print(
        f"Missing values : "
        f"{quality['missing_status']}"
    )

    print(
        f"Duplicate rows : "
        f"{quality['duplicate_status']}"
    )

    print(
        f"Data types     : "
        f"{quality['data_type_status']}"
    )

    print(
        f"Correlation    : "
        f"{quality['correlation_status']}"
    )

    print(
        f"Outliers       : "
        f"{quality['outlier_status']}"
    )

    # ---------------------------------------------------------
    # Columns
    # ---------------------------------------------------------

    print("\n## Columns and Data Types")

    for col, description in analysis[
        "columns"
    ].items():

        print(
            f"{col:<30} "
            f"{description}"
        )

    # ---------------------------------------------------------
    # Highly correlated columns
    # ---------------------------------------------------------

    print("\n## Similar Columns")

    correlation_pairs = quality[
        "high_correlation_pairs"
    ]

    if correlation_pairs:

        for column_a, column_b, value in correlation_pairs:

            print(
                f"{column_a} <-> "
                f"{column_b} "
                f"(corr={value})"
            )

    else:
        print("None")

    # ---------------------------------------------------------
    # Target balance
    # ---------------------------------------------------------

    target_balance = analysis.get(
        "target_balance"
    )

    if target_balance is not None:

        print("\n## Target Balance")

        if target_balance["is_imbalanced"]:
            print(
                "The dataset is imbalanced."
            )
        else:
            print(
                "Target classes are balanced."
            )


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Manual testing
    # ---------------------------------------------------------

    csv_path = input(
        "Enter the path to your CSV file: "
    )

    # Remove accidental quotation marks from Windows paths.
    csv_path = csv_path.strip().strip('"').strip("'")

    print("\nLoading dataset...")

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    print("\nDataset loaded successfully.")

    # Run initial analysis.
    analysis = analyze_dataset(df)

    print_report(analysis)

    # ---------------------------------------------------------
    # Target selection
    # ---------------------------------------------------------

    print("\nAvailable Columns:")

    for column in df.columns:
        print(
            "-",
            column,
        )

    target_column = input(
        "\nEnter the target column: "
    ).strip()

    # Run analysis again with target information.
    analysis = analyze_dataset(
        df,
        target_column=target_column,
    )

    print("\n## Target Balance")

    target_balance = analysis[
        "target_balance"
    ]

    if target_balance["is_imbalanced"]:
        print(
            "The dataset is imbalanced."
        )
    else:
        print(
            "Target classes are balanced."
        )

    print(
        "\nAnalysis completed successfully!"
    )

    print(
        f"Target Column: "
        f"{target_column}"
    )