from pathlib import Path

import pandas as pd

from analysis.analyzer import analyze_dataset
from explainability.shap_explainer import explain_best_model
from machine_learning.train_models import train_models
from reports.pdf_report import generate_pdf_report


TEST_SAMPLE_SIZE = 5000


def load_dataset():
    """
    Ask the user for a CSV file and load it safely.
    """

    csv_path = input(
        "\nEnter CSV path: "
    ).strip().strip('"').strip("'")

    file_path = Path(csv_path)

    if not file_path.exists():
        print("\nERROR: The CSV file could not be found.")
        return None, None

    if file_path.suffix.lower() != ".csv":
        print("\nERROR: Please select a CSV file.")
        return None, None

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

    except Exception as error:
        print("\nERROR: The CSV could not be loaded.")
        print(f"Reason: {error}")
        return None, None

    if df.empty:
        print("\nERROR: The uploaded CSV is empty.")
        return None, None

    print("\nDataset loaded successfully.")

    print(
        f"Original dataset size: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    return df, file_path


def prepare_test_sample(df):
    """
    Use a smaller sample when testing large datasets.
    """

    if len(df) > TEST_SAMPLE_SIZE:
        df = df.sample(
            n=TEST_SAMPLE_SIZE,
            random_state=42,
        ).copy()

        print("\nLarge dataset detected.")

        print(
            f"Using {TEST_SAMPLE_SIZE:,} rows "
            f"for this test."
        )

    return df


def choose_target(df):
    """
    Display the available columns and ask the user
    to select a valid target column.
    """

    print("\nAvailable columns:")

    for column in df.columns:
        print("-", column)

    target_column = input(
        "\nEnter target column: "
    ).strip()

    if target_column not in df.columns:
        print(
            f"\nERROR: '{target_column}' "
            f"is not a valid column."
        )
        return None

    target_data = df[target_column].dropna()

    if target_data.empty:
        print(
            "\nERROR: The selected target column "
            "contains no usable values."
        )
        return None

    if target_data.nunique() < 2:
        print(
            "\nERROR: The selected target must contain "
            "at least two different values."
        )
        return None

    return target_column


def run_dataset_doctor(
    df,
    dataset_name,
    target_column,
):
    """
    Run dataset analysis, machine learning,
    SHAP explainability, and PDF generation.
    """

    # ---------------------------------------------------------
    # Dataset analysis
    # ---------------------------------------------------------

    print("\nRunning dataset analysis...")

    analysis_output = analyze_dataset(
        df,
        target_column=target_column,
    )

    # ---------------------------------------------------------
    # Machine learning
    # ---------------------------------------------------------

    print("Training machine-learning models...")

    training_output = train_models(
        df,
        target_column,
    )

    print(
        f"\nProblem type: "
        f"{training_output['problem_type'].capitalize()}"
    )

    print("\nModel Results:")

    for model_name, model_info in training_output["results"].items():
        print(f"\n{model_name}")

        for metric_name, metric_value in model_info["metrics"].items():
            print(
                f"  {metric_name}: "
                f"{metric_value:.4f}"
            )

    print(
        f"\nBest model: "
        f"{training_output['best_model_name']}"
    )

    # ---------------------------------------------------------
    # SHAP explainability
    # ---------------------------------------------------------

    print("\nRunning SHAP...")

    shap_output = explain_best_model(
        training_output
    )

    print("\nTop SHAP features:")

    print(
        shap_output["feature_importance"].head(10)
    )

    # ---------------------------------------------------------
    # Report name
    # ---------------------------------------------------------

    report_name = input(
        "\nEnter a name for the PDF report: "
    ).strip()

    if not report_name:
        report_name = (
            f"{Path(dataset_name).stem} Report"
        )

    # ---------------------------------------------------------
    # Generate PDF
    # ---------------------------------------------------------

    print("\nGenerating PDF...")

    pdf_path = generate_pdf_report(
        analysis=analysis_output,
        target_column=target_column,
        training_output=training_output,
        shap_output=shap_output,
        dataset_name=dataset_name,
        report_name=report_name,
    )

    print("\nPDF generated successfully:")
    print(pdf_path)


def main():
    """
    Allow the user to analyse multiple CSV datasets
    during the same program session.
    """

    while True:

        # -----------------------------------------------------
        # Select dataset
        # -----------------------------------------------------

        print("\nSelect a CSV dataset to analyse.")

        df, file_path = load_dataset()

        if df is not None:

            print("\n====================================")
            print("          DATASET ANALYSIS")
            print("====================================")
            print(f"Dataset: {file_path.name}")

            # -------------------------------------------------
            # Prepare test sample
            # -------------------------------------------------

            df = prepare_test_sample(df)

            # -------------------------------------------------
            # Select target
            # -------------------------------------------------

            target_column = choose_target(df)

            if target_column is not None:

                try:
                    run_dataset_doctor(
                        df=df,
                        dataset_name=file_path.name,
                        target_column=target_column,
                    )

                except ValueError as error:
                    print(
                        "\nThe analysis could not be completed."
                    )
                    print(f"Reason: {error}")

                except MemoryError:
                    print(
                        "\nThe dataset requires more memory "
                        "than is currently available."
                    )
                    print(
                        "Try a smaller dataset or sample."
                    )

                except Exception as error:
                    print(
                        "\nAn unexpected error occurred."
                    )
                    print(f"Reason: {error}")

        # -----------------------------------------------------
        # Analyse another dataset?
        # -----------------------------------------------------

        again = input(
            "\nWould you like to analyse "
            "another dataset? (y/n): "
        ).strip().lower()

        if again not in {"y", "yes"}:
            print("\nAnalysis session closed.")
            break


if __name__ == "__main__":
    main()