import pandas as pd

from machine_learning.train_models import train_models
from explainability.shap_explainer import explain_best_model


# Get CSV file path.
csv_path = input("Enter CSV path: ").strip().strip('"')

# Load the CSV.
df = pd.read_csv(csv_path)

print(f"\nOriginal dataset size: {df.shape[0]:,} rows")

# Use a smaller sample for development/testing.
# This does NOT change the final application.
TEST_SAMPLE_SIZE = 5000

if len(df) > TEST_SAMPLE_SIZE:
    df = df.sample(
        n=TEST_SAMPLE_SIZE,
        random_state=42,
    )

    print(
        f"Large dataset detected. "
        f"Using {TEST_SAMPLE_SIZE:,} rows for this test."
    )

print(
    f"Test dataset size: "
    f"{df.shape[0]:,} rows x {df.shape[1]} columns"
)


# Display available columns.
print("\nAvailable columns:")

for column in df.columns:
    print("-", column)


# Ask user for target.
target_column = input(
    "\nEnter target column: "
).strip()


print("\nTraining machine-learning models...")

training_output = train_models(
    df,
    target_column,
)


print("\nProblem type:")
print(
    training_output["problem_type"].capitalize()
)


print("\nModel results:")

for model_name, info in training_output["results"].items():
    print(f"\n{model_name}")

    for metric_name, metric_value in info["metrics"].items():
        print(
            f"  {metric_name}: "
            f"{metric_value:.4f}"
        )


print("\nBest model:")
print(
    training_output["best_model_name"]
)


print("\nRunning SHAP...")

shap_output = explain_best_model(
    training_output
)


print("\nTop 10 most important features:")

print(
    shap_output[
        "feature_importance"
    ].head(10)
)


print("\nSHAP plot saved to:")

print(
    shap_output[
        "summary_plot_path"
    ]
)