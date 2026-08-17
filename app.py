from pathlib import Path
from uuid import uuid4

import pandas as pd

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from analysis.analyzer import analyze_dataset
from explainability.shap_explainer import explain_best_model
from machine_learning.train_models import train_models
from reports.pdf_report import generate_pdf_report


app = Flask(__name__)

# Required for Flask sessions and flash messages.
app.secret_key = "datamedic-ai-development-key"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
REPORT_FOLDER = BASE_DIR / "reports" / "generated"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_EXTENSIONS = {
    "csv",
}

# During development, very large datasets are sampled
# so model training and SHAP remain manageable.
MAX_ANALYSIS_ROWS = 5000


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def allowed_file(filename):
    """
    Check whether the uploaded file is a CSV.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def load_csv(file_path):
    """
    Load a CSV safely.
    """

    return pd.read_csv(
        file_path,
        low_memory=False,
    )


def prepare_analysis_sample(df):
    """
    Use a manageable sample when a dataset is very large.

    The original uploaded dataset remains unchanged.
    """

    if len(df) > MAX_ANALYSIS_ROWS:
        return df.sample(
            n=MAX_ANALYSIS_ROWS,
            random_state=42,
        ).copy()

    return df.copy()


def serialise_model_results(training_output):
    """
    Convert model results into a template-friendly structure.
    """

    results = []

    for model_name, model_info in training_output["results"].items():
        results.append(
            {
                "name": model_name,
                "metrics": model_info["metrics"],
                "is_best": (
                    model_name
                    == training_output["best_model_name"]
                ),
            }
        )

    return results


# ---------------------------------------------------------
# Home / Upload
# ---------------------------------------------------------

@app.route(
    "/",
    methods=["GET", "POST"],
)
def index():
    """
    Upload a CSV dataset.
    """

    if request.method == "POST":

        if "dataset" not in request.files:
            flash(
                "Please choose a CSV file.",
                "error",
            )
            return redirect(
                url_for("index")
            )

        uploaded_file = request.files["dataset"]

        if uploaded_file.filename == "":
            flash(
                "Please choose a CSV file.",
                "error",
            )
            return redirect(
                url_for("index")
            )

        if not allowed_file(
            uploaded_file.filename
        ):
            flash(
                "Only CSV files are supported.",
                "error",
            )
            return redirect(
                url_for("index")
            )

        original_filename = secure_filename(
            uploaded_file.filename
        )

        unique_filename = (
            f"{uuid4().hex}_"
            f"{original_filename}"
        )

        saved_path = (
            UPLOAD_FOLDER
            / unique_filename
        )

        uploaded_file.save(
            saved_path
        )

        try:
            df = load_csv(
                saved_path
            )

        except Exception as error:
            saved_path.unlink(
                missing_ok=True
            )

            flash(
                f"The CSV could not be read: {error}",
                "error",
            )

            return redirect(
                url_for("index")
            )

        if df.empty:
            saved_path.unlink(
                missing_ok=True
            )

            flash(
                "The uploaded CSV is empty.",
                "error",
            )

            return redirect(
                url_for("index")
            )

        # Store only lightweight information in the Flask session.
        session["uploaded_file"] = str(
            saved_path
        )

        session["dataset_name"] = (
            original_filename
        )

        return redirect(
            url_for("select_target")
        )

    return render_template(
        "index.html"
    )


# ---------------------------------------------------------
# Target Selection
# ---------------------------------------------------------

@app.route(
    "/select-target",
    methods=["GET", "POST"],
)
def select_target():
    """
    Show dataset information and allow target selection.
    """

    uploaded_file = session.get(
        "uploaded_file"
    )

    if not uploaded_file:
        flash(
            "Please upload a dataset first.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    try:
        df = load_csv(
            uploaded_file
        )

    except Exception as error:
        flash(
            f"The dataset could not be loaded: {error}",
            "error",
        )

        return redirect(
            url_for("index")
        )

    dataset_name = session.get(
        "dataset_name",
        "Uploaded Dataset",
    )

    basic_analysis = analyze_dataset(
        df
    )

    preview = (
        df.head(10)
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    if request.method == "POST":

        target_column = request.form.get(
            "target_column",
            "",
        ).strip()

        if target_column not in df.columns:
            flash(
                "Please select a valid target column.",
                "error",
            )

            return redirect(
                url_for("select_target")
            )

        target_data = df[
            target_column
        ].dropna()

        if target_data.empty:
            flash(
                "The selected target column contains no usable values.",
                "error",
            )

            return redirect(
                url_for("select_target")
            )

        if target_data.nunique() < 2:
            flash(
                "The target column must contain at least two different values.",
                "error",
            )

            return redirect(
                url_for("select_target")
            )

        session[
            "target_column"
        ] = target_column

        return redirect(
            url_for("results")
        )

    return render_template(
        "select.html",
        dataset_name=dataset_name,
        columns=df.columns.tolist(),
        analysis=basic_analysis,
        preview=preview,
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

@app.route(
    "/results",
    methods=["GET"],
)
def results():
    """
    Run analysis, machine learning, SHAP,
    and display the final results.
    """

    uploaded_file = session.get(
        "uploaded_file"
    )

    target_column = session.get(
        "target_column"
    )

    dataset_name = session.get(
        "dataset_name",
        "Uploaded Dataset",
    )

    if not uploaded_file or not target_column:
        flash(
            "Please upload a dataset and select a target first.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    try:
        original_df = load_csv(
            uploaded_file
        )

        analysis_df = prepare_analysis_sample(
            original_df
        )

        analysis_output = analyze_dataset(
            analysis_df,
            target_column=target_column,
        )

        training_output = train_models(
            analysis_df,
            target_column,
        )

        shap_output = explain_best_model(
            training_output
        )

    except Exception as error:
        return render_template(
            "error.html",
            error_message=str(
                error
            ),
        )

    model_results = serialise_model_results(
        training_output
    )

    top_features = (
        shap_output[
            "feature_importance"
        ]
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    # Store only what is needed for regenerating the report.
    session["last_analysis_ready"] = True

    return render_template(
        "results.html",
        dataset_name=dataset_name,
        target_column=target_column,
        analysis=analysis_output,
        training=training_output,
        model_results=model_results,
        top_features=top_features,
        shap_plot=shap_output[
            "summary_plot_path"
        ],
        sampled_rows=len(
            analysis_df
        ),
        original_rows=len(
            original_df
        ),
    )


# ---------------------------------------------------------
# PDF Download
# ---------------------------------------------------------

@app.route(
    "/download-report",
    methods=["POST"],
)
def download_report():
    """
    Generate and download a PDF report.
    """

    uploaded_file = session.get(
        "uploaded_file"
    )

    target_column = session.get(
        "target_column"
    )

    dataset_name = session.get(
        "dataset_name",
        "Uploaded Dataset",
    )

    if not uploaded_file or not target_column:
        flash(
            "Please run an analysis first.",
            "error",
        )

        return redirect(
            url_for("index")
        )

    report_name = request.form.get(
        "report_name",
        "",
    ).strip()

    if not report_name:
        report_name = (
            f"{Path(dataset_name).stem} Report"
        )

    try:
        original_df = load_csv(
            uploaded_file
        )

        analysis_df = prepare_analysis_sample(
            original_df
        )

        analysis_output = analyze_dataset(
            analysis_df,
            target_column=target_column,
        )

        training_output = train_models(
            analysis_df,
            target_column,
        )

        shap_output = explain_best_model(
            training_output
        )

        pdf_path = generate_pdf_report(
            analysis=analysis_output,
            target_column=target_column,
            training_output=training_output,
            shap_output=shap_output,
            dataset_name=dataset_name,
            report_name=report_name,
        )

    except Exception as error:
        return render_template(
            "error.html",
            error_message=str(
                error
            ),
        )

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=Path(
            pdf_path
        ).name,
    )


# ---------------------------------------------------------
# Start Flask
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
    )