from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import FEATURE_COLUMNS, NUMERIC_COLUMNS, TEXT_COLUMNS, prepare_features


MODEL_PATH = Path(__file__).with_name("model.joblib")
LONG_TEXT_COLUMNS = {"company_profile", "description", "requirements", "benefits"}


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


st.title("Fake Job Posting Checker")
st.write(
    "Enter the details of a job posting to estimate whether it may be fraudulent. "
    "This is a screening aid, not a definitive judgment."
)

if not MODEL_PATH.is_file():
    st.error("The trained model file is missing. Train it first using the command shown below.")
    st.code(
        '.\\.venv\\Scripts\\python.exe train_model.py "C:\\path\\to\\fake_job_postings.csv"',
        language="powershell",
    )
    st.stop()

artifact = load_model(MODEL_PATH)

with st.form("job_posting_form"):
    st.subheader("Job posting details")
    submitted_values: dict[str, object] = {}

    for column in TEXT_COLUMNS:
        label = column.replace("_", " ").title()
        if column in LONG_TEXT_COLUMNS:
            submitted_values[column] = st.text_area(label, height=120)
        else:
            submitted_values[column] = st.text_input(label)

    st.subheader("Additional signals")
    numeric_labels = {
        "telecommuting": "Offers telecommuting",
        "has_company_logo": "Has a company logo",
        "has_questions": "Includes screening questions",
    }
    for column in NUMERIC_COLUMNS:
        submitted_values[column] = int(st.checkbox(numeric_labels[column]))

    submitted = st.form_submit_button("Check posting")

if submitted:
    input_frame = pd.DataFrame(
        [{column: submitted_values[column] for column in FEATURE_COLUMNS}]
    )
    model_input = prepare_features(input_frame)
    prediction = int(artifact["pipeline"].predict(model_input)[0])
    probability = float(artifact["pipeline"].predict_proba(model_input)[0][1])

    if prediction:
        st.warning(f"The model flags this posting as potentially fake ({probability:.1%} estimate).")
    else:
        st.success(f"The model classifies this posting as likely real ({probability:.1%} estimate).")
