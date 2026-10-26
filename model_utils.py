import pandas as pd


TEXT_COLUMNS = [
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
NUMERIC_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
FEATURE_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMN = "fraudulent"


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    text = features[TEXT_COLUMNS].fillna("").astype(str)
    features["combined_text"] = text.agg(" ".join, axis=1)
    return features[["combined_text", *NUMERIC_COLUMNS]]
