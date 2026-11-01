import pandas as pd
import numpy as np
import joblib
import re

from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Load data
# -----------------------------

data = pd.read_csv("ats_dataset_clean.csv")

print("Rows:", len(data))


# -----------------------------
# Clean text
# -----------------------------

data["resume"] = data["resume"].fillna("")
data["job_description"] = data["job_description"].fillna("")


# -----------------------------
# Load semantic model
# -----------------------------

print("Loading NLP model...")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Create separate embeddings
# -----------------------------

print("Encoding resumes...")

resume_embeddings = encoder.encode(
    data["resume"].tolist(),
    show_progress_bar=True
)

print("Encoding job descriptions...")

job_embeddings = encoder.encode(
    data["job_description"].tolist(),
    show_progress_bar=True
)


# -----------------------------
# Semantic similarity
# -----------------------------

semantic_scores = []

for resume_vec, job_vec in zip(
    resume_embeddings,
    job_embeddings
):

    score = util.cos_sim(
        resume_vec,
        job_vec
    ).item()

    semantic_scores.append(score)


data["semantic_match"] = semantic_scores


# -----------------------------
# Keyword overlap
# -----------------------------

def keyword_match(resume, job):

    resume_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
            resume.lower()
        )
    )

    job_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
            job.lower()
        )
    )

    # Ignore very common words
    stop_words = {
        "the", "and", "for", "with", "you",
        "are", "this", "that", "from", "your",
        "our", "will", "have", "has", "using",
        "work", "years", "job", "role"
    }

    job_words -= stop_words

    if not job_words:
        return 0

    matches = resume_words.intersection(job_words)

    return len(matches) / len(job_words)


data["keyword_match"] = [
    keyword_match(r, j)
    for r, j in zip(
        data["resume"],
        data["job_description"]
    )
]


# -----------------------------
# Basic resume features
# -----------------------------

data["resume_words"] = data["resume"].str.split().str.len()

data["job_words"] = (
    data["job_description"]
    .str.split()
    .str.len()
)

data["resume_length_ratio"] = (
    data["resume_words"] /
    data["job_words"].replace(0, 1)
)


# -----------------------------
# Section features
# -----------------------------

section_names = [
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
    "summary"
]

for section in section_names:

    data[f"has_{section}"] = (
        data["resume"]
        .str.lower()
        .str.contains(section)
        .astype(int)
    )


# -----------------------------
# Final ML features
# -----------------------------

feature_columns = [
    "semantic_match",
    "keyword_match",
    "resume_words",
    "job_words",
    "resume_length_ratio",
    "has_skills",
    "has_experience",
    "has_education",
    "has_projects",
    "has_certifications",
    "has_summary"
]

X = data[feature_columns]

y = data["ats_score"]


# -----------------------------
# Train / test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Random Forest
# -----------------------------

print("\nTraining model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n===== BETTER MODEL RESULTS =====")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))


# -----------------------------
# Feature importance
# -----------------------------

print("\n===== FEATURE IMPORTANCE =====")

for name, importance in sorted(
    zip(feature_columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):

    print(
        name,
        ":",
        round(importance, 3)
    )


# -----------------------------
# Save
# -----------------------------

joblib.dump(
    model,
    "ats_model_better.pkl"
)

print("\nSaved: ats_model_better.pkl")