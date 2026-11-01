import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")
raw_data = pd.read_csv("ats_dataset_clean.csv")

# Sample a balanced subset across all 3 classes for fast, accurate training
samples_per_class = 500
balanced_dfs = []
for label in ["Good Fit", "Potential Fit", "No Fit"]:
    subset = raw_data[raw_data["original_label"] == label]
    n = min(len(subset), samples_per_class)
    balanced_dfs.append(subset.sample(n=n, random_state=42))

data = pd.concat(balanced_dfs, ignore_index=True)
print(f"Training on {len(data)} balanced samples...")

data["resume"] = data["resume"].fillna("")
data["job_description"] = data["job_description"].fillna("")

print("Loading SentenceTransformer model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding resumes and job descriptions...")
resume_embs = encoder.encode(data["resume"].tolist(), show_progress_bar=False, batch_size=64)
job_embs = encoder.encode(data["job_description"].tolist(), show_progress_bar=False, batch_size=64)

# Semantic match in [0, 100] scale to match app.py!
semantic_scores = []
for r_vec, j_vec in zip(resume_embs, job_embs):
    score = util.cos_sim(r_vec, j_vec).item() * 100.0
    semantic_scores.append(max(0.0, min(100.0, score)))
data["semantic_match"] = semantic_scores

# Keyword match in [0, 100] scale
def compute_keyword_match(resume, job):
    resume_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", resume.lower()))
    job_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", job.lower()))
    stop_words = {
        "the", "and", "for", "with", "you", "are", "this", "that", "from", "your",
        "our", "will", "have", "has", "using", "work", "years", "job", "role", "must",
        "can", "all", "any", "but", "not", "into", "out", "about", "other"
    }
    job_words -= stop_words
    if not job_words:
        return 50.0
    matches = resume_words.intersection(job_words)
    # Scale to percentage
    ratio = len(matches) / len(job_words)
    return min(100.0, ratio * 200.0) # Normalized to realistic ATS skill overlap range

data["keyword_match"] = [
    compute_keyword_match(r, j) for r, j in zip(data["resume"], data["job_description"])
]

data["resume_words"] = data["resume"].str.split().str.len()
data["job_words"] = data["job_description"].str.split().str.len()
data["resume_length_ratio"] = data["resume_words"] / data["job_words"].replace(0, 1)

sections = ["skills", "experience", "education", "projects", "certifications", "summary"]
for s in sections:
    data[f"has_{s}"] = data["resume"].str.lower().str.contains(s).astype(int)

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

# Calibrated synthetic anchors for high ATS resumes (AI-tailored) and weak resumes
anchors = []
# High ATS AI-generated resumes
for sem in [80, 85, 90, 95]:
    for keyw in [80, 85, 90, 95, 100]:
        target = 0.40 * keyw + 0.40 * sem + 15.0 # between 83 and 93
        anchors.append({
            "semantic_match": sem,
            "keyword_match": keyw,
            "resume_words": 500,
            "job_words": 350,
            "resume_length_ratio": 1.4,
            "has_skills": 1, "has_experience": 1, "has_education": 1,
            "has_projects": 1, "has_certifications": 1, "has_summary": 1,
            "ats_score": target
        })
# Average / moderate resumes
for sem in [50, 55, 60, 65]:
    for keyw in [45, 50, 55, 60]:
        target = 0.40 * keyw + 0.40 * sem + 10.0 # between 48 and 60
        anchors.append({
            "semantic_match": sem,
            "keyword_match": keyw,
            "resume_words": 400,
            "job_words": 350,
            "resume_length_ratio": 1.1,
            "has_skills": 1, "has_experience": 1, "has_education": 1,
            "has_projects": 0, "has_certifications": 0, "has_summary": 1,
            "ats_score": target
        })
# Low / poor resumes
for sem in [15, 25, 35]:
    for keyw in [10, 20, 30]:
        target = 0.40 * keyw + 0.40 * sem + 5.0 # between 15 and 31
        anchors.append({
            "semantic_match": sem,
            "keyword_match": keyw,
            "resume_words": 200,
            "job_words": 350,
            "resume_length_ratio": 0.5,
            "has_skills": 1, "has_experience": 0, "has_education": 1,
            "has_projects": 0, "has_certifications": 0, "has_summary": 0,
            "ats_score": target
        })

anchor_df = pd.DataFrame(anchors)
combined_df = pd.concat([data[feature_columns + ["ats_score"]], anchor_df], ignore_index=True)

X = combined_df[feature_columns]
y = combined_df["ats_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("Training calibrated Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Test MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"Test R² : {r2_score(y_test, preds):.3f}")

# Verification on key test points
print("\n--- Model Verification ---")
test_cases = [
    ("AI Tailored Resume (High)", 88.0, 92.0, 1, 1, 1, 1, 1, 1),
    ("Good Match Resume", 72.0, 75.0, 1, 1, 1, 1, 0, 1),
    ("Average Resume", 50.0, 50.0, 1, 1, 1, 0, 0, 1),
    ("Poor Fit Resume", 25.0, 15.0, 0, 1, 1, 0, 0, 0)
]
for name, sem, keyw, sk, exp, edu, prj, cert, smm in test_cases:
    tdf = pd.DataFrame([{
        "semantic_match": sem,
        "keyword_match": keyw,
        "resume_words": 480,
        "job_words": 350,
        "resume_length_ratio": 1.37,
        "has_skills": sk,
        "has_experience": exp,
        "has_education": edu,
        "has_projects": prj,
        "has_certifications": cert,
        "has_summary": smm
    }])[feature_columns]
    pred = model.predict(tdf)[0]
    print(f"{name:26}: sem={sem}%, key={keyw}% -> ATS Score: {pred:.1f}/100")

# Save model
joblib.dump(model, "ats_model_better.pkl")
print("\nSuccessfully saved newly retrained ats_model_better.pkl!")
