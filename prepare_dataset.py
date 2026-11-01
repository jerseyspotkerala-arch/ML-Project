from datasets import load_dataset
import pandas as pd
import re

print("Downloading ATS dataset...")

dataset = load_dataset(
    "0xnbk/resume-ats-score-v1-en"
)

train = pd.DataFrame(dataset["train"])
validation = pd.DataFrame(dataset["validation"])

data = pd.concat(
    [train, validation],
    ignore_index=True
)

print("Original rows:", len(data))


# -----------------------------
# CLEAN TEXT
# -----------------------------

data["text"] = (
    data["text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# -----------------------------
# REMOVE EMPTY DATA
# -----------------------------

data = data[
    data["text"].str.len() > 100
]


# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

data = data.drop_duplicates(
    subset=["text"]
)


# -----------------------------
# CLEAN SCORES
# -----------------------------

data["ats_score"] = pd.to_numeric(
    data["ats_score"],
    errors="coerce"
)

data = data[
    data["ats_score"].between(0, 100)
]


# -----------------------------
# SPLIT RESUME / JOB
# -----------------------------

parts = data["text"].str.split(
    " SEP ",
    n=1,
    expand=True
)

data["resume"] = parts[0]
data["job_description"] = parts[1]

data = data.dropna(
    subset=["resume", "job_description"]
)


# -----------------------------
# FINAL DATASET
# -----------------------------

final_data = data[
    [
        "resume",
        "job_description",
        "ats_score",
        "original_label"
    ]
]

final_data.to_csv(
    "ats_dataset_clean.csv",
    index=False
)

print("\nCleaning complete!")
print("Final rows:", len(final_data))
print("\nLabels:")
print(final_data["original_label"].value_counts())

print("\nScore statistics:")
print(final_data["ats_score"].describe())