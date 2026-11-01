import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# Load cleaned dataset
# --------------------------------

data = pd.read_csv("ats_dataset_clean.csv")

print("Dataset:", len(data), "rows")


# --------------------------------
# Combine resume + job description
# --------------------------------

data["combined_text"] = (
    data["resume"].fillna("") +
    " [JOB] " +
    data["job_description"].fillna("")
)


# --------------------------------
# Load NLP embedding model
# --------------------------------

print("Loading NLP model...")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# Create embeddings
# --------------------------------

print("Creating embeddings...")

embeddings = encoder.encode(
    data["combined_text"].tolist(),
    show_progress_bar=True
)


# --------------------------------
# Target
# --------------------------------

y = data["ats_score"]


# --------------------------------
# Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# Train ML model
# --------------------------------

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------
# Evaluate
# --------------------------------

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


print("\n===== MODEL RESULTS =====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))


# --------------------------------
# Save model
# --------------------------------

joblib.dump(
    model,
    "ats_model.pkl"
)

print("\nModel saved as ats_model.pkl")