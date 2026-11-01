import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")

X = data[
    [
        "skills",
        "experience",
        "projects",
        "sections",
        "word_count",
        "keyword_match"
    ]
]

y = data["ats_score"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Model trained successfully!")