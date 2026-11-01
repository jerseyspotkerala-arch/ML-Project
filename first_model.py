import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Our training data
data = {
    "skills": [10, 3, 8, 2, 12],
    "experience": [1, 0, 1, 0, 1],
    "education": [1, 1, 1, 0, 1],
    "projects": [3, 1, 2, 0, 4],
    "keywords": [90, 40, 80, 20, 95],
    "ats_score": [90, 48, 82, 30, 94]
}

df = pd.DataFrame(data)

# X = information the model sees
X = df[[
    "skills",
    "experience",
    "education",
    "projects",
    "keywords"
]]

# y = correct answer
y = df["ats_score"]

# Create the ML model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X, y)

# Give the model a new resume
new_resume = [[7, 1, 1, 2, 75]]

# Ask the model for a prediction
prediction = model.predict(new_resume)

print("Predicted ATS Score:", prediction[0])