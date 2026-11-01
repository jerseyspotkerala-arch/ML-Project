import spacy
from sentence_transformers import SentenceTransformer, util

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

resume = """
Software Developer with experience building web applications.
Developed REST APIs using Python and Node.js.
Built machine learning projects and worked with SQL databases.
"""

job = """
We are looking for a Python developer with experience in
REST APIs, machine learning, SQL and backend development.
"""

# -----------------------------
# spaCy NLP
# -----------------------------

doc = nlp(resume)

print("Entities found:")

for entity in doc.ents:
    print(entity.text, "->", entity.label_)


# -----------------------------
# Semantic similarity
# -----------------------------

resume_embedding = model.encode(resume)
job_embedding = model.encode(job)

similarity = util.cos_sim(
    resume_embedding,
    job_embedding
)

score = float(similarity[0][0]) * 100

print("\nSemantic Similarity:", round(score, 2), "%")