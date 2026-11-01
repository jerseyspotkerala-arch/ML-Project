from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_match(resume_text, job_description):

    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_description)

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    )

    score = float(similarity[0][0]) * 100

    return round(score, 2)