import streamlit as st
import pymupdf
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_match(resume_text, job_description):

    resume_embedding = semantic_model.encode(resume_text)
    job_embedding = semantic_model.encode(job_description)

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    )

    return float(similarity[0][0]) * 100


# -----------------------------
# 1. Load ML model
# -----------------------------

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


# -----------------------------
# 2. Extract PDF text
# -----------------------------

def extract_text(uploaded_file):

    document = pymupdf.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# -----------------------------
# 3. Extract features
# -----------------------------

def extract_features(text):

    skills = [
        "Python", "JavaScript", "TypeScript", "SQL",
        "HTML5", "CSS3", "React", "Next.js",
        "Redux", "Tailwind CSS", "Node.js",
        "Express.js", "REST APIs", "MongoDB",
        "PostgreSQL", "MySQL", "Redis",
        "Git", "GitHub", "Docker",
        "Postman", "Linux", "AWS"
    ]

    found_skills = [
        skill for skill in skills
        if skill.lower() in text.lower()
    ]

    word_count = len(text.split())

    sections = [
        "PROFESSIONAL SUMMARY",
        "TECHNICAL SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "PROJECTS",
        "EDUCATION",
        "CERTIFICATIONS"
    ]

    sections_count = sum(
        1
        for section in sections
        if section.lower() in text.lower()
    )

    experience_years = 0

    matches = re.findall(
        r"(?i)(\d+)\+?\s*years?\s*of\s*experience",
        text
    )

    if matches:
        experience_years = max(map(int, matches))

    project_section = re.search(
        r"(?is)PROJECTS(.*?)(?=EDUCATION|CERTIFICATIONS|ADDITIONAL INFORMATION|$)",
        text
    )

    projects_count = 0

    if project_section:
        project_text = project_section.group(1)

        projects_count = len(
            re.findall(r"(?m)^.+?\s+—", project_text)
        )

    return {
        "skills": len(found_skills),
        "experience": experience_years,
        "projects": projects_count,
        "sections": sections_count,
        "word_count": word_count,
        "found_skills": found_skills
    }


# -----------------------------
# 4. Keyword matching
# -----------------------------

def calculate_keyword_match(text, job_description):

    skills = [
        "Python", "JavaScript", "TypeScript", "SQL",
        "HTML5", "CSS3", "React", "Next.js",
        "Redux", "Tailwind CSS", "Node.js",
        "Express.js", "REST APIs", "MongoDB",
        "PostgreSQL", "MySQL", "Redis",
        "Git", "GitHub", "Docker",
        "Postman", "Linux", "AWS"
    ]

    required = [
        skill for skill in skills
        if skill.lower() in job_description.lower()
    ]

    if not required:
        return 0, []

    matched = [
        skill for skill in required
        if skill.lower() in text.lower()
    ]

    percentage = len(matched) / len(required) * 100

    return percentage, matched


# -----------------------------
# 5. Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄"
)

st.title("📄 AI Resume Builder")

mode = st.radio(
    "Choose an option:",
    ["ATS Resume Analyzer", "Create New Resume"]
)


# =====================================================
# ATS ANALYZER
# =====================================================

if mode == "ATS Resume Analyzer":

    st.write(
        "Upload your resume and compare it with a job description."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

    job_description = st.text_area(
        "Paste Job Description"
    )

    if st.button("Analyze Resume"):

        if uploaded_file is None:
            st.error("Please upload a resume PDF.")

        elif not job_description.strip():
            st.error("Please enter a job description.")

        else:

            text = extract_text(uploaded_file)

            features = extract_features(text)

            keyword_match, matched_skills = calculate_keyword_match(
                text,
                job_description
            )

            def semantic_match(resume_text, job_description):

    resume_embedding = semantic_model.encode(resume_text)
    job_embedding = semantic_model.encode(job_description)

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    )

    return float(similarity[0][0]) * 100

            model_input = pd.DataFrame([{
                "skills": features["skills"],
                "experience": features["experience"],
                "projects": features["projects"],
                "sections": features["sections"],
                "word_count": features["word_count"],
                "keyword_match": keyword_match
            }])

            ats_score = model.predict(model_input)[0]

            ats_score = max(0, min(100, ats_score))

            st.success(
                f"ATS Score: {ats_score:.1f}/100"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Skills",
                features["skills"]
            )

            col2.metric(
                "Experience",
                f"{features['experience']} years"
            )

            col3.metric(
                "Projects",
                features["projects"]
            )

            st.metric(
                "Keyword Match",
                f"{keyword_match:.1f}%"
            )

            st.metric(
    "Semantic Match",
    f"{semantic_score:.1f}%"
)

            st.subheader("Skills Found")

            st.write(
                ", ".join(features["found_skills"])
            )

            st.subheader("Matched Job Skills")

            st.write(
                ", ".join(matched_skills)
                if matched_skills
                else "No matching skills found."
            )

            st.subheader("Recommendations")

            if keyword_match < 60:
                st.warning(
                    "Add more skills and keywords relevant to the job."
                )
            else:
                st.success(
                    "Good keyword alignment with the job."
                )

            if features["projects"] == 0:
                st.warning(
                    "Consider adding relevant projects."
                )


# =====================================================
# CREATE RESUME
# =====================================================

else:

    st.header("Create a New Resume")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    role = st.text_input("Target Job Role")

    skills_input = st.text_area(
        "Skills (comma separated)"
    )

    experience = st.text_area(
        "Experience"
    )

    projects = st.text_area(
        "Projects"
    )

    education = st.text_area(
        "Education"
    )

    summary = st.text_area(
        "Professional Summary"
    )

    if st.button("Generate Resume"):

        st.success("Resume generated!")

        st.markdown("---")

        st.header(name)

        st.write(f"**{role}**")

        st.write(f"📧 {email}")
        st.write(f"📱 {phone}")

        st.subheader("Professional Summary")
        st.write(summary)

        st.subheader("Skills")
        st.write(skills_input)

        st.subheader("Experience")
        st.write(experience)

        st.subheader("Projects")
        st.write(projects)

        st.subheader("Education")
        st.write(education)