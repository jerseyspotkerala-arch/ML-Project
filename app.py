import streamlit as st
import pymupdf
import pandas as pd
import numpy as np
import re
import os
import io
import json
import base64
import joblib

from sentence_transformers import SentenceTransformer, util
from google import genai

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "Analyze Resume",
    "analysis_done": False,
    "analysis": None,
    "ai_resume": None,
    "create_ai_resume": None,
    "pdf_bytes": None,
    "create_pdf_bytes": None,
    "gemini_api_key": "",
    "gemini_model": "Auto-Switch (Lowest Demand)",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def _load_ml_model_cached(model_path, mtime):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def load_ml_model():
    model_path = "ats_model_better.pkl"
    mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
    return _load_ml_model_cached(model_path, mtime)


semantic_model = load_semantic_model()
ml_model = load_ml_model()


# =========================================================
# GEMINI
# =========================================================

def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()


def get_gemini_api_key():
    # 1. Check existing environment variable
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Reload .env file if key wasn't in os.environ
    load_env_file()
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 3. Streamlit secrets
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            sec_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if sec_key:
                return sec_key
    except Exception:
        pass

    return None


def get_gemini_client():
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


LOW_DEMAND_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]

def call_gemini(client, prompt):
    chosen = st.session_state.get("gemini_model", "Auto-Switch (Lowest Demand)")
    models_to_try = list(LOW_DEMAND_MODELS)

    # Honor user preference if specified
    if "flash-lite-latest" in chosen:
        preferred = "gemini-flash-lite-latest"
    elif "3.5-flash-lite" in chosen:
        preferred = "gemini-3.5-flash-lite"
    elif "3.6-flash" in chosen:
        preferred = "gemini-3.6-flash"
    elif "3.8-flash" in chosen:
        preferred = "gemini-3.8-flash"
    else:
        preferred = None

    if preferred and preferred in models_to_try:
        models_to_try.remove(preferred)
        models_to_try.insert(0, preferred)

    last_err = None
    transient_triggers = [
        "503", "429", "unavailable", "demand", "overload",
        "resource_exhausted", "quota", "busy", "rate limit",
        "not found", "404", "not_found"
    ]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            # If high demand or transient error, auto-switch to next low-demand model!
            if any(trigger in err_msg for trigger in transient_triggers):
                continue
            raise e

    if last_err:
        raise Exception(
            f"All available Gemini models are currently experiencing high demand. Please try again shortly. Details: {last_err}"
        )
    return ""



# =========================================================
# SKILLS
# =========================================================

SKILLS = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Dart", "R", "Scala", "Bash", "Shell", "SQL",
    # Web & Frontend
    "HTML", "HTML5", "CSS", "CSS3", "React", "Next.js", "Angular", "Vue", "Vue.js",
    "Redux", "Tailwind CSS", "Bootstrap", "Sass", "jQuery", "Webpack",
    # Backend & Frameworks
    "Node.js", "Express.js", "REST APIs", "RESTful APIs", "GraphQL", "FastAPI",
    "Django", "Flask", "Spring Boot", "ASP.NET", ".NET", "Laravel", "Microservices",
    # Databases & Storage
    "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis", "DynamoDB", "Oracle",
    "MS SQL Server", "Elasticsearch", "Firebase",
    # Cloud, DevOps & Tools
    "Git", "GitHub", "GitLab", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Linux", "Unix", "CI/CD", "Jenkins", "Terraform", "Postman", "Jira",
    # Data Science & AI
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Pandas",
    "NumPy", "Scikit-learn", "NLP", "Natural Language Processing", "OpenCV",
    "Computer Vision", "Generative AI", "LLM", "Data Analysis",
    # Engineering & Methodologies
    "Agile", "Scrum", "Unit Testing", "Integration Testing", "TDD", "SDLC",
    "Debugging", "System Design", "OOP", "Object-Oriented Programming"
]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    document = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text.strip()


# =========================================================
# BASIC RESUME FEATURES
# =========================================================

def extract_resume_features(text):

    lower = text.lower()

    found_skills = []

    for skill in SKILLS:
        if skill.lower() in lower:
            found_skills.append(skill)

    word_count = len(text.split())

    has_skills = int(
        bool(
            re.search(
                r"technical skills|skills|core skills|key skills",
                lower
            )
        )
    )

    has_experience = int(
        bool(
            re.search(
                r"experience|employment|work history",
                lower
            )
        )
    )

    has_education = int(
        "education" in lower
    )

    has_projects = int(
        "projects" in lower
    )

    has_certifications = int(
        bool(
            re.search(
                r"certifications|certificates|licenses",
                lower
            )
        )
    )

    has_summary = int(
        bool(
            re.search(
                r"professional summary|summary|profile|objective",
                lower
            )
        )
    )

    experience_years = 0

    matches = re.findall(
        r"(\d+)\+?\s*years?\s*(?:of)?\s*experience",
        lower
    )

    if matches:
        experience_years = max(
            int(x) for x in matches
        )

    return {
        "found_skills": found_skills,
        "resume_words": word_count,
        "has_skills": has_skills,
        "has_experience": has_experience,
        "has_education": has_education,
        "has_projects": has_projects,
        "has_certifications": has_certifications,
        "has_summary": has_summary,
        "experience_years": experience_years,
    }


# =========================================================
# JOB KEYWORD MATCHING
# =========================================================

def calculate_keyword_match(resume_text, job_description):

    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    required = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, job_lower):
            required.append(skill)

    # If no predefined skills found, extract key technical terms from job description
    if not required:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b", job_description)
        stop_words = {
            "the", "and", "for", "with", "you", "are", "this", "that", "from", "your",
            "our", "will", "have", "has", "using", "work", "years", "job", "role", "must",
            "can", "all", "any", "but", "not", "into", "out", "about", "other", "looking",
            "experience", "candidate", "responsibilities", "requirements", "skills", "team",
            "company", "position", "applicant", "preferred", "qualifications", "apply"
        }
        filtered = [w for w in words if w.lower() not in stop_words]
        from collections import Counter
        common = [w for w, _ in Counter(filtered).most_common(12)]
        required = common

    if not required:
        return 75.0, [], []

    matched = []
    missing = []

    for skill in required:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, resume_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    score = (len(matched) / len(required)) * 100.0
    return score, matched, missing


# =========================================================
# SEMANTIC MATCH
# =========================================================

def semantic_match(resume_text, job_description):

    resume_embedding = semantic_model.encode(
        resume_text,
        convert_to_tensor=True
    )

    job_embedding = semantic_model.encode(
        job_description,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    )

    raw_score = float(similarity[0][0])
    # Calibrate MiniLM cosine similarity:
    # 0.15 is unrelated, 0.60+ is an outstanding contextual match
    calibrated = (raw_score - 0.15) / (0.60 - 0.15) * 100.0
    return max(0.0, min(100.0, calibrated))


# =========================================================
# ML ATS SCORE
# =========================================================

def predict_ats_score(
    resume_text,
    job_description,
    keyword_score
):
    current_model = load_ml_model()

    features = extract_resume_features(resume_text)

    semantic_score = semantic_match(
        resume_text,
        job_description
    )

    # 1. Section presence score (out of 100)
    section_score = (
        features["has_skills"] * 25 +
        features["has_experience"] * 25 +
        features["has_education"] * 20 +
        features["has_projects"] * 10 +
        features["has_certifications"] * 10 +
        features["has_summary"] * 10
    )

    # 2. Length & Formatting score (out of 100)
    words = features["resume_words"]
    if 350 <= words <= 850:
        length_score = 100.0
    elif 250 <= words < 350 or 850 < words <= 1100:
        length_score = 85.0
    elif 150 <= words < 250 or 1100 < words <= 1400:
        length_score = 70.0
    else:
        length_score = 50.0

    # 3. Industry-Standard Weighted ATS Score
    # 45% Keyword Match + 30% Semantic Match + 15% Section Completeness + 10% Length
    formula_score = (
        0.45 * keyword_score +
        0.30 * semantic_score +
        0.15 * section_score +
        0.10 * length_score
    )

    # 4. Blend with ML model prediction
    if current_model is not None:
        try:
            job_words = len(job_description.split())
            ratio = words / max(job_words, 1)

            model_input = pd.DataFrame([{
                "semantic_match": semantic_score,
                "keyword_match": keyword_score,
                "resume_words": words,
                "job_words": job_words,
                "resume_length_ratio": ratio,
                "has_skills": features["has_skills"],
                "has_experience": features["has_experience"],
                "has_education": features["has_education"],
                "has_projects": features["has_projects"],
                "has_certifications": features["has_certifications"],
                "has_summary": features["has_summary"],
            }])

            if hasattr(current_model, "feature_names_in_"):
                model_input = model_input[list(current_model.feature_names_in_)]

            ml_pred = current_model.predict(model_input)[0]
            ats_score = 0.60 * formula_score + 0.40 * ml_pred
        except Exception:
            ats_score = formula_score
    else:
        ats_score = formula_score

    return max(0.0, min(100.0, float(ats_score)))


# =========================================================
# GEMINI AI RESUME IMPROVEMENT
# =========================================================

def generate_ai_resume(
    resume_text,
    job_description,
    missing_skills
):

    client = get_gemini_client()

    if client is None:
        raise Exception(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
        )

    missing = ", ".join(missing_skills)

    prompt = f"""
You are an expert ATS resume writer.

Rewrite the resume below for the supplied job description.

IMPORTANT RULES:

1. Never invent employment, education, certifications,
   technologies, achievements, dates, companies or metrics.

2. Preserve all factual information from the original resume.

3. Improve wording for ATS and recruiters.

4. Use standard ATS section names.

5. Include certifications if they exist in the original resume.

6. Do not remove useful technical skills.

7. Naturally incorporate job-description keywords only when
   they are supported by the original resume.

8. Use strong action verbs.

9. Keep the resume professional and concise.

10. Do not use tables, columns, graphics, icons or emojis.

11. Produce a single-column ATS-friendly resume.

12. Use exactly these sections when information exists:

PROFESSIONAL SUMMARY
TECHNICAL SKILLS
PROFESSIONAL EXPERIENCE
PROJECTS
EDUCATION
CERTIFICATIONS

13. Do not create fake information to fill missing sections.

JOB DESCRIPTION:
{job_description}

MISSING JOB SKILLS:
{missing}

ORIGINAL RESUME:
{resume_text}

Return only the improved resume text.
Do not explain what you changed.
"""

    return call_gemini(client, prompt)


# =========================================================
# GEMINI CREATE NEW RESUME
# =========================================================

def generate_new_resume(
    name,
    email,
    phone,
    location,
    role,
    skills,
    experience,
    projects,
    education,
    certifications,
    summary,
    job_description
):

    client = get_gemini_client()

    if client is None:
        raise Exception(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
        )

    prompt = f"""
Create a professional ATS-friendly resume.

IMPORTANT:

- Never invent facts.
- Only use information supplied below.
- Improve wording and structure.
- Use standard ATS section headings.
- Keep it single-column.
- No tables.
- No graphics.
- No emojis.
- No fake metrics.
- Include CERTIFICATIONS when supplied.
- Optimize naturally for the job description.

Use this order:

NAME
TARGET ROLE
CONTACT
PROFESSIONAL SUMMARY
TECHNICAL SKILLS
PROFESSIONAL EXPERIENCE
PROJECTS
EDUCATION
CERTIFICATIONS

NAME:
{name}

EMAIL:
{email}

PHONE:
{phone}

LOCATION:
{location}

TARGET ROLE:
{role}

SKILLS:
{skills}

EXPERIENCE:
{experience}

PROJECTS:
{projects}

EDUCATION:
{education}

CERTIFICATIONS:
{certifications}

SUMMARY:
{summary}

JOB DESCRIPTION:
{job_description}

Return only the final resume text.
"""

    return call_gemini(client, prompt)


# =========================================================
# RESUME PARSER
# =========================================================

def clean_ai_text(text):

    text = text.replace("```text", "")
    text = text.replace("```markdown", "")
    text = text.replace("```", "")

    return text.strip()


def parse_resume_sections(text):

    text = clean_ai_text(text)

    headings = [
        "PROFESSIONAL SUMMARY",
        "SUMMARY",
        "TECHNICAL SKILLS",
        "SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "WORK EXPERIENCE",
        "EXPERIENCE",
        "PROJECTS",
        "EDUCATION",
        "CERTIFICATIONS",
        "CERTIFICATIONS & PROFESSIONAL DEVELOPMENT",
    ]

    pattern = (
        r"(?im)^(?:"
        + "|".join(
            re.escape(h) for h in headings
        )
        + r")\s*:?\s*$"
    )

    matches = list(
        re.finditer(pattern, text)
    )

    sections = []

    if not matches:
        return [
            ("RESUME", text)
        ]

    first_start = matches[0].start()

    header = text[:first_start].strip()

    if header:
        sections.append(
            ("HEADER", header)
        )

    for i, match in enumerate(matches):

        heading = match.group().strip().rstrip(":")

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        if heading.upper() == "SUMMARY":
            heading = "PROFESSIONAL SUMMARY"

        elif heading.upper() == "SKILLS":
            heading = "TECHNICAL SKILLS"

        elif heading.upper() in [
            "WORK EXPERIENCE",
            "EXPERIENCE"
        ]:
            heading = "PROFESSIONAL EXPERIENCE"

        sections.append(
            (heading.upper(), content)
        )

    return sections


# =========================================================
# THEME
# =========================================================

THEMES = {

    "Professional Blue": {
        "primary": colors.HexColor("#174EA6"),
        "secondary": colors.HexColor("#EAF2FF"),
        "text": colors.HexColor("#202124"),
    },

    "Modern Green": {
        "primary": colors.HexColor("#137333"),
        "secondary": colors.HexColor("#E8F5E9"),
        "text": colors.HexColor("#202124"),
    },

    "Executive Black": {
        "primary": colors.HexColor("#222222"),
        "secondary": colors.HexColor("#EEEEEE"),
        "text": colors.HexColor("#202124"),
    },

}


# =========================================================
# PDF GENERATOR
# =========================================================

def create_resume_pdf(
    resume_text,
    theme_name="Professional Blue"
):

    theme = THEMES[theme_name]

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="ATS Resume"
    )

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "Name",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=theme["primary"],
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    contact_style = ParagraphStyle(
        "Contact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=theme["text"],
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=theme["primary"],
        spaceBefore=7,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12,
        textColor=theme["text"],
        spaceAfter=3,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        bulletIndent=0,
        spaceAfter=2,
    )

    story = []

    sections = parse_resume_sections(
        resume_text
    )

    first_section = True

    for heading, content in sections:

        if heading == "HEADER":

            lines = [
                x.strip()
                for x in content.splitlines()
                if x.strip()
            ]

            if lines:

                story.append(
                    Paragraph(
                        lines[0],
                        name_style
                    )
                )

                if len(lines) > 1:

                    contact = " | ".join(
                        lines[1:]
                    )

                    story.append(
                        Paragraph(
                            contact,
                            contact_style
                        )
                    )

            continue

        if heading == "RESUME":

            story.append(
                Paragraph(
                    content.replace(
                        "\n",
                        "<br/>"
                    ),
                    body_style
                )
            )

            continue

        # Section heading
        story.append(
            Paragraph(
                heading,
                section_style
            )
        )

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.7,
                color=theme["primary"],
                spaceBefore=0,
                spaceAfter=4,
            )
        )

        lines = content.splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                story.append(
                    Spacer(1, 2)
                )
                continue

            # Bullet
            if line.startswith(
                ("•", "-", "*", "▪", "●")
            ):

                clean = re.sub(
                    r"^[•\-\*▪●]\s*",
                    "",
                    line
                )

                story.append(
                    Paragraph(
                        "• " + clean,
                        bullet_style
                    )
                )

            else:

                safe_line = line.replace(
                    "&",
                    "&amp;"
                )

                story.append(
                    Paragraph(
                        safe_line,
                        body_style
                    )
                )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# PDF PREVIEW
# =========================================================

def show_pdf_preview(pdf_bytes):

    if not pdf_bytes:
        return

    encoded = base64.b64encode(
        pdf_bytes
    ).decode("utf-8")

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{encoded}"
        width="100%"
        height="900"
        type="application/pdf">
    </iframe>
    """

    st.components.v1.html(
        pdf_display,
        height=920,
        scrolling=True
    )


# =========================================================
# UI HEADER
# =========================================================

st.title("📄 AI Resume ATS Analyzer")

st.caption(
    "ATS analysis • Semantic matching • Generative AI • PDF Resume Builder"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Navigation")

    page = st.radio(
        "Choose:",
        [
            "Analyze Resume",
            "Create New Resume"
        ],
        index=(
            0
            if st.session_state.page == "Analyze Resume"
            else 1
        )
    )

    st.session_state.page = page

    st.divider()

    st.subheader("⚡ AI Engine")
    model_choice = st.selectbox(
        "Model Routing",
        [
            "Auto-Switch (Lowest Demand)",
            "gemini-flash-lite-latest (Fastest)",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.8-flash",
        ],
        index=0,
        help="Automatically switches to models with less demand to prevent 503/high-demand errors."
    )
    st.session_state.gemini_model = model_choice

    st.divider()

    st.info(
        "ATS-friendly resumes use clear headings, "
        "normal text, standard fonts and simple layouts."
    )


# =========================================================
# ANALYZE RESUME
# =========================================================

if st.session_state.page == "Analyze Resume":

    st.header("ATS Resume Analyzer")

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        key="analyze_pdf"
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        key="job_description"
    )

    analyze_button = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if uploaded_file is None:

            st.error(
                "Please upload a resume PDF."
            )

        elif not job_description.strip():

            st.error(
                "Please enter the job description."
            )

        else:

            with st.spinner(
                "Analyzing resume..."
            ):

                resume_text = extract_pdf_text(
                    uploaded_file
                )

                features = extract_resume_features(
                    resume_text
                )

                keyword_score, matched, missing = (
                    calculate_keyword_match(
                        resume_text,
                        job_description
                    )
                )

                semantic_score = semantic_match(
                    resume_text,
                    job_description
                )

                ats_score = predict_ats_score(
                    resume_text,
                    job_description,
                    keyword_score
                )

                st.session_state.analysis = {
                    "resume_text": resume_text,
                    "job_description": job_description,
                    "features": features,
                    "keyword_score": keyword_score,
                    "matched": matched,
                    "missing": missing,
                    "semantic_score": semantic_score,
                    "ats_score": ats_score,
                }

                st.session_state.analysis_done = True
                st.session_state.ai_resume = None
                st.session_state.pdf_bytes = None

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    if st.session_state.analysis_done:

        result = st.session_state.analysis

        st.divider()

        st.subheader("ATS Analysis")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "ATS Score",
            f"{result['ats_score']:.1f}/100"
        )

        c2.metric(
            "Keyword Match",
            f"{result['keyword_score']:.1f}%"
        )

        c3.metric(
            "Semantic Match",
            f"{result['semantic_score']:.1f}%"
        )

        st.divider()

        features = result["features"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Skills",
            len(features["found_skills"])
        )

        c2.metric(
            "Words",
            features["resume_words"]
        )

        c3.metric(
            "Experience",
            f"{features['experience_years']} years"
        )

        c4.metric(
            "Projects",
            features["has_projects"]
        )

        st.subheader("Skills Found")

        st.write(
            ", ".join(
                features["found_skills"]
            )
            if features["found_skills"]
            else "No recognized skills."
        )

        st.subheader("Matched Job Skills")

        st.write(
            ", ".join(result["matched"])
            if result["matched"]
            else "No matching skills found."
        )

        st.subheader("Missing Job Skills")

        if result["missing"]:

            st.warning(
                ", ".join(result["missing"])
            )

        else:

            st.success(
                "No missing skills detected from the supported skill list."
            )

        st.divider()

        # -------------------------------------------------
        # AI RESUME GENERATOR
        # -------------------------------------------------

        st.subheader(
            "🤖 Generative AI Resume"
        )

        st.write(
            "Generate a completely rewritten, ATS-friendly "
            "version of this resume using Gemini."
        )

        ai_button = st.button(
            "✨ Generate ATS Resume with AI",
            type="primary",
            use_container_width=True,
            key="generate_ai_analyzer"
        )

        if ai_button:

            with st.spinner(
                "Gemini is improving your resume..."
            ):

                try:

                    improved = generate_ai_resume(
                        result["resume_text"],
                        result["job_description"],
                        result["missing"]
                    )

                    st.session_state.ai_resume = improved
                    st.session_state.pdf_bytes = None

                except Exception as e:

                    st.error(
                        f"AI improvement error: {e}"
                    )

        # -------------------------------------------------
        # AI RESULT
        # -------------------------------------------------

        if st.session_state.ai_resume:

            st.success(
                "AI resume generated successfully."
            )

            st.subheader(
                "AI-Generated ATS Resume"
            )

            st.text_area(
                "Preview / Edit Resume",
                value=st.session_state.ai_resume,
                height=650,
                key="ai_resume_editor"
            )

            st.session_state.ai_resume = (
                st.session_state.ai_resume_editor
            )

            st.divider()

            theme = st.selectbox(
                "Choose PDF Color Theme",
                list(THEMES.keys()),
                key="analyzer_theme"
            )

            if st.button(
                "📄 Generate Professional PDF",
                type="primary",
                use_container_width=True,
                key="generate_analyzer_pdf"
            ):

                with st.spinner(
                    "Creating ATS-friendly PDF..."
                ):

                    st.session_state.pdf_bytes = (
                        create_resume_pdf(
                            st.session_state.ai_resume,
                            theme
                        )
                    )

            if st.session_state.pdf_bytes:

                st.subheader(
                    "PDF Preview"
                )

                show_pdf_preview(
                    st.session_state.pdf_bytes
                )

                st.download_button(
                    "⬇️ Download ATS Resume PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="AI_ATS_Resume.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="download_analyzer_pdf"
                )


# =========================================================
# CREATE NEW RESUME
# =========================================================

else:

    st.header("Create New Resume")

    st.write(
        "Enter your information and let AI create "
        "a professional ATS-friendly resume."
    )

    with st.form("resume_form"):

        name = st.text_input(
            "Full Name"
        )

        role = st.text_input(
            "Target Job Role"
        )

        email = st.text_input(
            "Email"
        )

        phone = st.text_input(
            "Phone"
        )

        location = st.text_input(
            "Location"
        )

        skills = st.text_area(
            "Skills",
            height=100
        )

        experience = st.text_area(
            "Professional Experience",
            height=180
        )

        projects = st.text_area(
            "Projects",
            height=150
        )

        education = st.text_area(
            "Education",
            height=100
        )

        certifications = st.text_area(
            "Certifications",
            height=100,
            placeholder="Example:\nAWS Certified Cloud Practitioner\nGoogle Data Analytics Certificate"
        )

        summary = st.text_area(
            "Professional Summary",
            height=120
        )

        job_description_create = st.text_area(
            "Target Job Description (optional)",
            height=180
        )

        submitted = st.form_submit_button(
            "✨ Generate Resume with AI",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not name.strip():

            st.error(
                "Please enter your name."
            )

        elif not role.strip():

            st.error(
                "Please enter your target job role."
            )

        else:

            with st.spinner(
                "Creating your ATS-friendly resume..."
            ):

                try:

                    new_resume = generate_new_resume(
                        name=name,
                        email=email,
                        phone=phone,
                        location=location,
                        role=role,
                        skills=skills,
                        experience=experience,
                        projects=projects,
                        education=education,
                        certifications=certifications,
                        summary=summary,
                        job_description=job_description_create
                    )

                    st.session_state.create_ai_resume = (
                        new_resume
                    )

                    st.session_state.create_pdf_bytes = None

                except Exception as e:

                    st.error(
                        f"AI generation error: {e}"
                    )

    # -----------------------------------------------------
    # CREATE RESULT
    # -----------------------------------------------------

    if st.session_state.create_ai_resume:

        st.success(
            "Your AI resume has been generated."
        )

        st.subheader(
            "Generated Resume"
        )

        st.text_area(
            "Review and edit before generating PDF",
            value=st.session_state.create_ai_resume,
            height=650,
            key="create_resume_editor"
        )

        st.session_state.create_ai_resume = (
            st.session_state.create_resume_editor
        )

        st.divider()

        create_theme = st.selectbox(
            "Choose PDF Color Theme",
            list(THEMES.keys()),
            key="create_theme"
        )

        if st.button(
            "📄 Create Professional PDF",
            type="primary",
            use_container_width=True,
            key="create_pdf_button"
        ):

            with st.spinner(
                "Creating PDF..."
            ):

                st.session_state.create_pdf_bytes = (
                    create_resume_pdf(
                        st.session_state.create_ai_resume,
                        create_theme
                    )
                )

        if st.session_state.create_pdf_bytes:

            st.subheader(
                "PDF Preview"
            )

            show_pdf_preview(
                st.session_state.create_pdf_bytes
            )

            st.download_button(
                "⬇️ Download ATS Resume PDF",
                data=st.session_state.create_pdf_bytes,
                file_name="Created_ATS_Resume.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="download_create_pdf"
            )