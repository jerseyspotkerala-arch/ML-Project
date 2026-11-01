import pymupdf
import re

# Read PDF
document = pymupdf.open("resume.pdf")
text = ""

for page in document:
    text += page.get_text()

document.close()

skills = [
    "Python", "JavaScript", "TypeScript", "SQL", "HTML5", "CSS3",
    "React", "Next.js", "Redux", "Tailwind CSS", "Node.js",
    "Express.js", "REST APIs", "MongoDB", "PostgreSQL", "MySQL",
    "Redis", "Git", "GitHub", "Docker", "Postman", "Linux", "AWS"
]

# Skills
found_skills = [
    skill for skill in skills
    if skill.lower() in text.lower()
]

# Words
word_count = len(text.split())

# Sections
sections = [
    "PROFESSIONAL SUMMARY",
    "TECHNICAL SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "PROJECTS",
    "EDUCATION",
    "CERTIFICATIONS"
]

sections_count = sum(
    1 for section in sections
    if section.lower() in text.lower()
)

# Experience
experience_years = 0

matches = re.findall(
    r"(?i)(\d+)\+?\s*years?\s*of\s*experience",
    text
)

if matches:
    experience_years = max(map(int, matches))

# Projects
project_section = re.search(
    r"(?is)PROJECTS(.*?)(?=EDUCATION|CERTIFICATIONS|ADDITIONAL INFORMATION|$)",
    text
)

projects_count = 0

if project_section:
    project_text = project_section.group(1)

    # In this resume, project names appear before an em dash
    projects_count = len(
        re.findall(r"(?m)^.+?\s+—", project_text)
    )

print("===== RESUME FEATURES =====")
print("Skills:", len(found_skills))
print("Experience:", experience_years, "years")
print("Projects:", projects_count)
print("Sections:", sections_count)
print("Word count:", word_count)
print("Skills found:", found_skills)