resume_text = """
I am a Python developer with experience in Django, SQL and Git.
I have also worked on machine learning projects.
"""

skills = [
    "Python",
    "Django",
    "SQL",
    "Git",
    "Machine Learning",
    "Java",
    "React",
    "Docker"
]

found_skills = []

for skill in skills:
    if skill.lower() in resume_text.lower():
        found_skills.append(skill)

print("Skills found:")
for skill in found_skills:
    print("-", skill)

print("Total skills:", len(found_skills))