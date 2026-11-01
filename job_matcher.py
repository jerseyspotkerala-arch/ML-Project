resume_text = """
I am a Python developer with experience in Django, SQL and Git.
I have also worked on machine learning projects.
"""

job_description = """
We are looking for a Python Developer.
Required skills include Python, Django, SQL, Git and Docker.
"""

required_skills = [
    "Python",
    "Django",
    "SQL",
    "Git",
    "Docker"
]

matched_skills = []
missing_skills = []

for skill in required_skills:
    if skill.lower() in resume_text.lower():
        matched_skills.append(skill)
    else:
        missing_skills.append(skill)

print("Matched skills:")
for skill in matched_skills:
    print("-", skill)

print("\nMissing skills:")
for skill in missing_skills:
    print("-", skill)

match_percentage = (
    len(matched_skills) / len(required_skills)
) * 100

print("\nKeyword Match:", match_percentage, "%")