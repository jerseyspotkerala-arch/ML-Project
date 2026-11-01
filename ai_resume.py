import os
from google import genai

# Load .env file if present
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

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY or create a .env file.")

client = genai.Client(api_key=api_key)

resume_text = """
Software Developer with experience in React, Node.js, Python, SQL,
MongoDB, Docker and REST APIs.
"""

prompt = f"""
You are an expert ATS resume improvement assistant.

Analyze this resume:

{resume_text}

Give exactly 3 useful improvements.
Focus on:
1. ATS keyword optimization
2. Stronger achievement-oriented bullet points
3. Missing or weak technical information

Do not invent experience or skills.
"""

LOW_DEMAND_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]

last_err = None
transient_triggers = [
    "503", "429", "unavailable", "demand", "overload",
    "resource_exhausted", "quota", "busy", "rate limit",
    "not found", "404", "not_found"
]

for model_name in LOW_DEMAND_MODELS:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        print(f"[Model used: {model_name}]\n")
        print(response.text)
        break
    except Exception as e:
        last_err = e
        err_msg = str(e).lower()
        if any(trigger in err_msg for trigger in transient_triggers):
            continue
        raise e
else:
    if last_err:
        raise last_err