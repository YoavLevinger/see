import os
import requests
from fastapi import FastAPI
from backend.shared.models import AdvisorRequest, AdvisorResponse

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def analyze_code_against_policy(policy: str, code: str) -> str:
    prompt = f"""
You are a software security advisor.
Given the following policy or security guideline:
{policy}

Review the following code and provide recommendations to improve compliance:

{code}

Respond with a numbered list of specific suggestions.
"""
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    return response.json()["response"]

@app.post("/advise", response_model=AdvisorResponse)
def give_advice(req: AdvisorRequest):
    folder_path = os.path.join("backend/generated-code", req.folder)
    code_text = ""

    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                with open(os.path.join(folder_path, file), "r") as f:
                    code_text += f"\n# {file}\n" + f.read()

    advice_text = analyze_code_against_policy(req.policy, code_text)
    recommendations = [line.split(". ", 1)[1] for line in advice_text.strip().splitlines() if ". " in line]

    return AdvisorResponse(recommendations=recommendations)