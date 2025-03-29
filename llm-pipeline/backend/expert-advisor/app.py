import os
import requests
import logging
from fastapi import FastAPI
from backend.shared.models import AdvisorRequest, AdvisorResponse
from PyPDF2 import PdfReader

app = FastAPI()

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "expert-advisor.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

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
    logging.info("Sending code and policy to LLM for analysis...")
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    logging.info("Received response from LLM.")
    return response.json()["response"]

@app.post("/advise", response_model=AdvisorResponse)
def give_advice(req: AdvisorRequest):
    folder_path = os.path.join("generated-code", req.folder)
    code_text = ""

    if os.path.exists(folder_path):
        logging.info(f"Reading .py files from folder: {folder_path}")
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                with open(os.path.join(folder_path, file), "r") as f:
                    code_text += f"\n# {file}\n" + f.read()
                logging.info(f"Included file: {file}")
    else:
        logging.warning(f"Folder does not exist: {folder_path}")

    if not code_text.strip():
        logging.warning("No Python code found to analyze.")
        return AdvisorResponse(recommendations=[])

    policy_text = ""
    for filename in os.listdir(folder_path):
        if filename.startswith("policy.") and filename.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(folder_path, filename))
                for page in reader.pages:
                    policy_text += page.extract_text() or ""
                logging.info(f"Extracted policy from {filename}")
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")

    if not policy_text.strip():
        logging.warning("No valid policy PDF found or it was empty.")
        return AdvisorResponse(recommendations=[])

    advice_text = analyze_code_against_policy(policy_text, code_text)
    recommendations = [line.split(". ", 1)[1] for line in advice_text.strip().splitlines() if ". " in line]
    logging.info(f"Generated {len(recommendations)} recommendations.")
    return AdvisorResponse(recommendations=recommendations)