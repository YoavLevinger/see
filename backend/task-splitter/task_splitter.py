import requests
from fastapi import FastAPI
from backend.shared.models import TaskRequest, SubTaskResponse
import logging
import os

app = FastAPI()

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "task-splitter.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the response."""
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    if response.status_code != 200:
        logging.error("LLM error: %s", response.text)
        return ""
    return response.json().get("response", "")

@app.post("/split", response_model=SubTaskResponse)
def split_description(task: TaskRequest):
    """
    Accepts a software description and returns:
    - All subtasks (general)
    - Development-only subtasks (for code generation)
    """
    logging.info("Received description for task splitting.")

    # Prompt 1: General subtasks (e.g. planning, design, dev, testing)

    general_prompt = (
        f"You are a software architect. Break the following software description into 30 subtasks, "
        f"including planning, development, testing, and documentation.\n\n"
        f"Description:\n{task.description}\n\n"
        f"Return the subtasks as a numbered list."
    )


    general_output = ask_llm(general_prompt)
    general_subtasks = [line.split(". ", 1)[1] for line in general_output.strip().splitlines() if ". " in line]

    # Prompt 2: Development-only subtasks (actual software code responsibilities)
    dev_prompt = (
        f"From the following description, list only the subtasks that result in code to be written. "
        f"Ignore planning, and documentation tasks.\n\n"
        f"break each subtask description to additional coding subtasks, total of 40 coding subtasks.\n\n"
        f"Description:\n{task.description}\n\n"
        f"Return only those as a numbered list."
    )
    dev_output = ask_llm(dev_prompt)
    dev_subtasks = [line.split(". ", 1)[1] for line in dev_output.strip().splitlines() if ". " in line]

    logging.info(f"Generated {len(general_subtasks)} general subtasks and {len(dev_subtasks)} dev subtasks.")

    return SubTaskResponse(
        subtasks=general_subtasks,
        dev_subtasks=dev_subtasks
    )
