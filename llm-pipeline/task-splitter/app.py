import requests
from fastapi import FastAPI
from shared.models import TaskRequest, SubTaskResponse

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def ask_llm(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    return response.json()["response"]

@app.post("/split", response_model=SubTaskResponse)
def split_description(task: TaskRequest):
    prompt = (
        f"You are a software architect. Break the following software description into 3-6 development subtasks:\n"
        f"{task.description}\n"
        f"Return the subtasks as a numbered list."
    )
    llm_output = ask_llm(prompt)

    subtasks = [line.split(". ", 1)[1] for line in llm_output.strip().splitlines() if ". " in line]

    return SubTaskResponse(subtasks=subtasks)
