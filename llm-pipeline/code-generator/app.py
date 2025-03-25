import requests
from fastapi import FastAPI
from shared.models import CodeGenRequest
import os

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def generate_code_from_llm(subtask: str) -> str:
    prompt = (
        f"You are a senior Python developer. Write Python code to implement this subtask:\n"
        f"{subtask}\n"
        f"Respond with only valid Python code, no explanations."
    )
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    return response.json()["response"]

@app.post("/generate")
def generate_code(req: CodeGenRequest):
    folder_path = os.path.join("generated-code", req.folder)
    os.makedirs(folder_path, exist_ok=True)

    code = generate_code_from_llm(req.subtask)

    filename = req.subtask.lower().replace(" ", "_") + ".py"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "w") as f:
        f.write(code)

    return {"status": "ok", "file": filepath}
