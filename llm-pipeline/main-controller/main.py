from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import uuid

app = FastAPI()

class DescriptionInput(BaseModel):
    description: str

@app.post("/process")
def handle_description(input: DescriptionInput):
    description = input.description
    folder_id = str(uuid.uuid4())

    # Split tasks
    response = requests.post("http://localhost:8001/split", json={"description": description})
    subtasks = response.json().get("subtasks", [])

    # Generate code for each
    for subtask in subtasks:
        requests.post("http://localhost:8002/generate", json={"subtask": subtask, "folder": folder_id})

    # Notify tool X
    payload = {
        "folder": folder_id,
        "description": description,
        "subtasks": subtasks
    }
    requests.post("http://localhost:8003/handle", json=payload)

    return {"status": "done", "folder": folder_id}