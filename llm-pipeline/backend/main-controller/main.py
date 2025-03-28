from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import uuid
import os

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

    # Create policy file (you can modify this policy)
    policy_text = "Ensure user input is validated. Avoid hardcoding credentials. Use secure file handling."
    policy_path = os.path.join("backend/generated-code", folder_id, "policy.txt")
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    with open(policy_path, "w") as f:
        f.write(policy_text)

    # Create document (which will include advisor feedback)
    requests.post("http://localhost:8004/create", json=payload)

    return {
        "status": "done",
        "folder": folder_id,
        "download": f"http://localhost:8004/download/{folder_id}"
    }
