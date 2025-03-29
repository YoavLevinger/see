from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import uuid
import os
import logging

app = FastAPI()

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "main-controller.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

class DescriptionInput(BaseModel):
    description: str

@app.post("/process")
def handle_description(input: DescriptionInput):
    description = input.description
    folder_id = str(uuid.uuid4())
    logging.info(f"Received description for processing. Folder ID: {folder_id}")

    # Split tasks
    response = requests.post("http://localhost:8001/split", json={"description": description})
    subtasks = response.json().get("subtasks", [])
    logging.info(f"Received {len(subtasks)} subtasks from task-splitter.")

    # Generate code for each
    for subtask in subtasks:
        requests.post("http://localhost:8002/generate", json={"subtask": subtask, "folder": folder_id})
        logging.info(f"Code generation triggered for subtask: {subtask}")

    # Notify tool X
    payload = {
        "folder": folder_id,
        "description": description,
        "subtasks": subtasks
    }
    requests.post("http://localhost:8003/handle", json=payload)
    logging.info("Notified tool-x with task payload.")

    # Create policy file
    policy_text = "Ensure user input is validated. Avoid hardcoding credentials. Use secure file handling."
    policy_path = os.path.join("generated-code", folder_id, "policy.txt")
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    with open(policy_path, "w") as f:
        f.write(policy_text)
    logging.info(f"Policy file written to: {policy_path}")

    # Create document
    requests.post("http://localhost:8004/create", json=payload)
    logging.info("Document creation triggered.")

    return {
        "status": "done",
        "folder": folder_id,
        "download": f"http://localhost:8004/download/{folder_id}"
    }
