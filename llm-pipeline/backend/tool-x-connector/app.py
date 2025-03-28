from fastapi import FastAPI
from backend.shared.models import ToolXRequest
import os

app = FastAPI()

@app.post("/handle")
def handle_input(req: ToolXRequest):
    print("Tool X received:")
    print("Description:", req.description)
    print("Folder:", req.folder)
    print("Subtasks:", req.subtasks)
    folder_path = os.path.join("backend/generated-code", req.folder)
    if os.path.exists(folder_path):
        print("Files:", os.listdir(folder_path))
    else:
        print("Folder does not exist.")

    return {"status": "received"}
