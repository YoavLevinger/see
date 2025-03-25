from fastapi import FastAPI
from shared.models import ToolXRequest
import os

app = FastAPI()

@app.post("/handle")
def handle_input(req: ToolXRequest):
    print("Tool X received:")
    print("Description:", req.description)
    print("Folder:", req.folder)
    print("Subtasks:", req.subtasks)
    print("Files:", os.listdir(os.path.join("generated-code", req.folder)))

    return {"status": "received"}