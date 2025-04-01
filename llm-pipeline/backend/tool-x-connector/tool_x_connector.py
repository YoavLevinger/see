from fastapi import FastAPI
from backend.shared.models import ToolXRequest
import os
import logging

app = FastAPI()

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "tool-x-connector.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

@app.post("/handle")
def handle_input(req: ToolXRequest):
    logging.info("Tool X received:")
    logging.info(f"Description: {req.description}")
    logging.info(f"Folder: {req.folder}")
    logging.info(f"Subtasks: {req.subtasks}")

    folder_path = os.path.join("generated-code", req.folder)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        logging.info(f"Files: {files}")
    else:
        logging.warning("Folder does not exist.")

    return {"status": "received"}