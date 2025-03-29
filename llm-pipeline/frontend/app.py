from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import os
import uuid
import logging

app = FastAPI()

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "frontend.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    folder_id = str(uuid.uuid4())
    logging.info(f"New session started with folder ID: {folder_id}")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "folder_id": folder_id
    })

@app.post("/submit", response_class=HTMLResponse)
async def handle_submission(
    request: Request,
    description: str = Form(...),
    folder_id: str = Form(...),
    security_policy: UploadFile = File(None),
    accessibility_policy: UploadFile = File(None),
    performance_policy: UploadFile = File(None),
    other_files: list[UploadFile] = File(default=[])
):
    folder_path = os.path.join("generated-code", folder_id)
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Handling submission for folder ID: {folder_id}")

    policy_map = {
        "security": security_policy,
        "accessibility": accessibility_policy,
        "performance": performance_policy,
    }

    for name, file in policy_map.items():
        if file and file.filename.endswith(".pdf"):
            policy_path = os.path.join(folder_path, f"policy.{name}.pdf")
            with open(policy_path, "wb") as f:
                f.write(await file.read())
            logging.info(f"Uploaded {name} advisor policy to {policy_path}")

    for file in other_files:
        if not file.filename.strip():
            continue
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"Uploaded general file: {file.filename}")

    logging.info("Sending description to pipeline API...")
    response = requests.post(
        "http://localhost:8080/process",
        json={"description": description}
    )

    if response.status_code == 200:
        result = response.json()
        logging.info(f"Pipeline executed successfully. Download link: {result.get('download')}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Pipeline executed successfully!",
            "download_url": result.get("download"),
            "folder_id": folder_id
        })
    else:
        logging.error("Pipeline execution failed.")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Error submitting the task.",
            "folder_id": folder_id
        })
