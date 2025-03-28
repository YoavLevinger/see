from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import os
import uuid

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    folder_id = str(uuid.uuid4())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "folder_id": folder_id
    })

@app.post("/submit", response_class=HTMLResponse)
async def handle_submission(
    request: Request,
    description: str = Form(...),
    advisors: list[str] = Form(...),
    folder_id: str = Form(...),
    policy_files: list[UploadFile] = File(default=[])
):
    folder_path = os.path.join("generated-code", folder_id)
    os.makedirs(folder_path, exist_ok=True)

    # Upload each policy file
    for file in policy_files:
        if not file.filename.strip():
            continue  # skip empty uploads

        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

    # Save selected advisor roles as part of description if needed
    advisor_policy_map = {
        "security": "policy.security.txt",
        "accessibility": "policy.accessibility.txt",
        "performance": "policy.performance.txt",
    }

    # Send software description to pipeline
    response = requests.post(
        "http://localhost:8080/process",
        json={"description": description}
    )

    if response.status_code == 200:
        result = response.json()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Pipeline executed successfully!",
            "download_url": result.get("download"),
            "folder_id": folder_id
        })
    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Error submitting the task.",
            "folder_id": folder_id
        })
