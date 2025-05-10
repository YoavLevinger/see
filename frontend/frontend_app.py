from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import os
import uuid
import logging
from PyPDF2 import PdfReader
from fastapi.responses import FileResponse

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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/static/favicon.ico")

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
    performance_policy: UploadFile = File(None)
):
    logging.info(f"Handling submission for folder ID: {folder_id}")
    folder_path = os.path.join("generated-code", folder_id)
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Created or confirmed folder: {folder_path}")

    policy_map = {
        "security": security_policy,
        "accessibility": accessibility_policy,
        "performance": performance_policy,
    }

    combined_policy_text = ""

    for name, file in policy_map.items():
        if file:
            logging.info(f"Processing uploaded file for {name}: {file.filename}")
            if file.filename.endswith(".pdf"):
                policy_path = os.path.join(folder_path, f"policy.{name}.pdf")
                content = await file.read()
                if not content:
                    logging.warning(f"File {file.filename} is empty.")
                    continue

                with open(policy_path, "wb") as f:
                    f.write(content)
                logging.info(f"Uploaded {name} advisor policy to {policy_path}")

                try:
                    reader = PdfReader(policy_path)
                    num_pages = len(reader.pages)
                    logging.info(f"{name} PDF has {num_pages} pages.")
                    text_collected = False

                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            combined_policy_text += f"\n\n# {name.upper()} POLICY - PAGE {i + 1}\n{text.strip()}"
                            text_collected = True
                        else:
                            logging.warning(f"No text found on page {i + 1} of {policy_path}.")

                    if not text_collected:
                        logging.warning(f"No text extracted from any page in {policy_path}.")
                except Exception as e:
                    logging.warning(f"Failed to extract text from {policy_path}: {e}")
            else:
                logging.warning(f"Ignored {file.filename}: not a PDF.")
        else:
            logging.info(f"No {name} policy file uploaded.")

    if combined_policy_text.strip():
        policy_txt_path = os.path.join(folder_path, "policy.txt")
        with open(policy_txt_path, "w") as f:
            f.write(combined_policy_text)
        logging.info(f"Wrote combined policy text to {policy_txt_path}")
    else:
        logging.warning("No text extracted from any policy PDF.")

    # Send software description to main pipeline
    logging.info("Sending description to pipeline API...")
    try:
        response = requests.post(
            "http://localhost:8080/process",
            json={"description": description, "folder_id": folder_id}
        )
    except Exception as e:
        logging.error(f"Exception while calling pipeline API: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"Failed to reach pipeline API: {e}",
            "folder_id": folder_id
        })

    if response.status_code == 200:
        result = response.json()
        download_url = result.get("download")
        logging.info(f"Pipeline executed successfully. Download link: {download_url}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Pipeline executed successfully!",
            "download_url": download_url,
            "folder_id": folder_id
        })
    else:
        logging.error(f"Error submitting description: {response.status_code} {response.text}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Error submitting the task.",
            "folder_id": folder_id
        })
