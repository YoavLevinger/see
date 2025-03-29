from fastapi import FastAPI, Response
from backend.shared.models import DocRequest
import os
import markdown
import requests
from weasyprint import HTML
import logging

app = FastAPI()

# Set up logging to logs/
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "document-creator.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

@app.post("/create")
def create_document(req: DocRequest):
    folder_path = os.path.join("generated-code", req.folder)
    os.makedirs(folder_path, exist_ok=True)
    md_path = os.path.join(folder_path, "summary.md")
    pdf_path = os.path.join(folder_path, "summary.pdf")

    logging.info(f"Creating document for folder: {req.folder}")

    md_content = f"""# 📝 Software Project Report

## 📘 Description
{req.description}

## 🔨 Subtasks
"""
    for i, task in enumerate(req.subtasks, 1):
        md_content += f"{i}. {task}\n"

    md_content += "\n## 📂 Code Files"
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                md_content += f"\n\n### `{file}`\n"
                with open(os.path.join(folder_path, file), "r") as code_file:
                    md_content += "\n```python\n" + code_file.read() + "\n```"

    # Get expert advice
    policy_path = os.path.join(folder_path, "policy.txt")
    recommendations = []
    if os.path.exists(policy_path):
        with open(policy_path, "r") as p:
            policy = p.read()
        logging.info("Sending code for expert advisor analysis...")
        resp = requests.post("http://localhost:8005/advise", json={"folder": req.folder, "policy": policy})
        if resp.status_code == 200:
            recommendations = resp.json().get("recommendations", [])
            logging.info(f"Received {len(recommendations)} recommendations.")
        else:
            logging.warning("Advisor service returned non-200 status.")
    else:
        logging.info("No policy file found. Skipping expert recommendations.")

    md_content += "\n\n## 🧠 Expert Recommendations\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md_content += f"{i}. {rec}\n"
    else:
        md_content += "No recommendations were generated."

    # Write Markdown file
    with open(md_path, "w") as f:
        f.write(md_content)
    logging.info(f"Markdown written to {md_path}")

    # Convert to PDF
    html = markdown.markdown(md_content, extensions=['fenced_code'])
    HTML(string=html).write_pdf(pdf_path)
    logging.info(f"PDF written to {pdf_path}")

    return {"status": "document created", "pdf": pdf_path}

@app.get("/download/{folder_id}")
def download_pdf(folder_id: str):
    pdf_path = os.path.join("generated-code", folder_id, "summary.pdf")
    if not os.path.exists(pdf_path):
        logging.warning(f"PDF not found for folder: {folder_id}")
        return Response(status_code=404)

    with open(pdf_path, "rb") as f:
        logging.info(f"Serving PDF for folder: {folder_id}")
        return Response(f.read(), media_type="application/pdf")
