from fastapi import FastAPI, Response
from backend.shared.models import DocRequest
import os
import markdown
import requests
from weasyprint import HTML
import logging

app = FastAPI()

# Logging setup
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

    logging.info(f"Creating document in {folder_path}")

    md_content = f"""# 📝 Software Project Report

## 📘 Description
{req.description}

## 🧭 All Project Subtasks
"""
    for i, task in enumerate(req.subtasks, 1):
        md_content += f"{i}. {task}\n"

    md_content += "\n## 🔨 Development Code Subtasks\n"
    for i, task in enumerate(req.dev_subtasks, 1):
        md_content += f"{i}. {task}\n"

    md_content += "\n## 📂 Code Files"
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".py"):
                md_content += f"\n\n### `{file}`\n"
                try:
                    with open(os.path.join(folder_path, file), "r") as code_file:
                        md_content += "\n```python\n" + code_file.read() + "\n```"
                except Exception as e:
                    logging.warning(f"Could not read file {file}: {e}")

    # Expert recommendations
    policy_path = os.path.join(folder_path, "policy.txt")
    recommendations = []
    if os.path.exists(policy_path):
        try:
            with open(policy_path, "r") as p:
                policy = p.read()
            resp = requests.post("http://localhost:8005/advise", json={"folder": req.folder, "policy": policy})
            if resp.status_code == 200:
                recommendations = resp.json().get("recommendations", [])
        except Exception as e:
            logging.warning(f"Advisor failed: {e}")

    md_content += "\n\n## 🧠 Expert Recommendations\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md_content += f"{i}. {rec}\n"
    else:
        md_content += "No recommendations were generated."

    try:
        with open(md_path, "w") as f:
            f.write(md_content)
    except Exception as e:
        logging.error(f"Failed to write Markdown file: {e}")

    try:
        html = markdown.markdown(md_content, extensions=['fenced_code'])
        HTML(string=html).write_pdf(pdf_path)
        logging.info(f"Generated PDF at {pdf_path}")
    except Exception as e:
        logging.error(f"PDF generation failed: {e}")

    return {"status": "document created", "pdf": pdf_path}

@app.get("/download/{folder_id}")
def download_pdf(folder_id: str):
    pdf_path = os.path.join("generated-code", folder_id, "summary.pdf")
    if not os.path.exists(pdf_path):
        return Response(status_code=404)

    with open(pdf_path, "rb") as f:
        return Response(f.read(), media_type="application/pdf")
