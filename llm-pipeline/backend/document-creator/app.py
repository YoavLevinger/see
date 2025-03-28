from fastapi import FastAPI, Response
from backend.shared.models import DocRequest
import os
import markdown
import requests
from weasyprint import HTML

app = FastAPI()

@app.post("/create")
def create_document(req: DocRequest):
    folder_path = os.path.join("backend/generated-code", req.folder)
    os.makedirs(folder_path, exist_ok=True)
    md_path = os.path.join(folder_path, "summary.md")
    pdf_path = os.path.join(folder_path, "summary.pdf")

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
        resp = requests.post("http://localhost:8005/advise", json={"folder": req.folder, "policy": policy})
        if resp.status_code == 200:
            recommendations = resp.json().get("recommendations", [])

    md_content += "\n\n## 🧠 Expert Recommendations\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            md_content += f"{i}. {rec}\n"
    else:
        md_content += "No recommendations were generated."

    # Write Markdown file
    with open(md_path, "w") as f:
        f.write(md_content)

    # Convert to PDF
    html = markdown.markdown(md_content, extensions=['fenced_code'])
    # pdfkit.from_string(html, pdf_path)
    HTML(string=html).write_pdf(pdf_path)

    return {"status": "document created", "pdf": pdf_path}

@app.get("/download/{folder_id}")
def download_pdf(folder_id: str):
    pdf_path = os.path.join("backend/generated-code", folder_id, "summary.pdf")
    if not os.path.exists(pdf_path):
        return Response(status_code=404)

    with open(pdf_path, "rb") as f:
        return Response(f.read(), media_type="application/pdf")
