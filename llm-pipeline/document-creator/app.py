from fastapi import FastAPI, Response
from shared.models import DocRequest
import os
import markdown
import pdfkit

app = FastAPI()

@app.post("/create")
def create_document(req: DocRequest):
    folder_path = os.path.join("generated-code", req.folder)
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

    # Write Markdown file
    with open(md_path, "w") as f:
        f.write(md_content)

    # Convert to PDF
    html = markdown.markdown(md_content, extensions=['fenced_code'])
    pdfkit.from_string(html, pdf_path)

    return {"status": "document created", "pdf": pdf_path}

@app.get("/download/{folder_id}")
def download_pdf(folder_id: str):
    pdf_path = os.path.join("generated-code", folder_id, "summary.pdf")
    if not os.path.exists(pdf_path):
        return Response(status_code=404)

    with open(pdf_path, "rb") as f:
        return Response(f.read(), media_type="application/pdf")