from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import uuid
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

# Setup logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "main-controller.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)

class DescriptionInput(BaseModel):
    description: str

@app.post("/process")
def handle_description(input: DescriptionInput):
    description = input.description
    folder_id = str(uuid.uuid4())
    logger.info(f"Processing new input: {folder_id}")

    # Step 1: Get subtasks
    try:
        response = requests.post("http://localhost:8001/split", json={"description": description})
        response.raise_for_status()
        subtasks_data = response.json()
        subtasks = subtasks_data.get("subtasks", [])
        dev_subtasks = subtasks_data.get("dev_subtasks", [])
        logger.info(f"Subtasks: {len(subtasks)} | Dev subtasks: {len(dev_subtasks)}")
    except Exception as e:
        logger.error(f"Failed to split subtasks: {e}")
        return {"status": "error", "message": "Failed to split subtasks"}

    # Step 2: Generate code for dev subtasks in parallel
    def post_generate(subtask):
        logger.info(f"Submitting code generation for: {subtask}")
        try:
            res = requests.post("http://localhost:8002/generate", json={"subtask": subtask, "folder": folder_id})
            res.raise_for_status()
            logger.info(f"✅ Code generated for: {subtask}")
        except Exception as e:
            logger.error(f"❌ Code generation failed for: {subtask} | Error: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(post_generate, task) for task in dev_subtasks]
        for future in as_completed(futures):
            pass  # Results already logged inside function

    # Step 3: Notify Tool X
    try:
        payload = {
            "folder": folder_id,
            "description": description,
            "subtasks": subtasks
        }
        response = requests.post("http://localhost:8003/handle", json=payload)
        response.raise_for_status()
        logger.info("Tool X notified successfully.")
    except Exception as e:
        logger.warning(f"Tool X notification failed: {e}")

    # Step 4: Create policy file
    try:
        policy_text = "Ensure user input is validated. Avoid hardcoding credentials. Use secure file handling."
        policy_path = os.path.join("generated-code", folder_id, "policy.txt")
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        with open(policy_path, "w") as f:
            f.write(policy_text)
        logger.info("Default policy file written.")
    except Exception as e:
        logger.warning(f"Failed to write policy file: {e}")

    # Step 5: Effort estimation from past projects

    # Step 5B: Hybrid SBERT + Local Code Effort Estimation
    combined_effort = {}
    try:
        local_folder = os.path.join("generated-code", folder_id)
        combined_resp = requests.post(
            "http://localhost:8007/estimate-all",
            json={
                "description": description,
                "local_folder_path": local_folder
            }
        )
        if combined_resp.ok:
            combined_effort = combined_resp.json()
            logger.info("✅ Combined effort estimation retrieved.")
        else:
            logger.warning("⚠️ Combined effort estimation service returned error.")
    except Exception as e:
        logger.exception("❌ Failed to call combined effort estimator: %s", str(e))

    # After all code files are generated
    # effort_table = {}
    # try:
    #     resp = requests.post("http://localhost:8006/estimate", json={"description": description})
    #     if resp.ok:
    #         raw_effort = resp.json()
    #         effort_table = {
    #             "repositories": [
    #                 {"name": e["name"], "hours": e["estimated_hours"]}
    #                 for e in raw_effort.get("estimates", [])
    #             ],
    #             "average_time": raw_effort.get("average_hours")
    #         }
    #         logger.info("✅ Transformed effort estimation for document creator.")
    #     else:
    #         logger.warning("⚠️ Effort estimation request failed.")
    # except Exception as e:
    #     logger.exception("❌ Failed to call sbert-complexity-estimator: %s", str(e))

    # Step 6: Create documentation
    try:
        # doc_payload = {
        #     "folder_id": folder_id,
        #     "description": description,
        #     "subtasks": subtasks,
        #     "dev_subtasks": dev_subtasks,
        #     "policy_texts": {"default": policy_text},  # load policy
        #     "effort_table": effort_table,
        #     "expert_advice": {}  # or real advice if available
        # }
        doc_payload = {
            "folder_id": folder_id,
            "description": description,
            "subtasks": subtasks,
            "dev_subtasks": dev_subtasks,
            "policy_texts": {"default": policy_text},
            # "effort_table": effort_table,
            "expert_advice": {},
            "combined_effort": combined_effort  # ✅ NEW ENTRY
        }
        response = requests.post("http://localhost:8004/create", json=doc_payload)
        response.raise_for_status()
        logger.info("Document creator triggered.")
    except Exception as e:
        logger.error(f"Failed to trigger document creation: {e}")

    return {
        "status": "done",
        "folder": folder_id,
        "download": f"http://localhost:8004/download/{folder_id}"
    }
