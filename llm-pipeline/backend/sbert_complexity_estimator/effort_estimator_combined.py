
import os
import tempfile
import shutil
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.sbert_complexity_estimator.get_similarity_repositories_sbert import get_top_k_similar_repos
from backend.sbert_complexity_estimator.github_repo_complexity_evaluator_multiple_to_see import evaluate_multiple_repos, evaluate_codebase

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Request/Response Models ===

class CombinedEstimationRequest(BaseModel):
    description: str
    local_folder_path: str

class RepoEstimate(BaseModel):
    name: str
    owner: str
    hours: float
    description: str

class CombinedEstimationResponse(BaseModel):
    github_repositories: List[RepoEstimate]
    github_average: float
    local_effort: Dict[str, Any]

# === Helper: Outlier Removal ===

def remove_outliers(results: List[Dict], key: str = "hours") -> List[Dict]:
    values = [r[key] for r in results if key in r]
    if not values:
        return results
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [r for r in results if lower <= r[key] <= upper]
    return filtered if filtered else results  # fallback if all removed

# === Route: Combined Estimation ===

@app.post("/estimate-all", response_model=CombinedEstimationResponse)
async def estimate_all_effort(req: CombinedEstimationRequest):
    logging.info("📥 Received combined estimation request.")
    print(f"\n[INPUT] Description: {req.description}")
    print(f"[INPUT] Local Folder Path: {req.local_folder_path}")

    with tempfile.TemporaryDirectory(dir="./temp") as temp_dir:
        try:
            # Step 1: Retrieve similar repos
            logging.info("🔍 Step 1: Finding top-k similar GitHub repositories...")
            similar_df = get_top_k_similar_repos(req.description, top_k=5)
            if similar_df.empty:
                raise HTTPException(status_code=404, detail="No similar repositories found.")
            print("[INFO] Similar repos retrieved:")
            print(similar_df[["owner", "repo", "description"]].to_string(index=False))

            repo_tuples = [(row["owner"], row["repo"]) for _, row in similar_df.iterrows()]
            desc_lookup = {(row["owner"], row["repo"]): row["description"] for _, row in similar_df.iterrows()}

            # Step 2: Evaluate each similar repo
            logging.info("🧪 Step 2: Cloning and evaluating each similar repository...")
            all_results = evaluate_multiple_repos(repo_tuples, temp_dir)

            for r in all_results:
                r["description"] = desc_lookup.get((r["owner"], r["name"]), "")

            print("[RESULTS] Raw effort estimations (before outlier removal):")
            for r in all_results:
                print(f"- {r['owner']}/{r['name']}: {r.get('hours', 0)} hours")

            # Step 3: Outlier removal
            logging.info("📉 Step 3: Removing outliers from estimations...")
            filtered_results = remove_outliers(all_results)
            github_avg = round(np.mean([r["hours"] for r in filtered_results]), 2)

            print("[RESULTS] Filtered effort estimations:")
            for r in filtered_results:
                print(f"✓ {r['owner']}/{r['name']}: {r['hours']} hours")
            print(f"[SUMMARY] Average GitHub Effort (filtered): {github_avg} hours")

            # Step 4: Analyze local folder
            logging.info("💻 Step 4: Analyzing generated code folder...")
            local_result = evaluate_codebase(req.local_folder_path, complexity_mode="power", is_local=True)
            print("[RESULTS] Local effort estimation:")
            for k, v in local_result.items():
                print(f"  - {k}: {v}")

            return {
                "github_repositories": filtered_results,
                "github_average": github_avg,
                "local_effort": local_result
            }

        except Exception as e:
            logging.exception("❌ Estimation failed: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
