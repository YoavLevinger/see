import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import sys

# Add script directory to path so imports work
sys.path.append(os.path.dirname(__file__))

from get_similarity_repositories_sbert import get_top_k_similar_repos
from github_repo_complexity_evaluator_multiple_to_see import evaluate_multiple_repos

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Define input format
class EstimationRequest(BaseModel):
    description: str

# Define output format
class RepoEstimate(BaseModel):
    name: str
    similarity_score: float
    estimated_hours: float

class EstimationResponse(BaseModel):
    average_hours: float
    estimates: List[RepoEstimate]

@app.post("/estimate", response_model=EstimationResponse)
async def estimate_effort(req: EstimationRequest):
    logging.info("🔍 Received request to estimate effort for description: %s", req.description)

    # Ensure temp folder exists
    os.makedirs("./temp", exist_ok=True)

    # Create a temporary working directory for cloned repos
    with tempfile.TemporaryDirectory(dir="./temp") as temp_dir:
        logging.info("📁 Using temp directory: %s", temp_dir)

        try:
            # Step 1: Find similar repositories
            similar_repos = get_top_k_similar_repos(req.description, 5)
            logging.info("📦 Found %d similar repositories", len(similar_repos))

            if similar_repos.empty:
                raise HTTPException(status_code=404, detail="No similar repositories found.")

            # Step 2: Evaluate complexity and estimate effort
            repo_tuples = [(row["owner"], row["repo"]) for _, row in similar_repos.iterrows()]
            results = evaluate_multiple_repos(repo_tuples, temp_dir)

            # Create a lookup for similarity scores
            similarity_lookup = {
                (row["owner"], row["repo"]): row["similarity"] for _, row in similar_repos.iterrows()
            }

            # Step 3: Prepare response
            estimates = []
            total_hours = 0.0
            for repo in results:
                owner = repo.get("owner", "")
                name = repo.get("name", "")
                hours = repo.get("hours", 0.0)
                score = similarity_lookup.get((owner, name), 0.0)
                estimates.append(RepoEstimate(name=name, similarity_score=score, estimated_hours=hours))
                total_hours += hours

            avg_hours = total_hours / len(estimates)
            logging.info("✅ Effort estimation completed. Average hours: %.2f", avg_hours)

            return EstimationResponse(average_hours=avg_hours, estimates=estimates)

        except Exception as e:
            logging.exception("❌ Error during estimation: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
