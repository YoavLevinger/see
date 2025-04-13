
# github_repo_complexity_evaluator_multiple_to_see.py
# Author: Yoav Levinger (refactored)
# Purpose: Evaluate complexity and effort estimation of local and GitHub-hosted codebases
# using the academic PERT-based model built on LOC, CC, Halstead Volume, and AST Depth.

import os
import tempfile
import shutil
import subprocess
import pandas as pd
import sys
from contextlib import redirect_stdout
from tabulate import tabulate

from backend.sbert_complexity_estimator.code_effort_estimator import CodeMetrics, EffortEstimator


# Clone a GitHub repository and return its path
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    print(f"📦 Cloning {repo_url} to {repo_path}")
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Analyze a GitHub repo and return effort estimation using academic model
def evaluate_repo_complexity(owner, repo_name, complexity_mode="academic"):
    print(f"🔍 Evaluating repository: {owner}/{repo_name}")

    repo_path, temp_dir = clone_repo(owner, repo_name)

    try:
        metrics = CodeMetrics(repo_path)
        metrics.analyze()
        estimator = EffortEstimator(metrics)
        c_comp = estimator.calculate_composite_complexity()
        effort_result = estimator.calculate_effort(c_comp)

        result = {
            "owner": owner,
            "repo": repo_name,
            "C_comp": round(c_comp, 3),
            "complexity_mode": complexity_mode,
            "Optimistic": effort_result["Optimistic"],
            "Most_Likely": effort_result["Most Likely"],
            "Pessimistic": effort_result["Pessimistic"],
            "Effort (days)": effort_result["Effort (days)"],
            "hours": round(effort_result["Effort (days)"] * 8, 2)
        }

        print("\n📊 Academic PERT-Based Estimation")
        print("Formula: Effort = PERT(0.85 * C_comp, C_comp, 2.0 * C_comp)")
        print(tabulate(result.items(), headers=["Metric", "Value"], tablefmt="github"))

        return result

    finally:
        shutil.rmtree(temp_dir)

# Analyze local code folder (e.g., generated code) using same logic
def evaluate_codebase(local_path, complexity_mode="academic", is_local=True):
    print(f"🗂️ Analyzing local codebase: {local_path}")
    metrics = CodeMetrics(local_path)
    metrics.analyze()
    estimator = EffortEstimator(metrics)
    c_comp = estimator.calculate_composite_complexity()
    effort_result = estimator.calculate_effort(c_comp)

    result = {
        "owner": "local",
        "repo": os.path.basename(local_path),
        "C_comp": round(c_comp, 3),
        "complexity_mode": complexity_mode,
        "Optimistic": effort_result["Optimistic"],
        "Most_Likely": effort_result["Most Likely"],
        "Pessimistic": effort_result["Pessimistic"],
        "Effort (days)": effort_result["Effort (days)"],
        "estimated_effort_hours": round(effort_result["Effort (days)"] * 8, 2),
        "effort_details": effort_result
    }

    print("\n📊 Local Codebase Estimation")
    print(tabulate(result.items(), headers=["Metric", "Value"], tablefmt="github"))
    return result

# Batch evaluator for list of (owner, repo) tuples
def evaluate_multiple_repos(repo_list, temp_dir, log_file="batch_output.log", complexity_mode="academic"):
    print(f"🔁 Batch evaluating {len(repo_list)} GitHub repositories...")
    results = []

    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, data): [f.write(data) for f in self.files]
        def flush(self): [f.flush() for f in self.files]

    with open(log_file, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            for owner, repo in repo_list:
                try:
                    result = evaluate_repo_complexity(owner, repo, complexity_mode=complexity_mode)
                    results.append(result)
                except Exception as e:
                    print(f"⚠️ Error evaluating {owner}/{repo}: {e}")

            df = pd.DataFrame(results)
            output_csv = os.path.join(temp_dir, "effort_estimates.csv")
            df.to_csv(output_csv, index=False)
            print(f"✅ Saved results to {output_csv}")

            return [
                {"name": row["repo"], "hours": row["hours"], "owner": row["owner"]}
                for _, row in df.iterrows()
            ]
