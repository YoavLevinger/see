# GitHub Code Complexity Evaluator
# This script clones a GitHub repo, extracts code and git metrics,
# computes a normalized complexity score, prints, and exports to CSV.

import os
import subprocess
import shutil
import tempfile
import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
from pydriller import Repository
from pprint import pprint

# Helper function to clone a GitHub repository
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Analyze code complexity using Radon
def analyze_code_metrics(repo_path):
    code_metrics = {
        "loc": 0,
        "cyclomatic_complexity": 0,
        "halstead_volume": 0,
        "function_count": 0
    }
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):  # Adjust for other languages if needed
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        raw_metrics = analyze(code)
                        cc_metrics = cc_visit(code)
                        hv_metrics = h_visit(code)

                        code_metrics["loc"] += raw_metrics.loc
                        code_metrics["cyclomatic_complexity"] += sum(m.complexity for m in cc_metrics)
                        code_metrics["halstead_volume"] += hv_metrics.total.volume
                        code_metrics["function_count"] += len(cc_metrics)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return code_metrics

# Analyze Git metrics using PyDriller
def analyze_git_metrics(repo_path):
    git_metrics = {
        "commits": 0,
        "contributors": set(),
        "lines_added": 0,
        "lines_deleted": 0
    }
    for commit in Repository(repo_path).traverse_commits():
        git_metrics["commits"] += 1
        git_metrics["contributors"].add(commit.author.email)
        git_metrics["lines_added"] += commit.insertions
        git_metrics["lines_deleted"] += commit.deletions
    git_metrics["contributors"] = len(git_metrics["contributors"])
    return git_metrics

# Normalize metrics for fair comparison
def normalize_metrics(metrics, max_values):
    normalized = {}
    for key in metrics:
        normalized[key] = metrics[key] / max_values[key] if max_values[key] != 0 else 0
    return normalized

# Evaluate complexity using a weighted formula
def evaluate_complexity(normalized_metrics):
    score = (0.3 * normalized_metrics["loc"] +
             0.4 * normalized_metrics["cyclomatic_complexity"] +
             0.3 * normalized_metrics["halstead_volume"])
    return round(score, 4)

# Main orchestration function
def evaluate_repo_complexity(owner, repo_name):
    repo_path, temp_dir = clone_repo(owner, repo_name)

    # Step 1: Collect metrics
    code_metrics = analyze_code_metrics(repo_path)
    git_metrics = analyze_git_metrics(repo_path)

    # Combine metrics
    all_metrics = {**code_metrics, **git_metrics}

    # Step 2: Normalize static metrics only
    max_values = {
        "loc": 50000,
        "cyclomatic_complexity": 2000,
        "halstead_volume": 100000
    }
    normalized = normalize_metrics({
        "loc": code_metrics["loc"],
        "cyclomatic_complexity": code_metrics["cyclomatic_complexity"],
        "halstead_volume": code_metrics["halstead_volume"]
    }, max_values)

    # Step 3: Evaluate complexity
    complexity_score = evaluate_complexity(normalized)

    # Step 4: Store in CSV and Object
    df = pd.DataFrame([{
        "owner": owner,
        "repo": repo_name,
        **all_metrics,
        "complexity_score": complexity_score
    }])
    csv_path = os.path.join(temp_dir, f"{repo_name}_complexity_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Step 5: Print and return result
    print("\n--- Complexity Metrics ---")
    pprint(all_metrics)
    print(f"\nComplexity Score (0–1): {complexity_score}")
    print(f"\nCSV saved to: {csv_path}")

    # Cleanup cloned repo (optional)
    shutil.rmtree(temp_dir)

    return all_metrics, complexity_score

# Example call (replace with any public repo)
# evaluate_repo_complexity("psf", "requests")
